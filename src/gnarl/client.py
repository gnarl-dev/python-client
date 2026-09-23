"""The client for a Gnarl node.

A node is a peer in a decentralized search fabric rather than a coordinator, so
there is no cluster endpoint to point at: you talk to a node, and it answers
for the mesh. Any node will do.

    from gnarl import Client, query as q

    with Client("http://localhost:8080") as c:
        res = c.search("places", q.geo_distance("location", -33.8688, 151.2093, 1_000))

There is an :class:`AsyncClient` with the same surface.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from dataclasses import field as _dc_field
from typing import Any
from urllib.parse import quote

import httpx

from . import _models as m
from .errors import GnarlError, IncompleteResult, NotFound, error_from_response

__all__ = [
    "Client",
    "AsyncClient",
    "SearchResult",
    "BulkDoc",
    "NodeStatus",
    "NodeVersion",
    "DEFAULT_TIMEOUT",
    "failed_items",
]

#: Applied when no ``timeout`` is given.
DEFAULT_TIMEOUT = 30.0

_USER_AGENT = "gnarl-python"


# ─── Result types ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SearchResult:
    """The result of a search."""

    #: Matching documents, best first.
    hits: list[m.Hit]

    #: Hit-count summary. Read ``total.relation`` before ``total.value``: the
    #: default is a LOWER BOUND, not an exact count. :attr:`total_is_exact`
    #: says which.
    total: m.TotalHits

    #: True when the result may be incomplete because some claims did not
    #: answer.
    partial: bool

    #: Auditable claim-level completeness of this search.
    coverage: m.SearchCoverage

    #: Query execution time in milliseconds, as the node measured it.
    took: int

    #: Execution timing and per-peer fan-out. Present only when the request
    #: asked for it with ``profile=True``.
    profile: m.SearchProfile | None = None

    @property
    def total_is_exact(self) -> bool:
        """Whether :attr:`total` is a count rather than a lower bound."""
        return self.total.relation is m.Relation1.eq

    def sources(self) -> list[dict[str, Any]]:
        """The ``_source`` of every hit that has one."""
        return [h.field_source for h in self.hits if h.field_source is not None]

    def __len__(self) -> int:
        return len(self.hits)

    def __iter__(self) -> Iterator[m.Hit]:
        return iter(self.hits)


@dataclass
class BulkDoc:
    """A document with an explicit id, for a bulk request where ids matter."""

    id: str
    document: Mapping[str, Any]


@dataclass(frozen=True)
class NodeStatus:
    """A node's self-report.

    The fields mirror the description exactly. There is deliberately no
    ``version`` here — that is a separate endpoint (:meth:`Client.version`),
    and an invented field would read as ``None`` forever without ever failing.
    """

    #: The node's 64-char hex identity.
    node_id: str

    #: Mesh scope this node is running in: private, public, lan, dev-mesh or
    #: single-node.
    mode: str

    #: How many peers this node knows, and how many it can currently reach.
    #: They differ during a partition, which is the point.
    peers: int
    reachable_peers: int

    #: Claim units this node holds, and how many can answer right now.
    claims: int
    serving_ready: int

    #: Claims whose storage proof has been verified.
    proof_verified: int

    #: Absent only for an ephemeral in-memory node.
    data_dir: str | None = None

    raw: dict[str, Any] = _dc_field(default_factory=dict, repr=False)

    @classmethod
    def _parse(cls, body: dict[str, Any]) -> NodeStatus:
        return cls(
            node_id=body.get("node_id", ""),
            mode=body.get("mode", ""),
            peers=body.get("peers", 0),
            reachable_peers=body.get("reachable_peers", 0),
            claims=body.get("claims", 0),
            serving_ready=body.get("serving_ready", 0),
            proof_verified=body.get("proof_verified", 0),
            data_dir=body.get("data_dir"),
            raw=body,
        )


@dataclass(frozen=True)
class NodeVersion:
    """The node's build identity, from a different endpoint than status."""

    version: str
    commit: str | None = None


def failed_items(result: m.BulkIndexResponse) -> list[m.BulkItemResult]:
    """The items in a bulk result that did not succeed.

    Bulk answers 200 with individual failures, so a caller who checks only the
    HTTP status loses writes without seeing an error. This makes the check a
    one-liner, so there is no excuse to skip it.
    """
    if not result.errors:
        return []
    return [it for it in result.items if it.error is not None]


# ─── Request plumbing, shared by both clients ───────────────────────────────


def _normalize_base_url(addr: str) -> str:
    """Resolve the base URL.

    A scheme-less address becomes **https**. A node serves TLS by default, and
    defaulting to http would silently downgrade a caller who wrote
    ``search.example.com``.
    """
    if not addr:
        raise ValueError("gnarl: empty address")
    if "://" not in addr:
        addr = "https://" + addr
    parsed = httpx.URL(addr)
    if not parsed.host:
        raise ValueError(f"gnarl: {addr!r}: no host")
    return str(parsed).rstrip("/")


def _esc(segment: str) -> str:
    """Escape one path segment. Index names and document ids reach the URL
    directly, and a document id is caller data that may contain a slash."""
    return quote(segment, safe="")


def _document_body(doc: Mapping[str, Any], doc_id: str | None) -> dict[str, Any]:
    """Merge a document with an optional ``_id``.

    The wire shape puts document fields at the TOP LEVEL beside ``_id`` rather
    than under a wrapper, so the id cannot be attached by nesting.
    """
    if doc is None:
        raise ValueError("gnarl: document is None")
    if not isinstance(doc, Mapping):
        raise TypeError(
            f"gnarl: a document must be a mapping, got {type(doc).__name__}"
        )
    body = dict(doc)
    if doc_id is not None:
        if doc_id == "":
            raise ValueError("gnarl: empty document id (pass None to have one assigned)")
        body["_id"] = doc_id
    return body


def _create_index_body(schema: m.IndexSchema) -> dict[str, Any]:
    """Serialize a schema, sending only what the caller actually declared.

    ``exclude_unset`` and not just ``exclude_none``: several field options
    carry non-None defaults that apply to one field type only, so without it a
    plain ``text`` field would arrive carrying ``distance_metric`` and
    ``quantization``. The node ignores them, but a request should say what was
    asked for.
    """
    return m.CreateIndexRequest(schema=schema).model_dump(
        mode="json", by_alias=True, exclude_none=True, exclude_unset=True
    )


def _search_body(
    query: m.Query | None,
    size: int | None,
    from_: int | None,
    sort: Sequence[Any] | None,
    search_after: Sequence[Any] | None,
    source: bool | list[str] | None,
    track_total_hits: bool | None,
    profile: bool,
    verify: bool,
    deadline_ms: int | None,
) -> dict[str, Any]:
    # Built as a mapping keyed by WIRE names, then validated, so the request
    # carries only what the caller actually asked for. Anything absent here
    # stays unset, and `exclude_unset` keeps it off the wire — which matters
    # because several fields have a non-None default (`from` is 0, `size` is
    # 10) that would otherwise be sent back to the node as though the caller
    # had chosen it.
    payload: dict[str, Any] = {}
    if query is not None:
        payload["query"] = query
    if size is not None:
        payload["size"] = size
    if from_:
        payload["from"] = from_
    if sort is not None:
        payload["sort"] = list(sort)
    if search_after is not None:
        payload["search_after"] = list(search_after)
    if source is not None:
        payload["_source"] = source
    if track_total_hits is not None:
        payload["track_total_hits"] = track_total_hits
    # `profile` and `verify` each make the node do work it otherwise skips, so
    # an unconditional `false` would misstate intent even though it costs the
    # same on the wire.
    if profile:
        payload["profile"] = True
    if verify:
        payload["verify"] = True
    if deadline_ms:
        payload["scope"] = {"deadline_ms": deadline_ms}

    # Validated rather than sent raw: a size past the node's ceiling, or a
    # deadline below 1 ms, should fail here naming the field rather than as a
    # 400 the caller has to map back to their own call.
    req = m.SearchRequest.model_validate(payload)
    return req.model_dump(
        mode="json", by_alias=True, exclude_none=True, exclude_unset=True
    )


def _to_search_result(body: dict[str, Any]) -> SearchResult:
    parsed = m.SearchResponse.model_validate(body)
    return SearchResult(
        hits=parsed.hits.hits,
        total=parsed.hits.total,
        partial=parsed.partial,
        coverage=parsed.coverage,
        took=parsed.took,
        profile=parsed.profile,
    )


def _check_complete(result: SearchResult) -> None:
    """Raise if a completeness-requiring search did not get one.

    Both conditions matter. ``partial`` is the node's own verdict; the claim
    arithmetic catches the case where it was not set but a claim went unserved
    anyway. The point of asking for completeness is not to trust one flag.
    """
    if result.partial or result.coverage.served_claims < result.coverage.expected_claims:
        raise IncompleteResult(result)


def _headers(token: str | None, user_agent: str, has_body: bool) -> dict[str, str]:
    h = {"Accept": "application/json", "User-Agent": user_agent}
    if has_body:
        h["Content-Type"] = "application/json"
    if token:
        h["Authorization"] = "Bearer " + token
    return h


def _decode(resp: httpx.Response) -> Any:
    if not resp.content:
        return None
    try:
        return json.loads(resp.content)
    except ValueError as exc:
        raise GnarlError(
            type="invalid_response",
            reason=(
                f"{resp.request.method} {resp.request.url.path}: "
                f"response was not JSON: {exc}"
            ),
            status=resp.status_code,
        ) from exc


def _raise_for_status(resp: httpx.Response) -> None:
    if resp.status_code < 200 or resp.status_code >= 300:
        raise error_from_response(resp.status_code, resp.content, resp.headers)


# ─── Sync client ────────────────────────────────────────────────────────────


class Client:
    """Talks to one Gnarl node. Safe for concurrent use across threads.

    :param addr: the node's address. A missing scheme means https.
    :param token: an RBAC capability token, sent as a bearer token. A node with
        RBAC enabled exempts loopback callers, so a local node usually needs no
        token; a remote one always does.
    :param timeout: seconds, applied to each request.
    :param verify: TLS verification. A node generates a self-signed certificate
        on first run, so ``verify=False`` is the switch you reach for against a
        development node. It disables the protection TLS exists to provide —
        never set it against a node you did not start yourself. A CA bundle
        path is the right answer for anything else.
    :param http_client: supply your own ``httpx.Client`` for a custom
        transport, proxy, pool or tracing hook. ``timeout`` and ``verify`` are
        then yours to set.
    """

    def __init__(
        self,
        addr: str,
        *,
        token: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        verify: bool | str = True,
        user_agent: str = _USER_AGENT,
        http_client: httpx.Client | None = None,
    ) -> None:
        self._base = _normalize_base_url(addr)
        self._token = token
        self._user_agent = user_agent
        self._owns_http = http_client is None
        self._http = http_client or httpx.Client(timeout=timeout, verify=verify)

    # -- lifecycle --

    def close(self) -> None:
        """Close the underlying connection pool, if this client owns it."""
        if self._owns_http:
            self._http.close()

    def __enter__(self) -> Client:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # -- transport --

    def _do(
        self, method: str, path: str, body: Any = None, *, want_json: bool = True
    ) -> Any:
        content = None if body is None else json.dumps(body).encode()
        try:
            resp = self._http.request(
                method,
                self._base + path,
                content=content,
                headers=_headers(self._token, self._user_agent, content is not None),
            )
        except httpx.HTTPError as exc:
            raise GnarlError(
                type="transport_error", reason=f"{method} {path}: {exc}"
            ) from exc
        _raise_for_status(resp)
        return _decode(resp) if want_json else None

    # -- indexes --

    def create_index(self, name: str, schema: m.IndexSchema) -> None:
        """Create an index with the given schema.

        Raises :class:`~gnarl.errors.AlreadyExists` if it is already there.
        """
        if not name:
            raise ValueError("gnarl: create_index: empty index name")
        body = _create_index_body(schema)
        self._do("PUT", f"/v1/indexes/{_esc(name)}", body, want_json=False)

    def delete_index(self, name: str) -> None:
        """Remove an index and everything in it."""
        self._do("DELETE", f"/v1/indexes/{_esc(name)}", want_json=False)

    def index_exists(self, name: str) -> bool:
        """Whether the index exists.

        This distinguishes "absent" from "could not tell": a transport failure
        or a 500 raises rather than returning False. Treating those as absent
        is how a caller ends up recreating — or deleting — live data.
        """
        try:
            self._do("GET", f"/v1/indexes/{_esc(name)}", want_json=False)
            return True
        except NotFound:
            return False

    def list_indexes(self) -> list[m.IndexMetadata]:
        """Every index this node can see.

        The endpoint is cursor-paginated and this follows it to the end. A
        caller who stops at the first page silently sees a prefix of their own
        data, which is exactly the bug the cursor exists to prevent — so the
        convenient method is the complete one, and :meth:`list_indexes_page` is
        there when you want the pages yourself.
        """
        out: list[m.IndexMetadata] = []
        cursor: str | None = None
        while True:
            page, cursor = self.list_indexes_page(cursor)
            out.extend(page)
            if not cursor:
                return out

    def list_indexes_page(
        self, after: str | None = None
    ) -> tuple[list[m.IndexMetadata], str | None]:
        """One page of indexes, and the cursor for the next (``None`` at the
        end)."""
        path = "/v1/indexes"
        if after:
            path += "?after=" + quote(after, safe="")
        raw = m.IndexListResponse.model_validate(self._do("GET", path))
        return raw.indexes, raw.next_after

    def get_schema(self, name: str) -> m.IndexSchema:
        """The index's current mapping."""
        body = self._do("GET", f"/v1/indexes/{_esc(name)}/_schema")
        return m.IndexSchema.model_validate(body)

    def count(self, name: str) -> int:
        """How many documents the index holds."""
        body = self._do("GET", f"/v1/indexes/{_esc(name)}/_count")
        return int(body["count"])

    # -- documents --

    def index_document(
        self, index: str, doc: Mapping[str, Any], *, id: str | None = None
    ) -> str:
        """Index one document, returning the id the node assigned or accepted.

        Pass ``id=None`` to have one generated.
        """
        body = _document_body(doc, id)
        raw = self._do("POST", f"/v1/indexes/{_esc(index)}/_doc", body)
        return m.IndexDocumentResponse.model_validate(raw).field_id

    def get_document(self, index: str, id: str) -> dict[str, Any]:
        """A document's stored fields.

        Raises :class:`~gnarl.errors.NotFound` when the document is absent.
        """
        raw = self._do("GET", f"/v1/indexes/{_esc(index)}/_doc/{_esc(id)}")
        return m.GetDocumentResponse.model_validate(raw).field_source

    def delete_document(self, index: str, id: str) -> None:
        """Remove one document by id."""
        self._do(
            "DELETE", f"/v1/indexes/{_esc(index)}/_doc/{_esc(id)}", want_json=False
        )

    def bulk(
        self, index: str, docs: Iterable[Mapping[str, Any] | BulkDoc]
    ) -> m.BulkIndexResponse:
        """Index many documents in one request.

        Accepts plain mappings, or :class:`BulkDoc` where you choose the ids.

        Check :func:`failed_items` before assuming the batch succeeded: a bulk
        request can return 200 with individual items failed, which is the most
        common way to lose writes silently.
        """
        body = _bulk_body(docs)
        raw = self._do("POST", f"/v1/indexes/{_esc(index)}/_bulk", body)
        return m.BulkIndexResponse.model_validate(raw)

    # -- search --

    def search(
        self,
        index: str,
        query: m.Query | None = None,
        *,
        size: int | None = None,
        from_: int | None = None,
        sort: Sequence[Any] | None = None,
        search_after: Sequence[Any] | None = None,
        source: bool | list[str] | None = None,
        track_total_hits: bool | None = None,
        require_complete: bool = False,
        profile: bool = False,
        verify: bool = False,
        deadline_ms: int | None = None,
    ) -> SearchResult:
        """Run a query against one index.

        :param require_complete: turn a partial result into an error. A search
            spans claims held by many peers and a node answers with whatever it
            could reach — the right default for interactive search and the
            wrong one for anything auditable. When set, an incomplete result
            raises :class:`~gnarl.errors.IncompleteResult`, which carries the
            partial result so you can still inspect it.
        :param verify: require tamper evidence. Every served claim must be
            PROVEN against an anchor the node holds independently of whoever
            served it; a claim nobody could check counts against completeness
            exactly as an unanswered one does. Distinct from
            ``require_complete``: that asks whether every claim ANSWERED, this
            asks whether every answer was PROVEN. Set both to demand both.
        :param profile: ask for execution timing and per-peer fan-out in
            :attr:`SearchResult.profile`.
        :param deadline_ms: how long a shard may take. A shard that misses it
            is skipped and reported in coverage, so a tight deadline degrades
            into an honest partial answer rather than an error. The node clamps
            this to its own timeout: you may ask for less, never more.
        :param from_: pagination offset. Prefer ``search_after`` past a few
            pages — every claim must collect ``from + size`` rows, so deep
            offsets cost more everywhere.
        """
        if not index:
            raise ValueError("gnarl: search: empty index name")
        body = _search_body(
            query, size, from_, sort, search_after, source,
            track_total_hits, profile, verify, deadline_ms,
        )
        raw = self._do("POST", f"/v1/indexes/{_esc(index)}/_search", body)
        result = _to_search_result(raw)
        if require_complete:
            _check_complete(result)
        return result

    # -- node --

    def version(self) -> NodeVersion:
        """The node's build version."""
        body = self._do("GET", "/v1/node/version")
        return NodeVersion(version=body.get("version", ""), commit=body.get("commit"))

    def status(self) -> NodeStatus:
        """The node's status.

        Needs no token even on an RBAC node: observability is deliberately
        ungated, so this keeps working during the incident you need it for.
        """
        return NodeStatus._parse(self._do("GET", "/v1/node/status"))

    def ping(self) -> None:
        """Whether the node answers. Status without the body; raises if not."""
        self._do("GET", "/v1/node/status", want_json=False)


def _bulk_body(docs: Iterable[Mapping[str, Any] | BulkDoc]) -> dict[str, Any]:
    encoded: list[dict[str, Any]] = []
    for i, d in enumerate(docs):
        try:
            if isinstance(d, BulkDoc):
                encoded.append(_document_body(d.document, d.id))
            else:
                encoded.append(_document_body(d, None))
        except (TypeError, ValueError) as exc:
            raise type(exc)(f"gnarl: bulk: document {i}: {exc}") from exc
    if not encoded:
        raise ValueError("gnarl: bulk: no documents")
    return {"documents": encoded}


# ─── Async client ───────────────────────────────────────────────────────────


class AsyncClient:
    """:class:`Client` over ``httpx.AsyncClient``. Same surface, awaited.

    Every parameter and every behaviour matches the sync client; see it for the
    documentation.
    """

    def __init__(
        self,
        addr: str,
        *,
        token: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        verify: bool | str = True,
        user_agent: str = _USER_AGENT,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base = _normalize_base_url(addr)
        self._token = token
        self._user_agent = user_agent
        self._owns_http = http_client is None
        self._http = http_client or httpx.AsyncClient(timeout=timeout, verify=verify)

    async def aclose(self) -> None:
        """Close the underlying connection pool, if this client owns it."""
        if self._owns_http:
            await self._http.aclose()

    async def __aenter__(self) -> AsyncClient:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    async def _do(
        self, method: str, path: str, body: Any = None, *, want_json: bool = True
    ) -> Any:
        content = None if body is None else json.dumps(body).encode()
        try:
            resp = await self._http.request(
                method,
                self._base + path,
                content=content,
                headers=_headers(self._token, self._user_agent, content is not None),
            )
        except httpx.HTTPError as exc:
            raise GnarlError(
                type="transport_error", reason=f"{method} {path}: {exc}"
            ) from exc
        _raise_for_status(resp)
        return _decode(resp) if want_json else None

    async def create_index(self, name: str, schema: m.IndexSchema) -> None:
        if not name:
            raise ValueError("gnarl: create_index: empty index name")
        body = _create_index_body(schema)
        await self._do("PUT", f"/v1/indexes/{_esc(name)}", body, want_json=False)

    async def delete_index(self, name: str) -> None:
        await self._do("DELETE", f"/v1/indexes/{_esc(name)}", want_json=False)

    async def index_exists(self, name: str) -> bool:
        try:
            await self._do("GET", f"/v1/indexes/{_esc(name)}", want_json=False)
            return True
        except NotFound:
            return False

    async def list_indexes(self) -> list[m.IndexMetadata]:
        out: list[m.IndexMetadata] = []
        cursor: str | None = None
        while True:
            page, cursor = await self.list_indexes_page(cursor)
            out.extend(page)
            if not cursor:
                return out

    async def list_indexes_page(
        self, after: str | None = None
    ) -> tuple[list[m.IndexMetadata], str | None]:
        path = "/v1/indexes"
        if after:
            path += "?after=" + quote(after, safe="")
        raw = m.IndexListResponse.model_validate(await self._do("GET", path))
        return raw.indexes, raw.next_after

    async def get_schema(self, name: str) -> m.IndexSchema:
        body = await self._do("GET", f"/v1/indexes/{_esc(name)}/_schema")
        return m.IndexSchema.model_validate(body)

    async def count(self, name: str) -> int:
        body = await self._do("GET", f"/v1/indexes/{_esc(name)}/_count")
        return int(body["count"])

    async def index_document(
        self, index: str, doc: Mapping[str, Any], *, id: str | None = None
    ) -> str:
        body = _document_body(doc, id)
        raw = await self._do("POST", f"/v1/indexes/{_esc(index)}/_doc", body)
        return m.IndexDocumentResponse.model_validate(raw).field_id

    async def get_document(self, index: str, id: str) -> dict[str, Any]:
        raw = await self._do("GET", f"/v1/indexes/{_esc(index)}/_doc/{_esc(id)}")
        return m.GetDocumentResponse.model_validate(raw).field_source

    async def delete_document(self, index: str, id: str) -> None:
        await self._do(
            "DELETE", f"/v1/indexes/{_esc(index)}/_doc/{_esc(id)}", want_json=False
        )

    async def bulk(
        self, index: str, docs: Iterable[Mapping[str, Any] | BulkDoc]
    ) -> m.BulkIndexResponse:
        body = _bulk_body(docs)
        raw = await self._do("POST", f"/v1/indexes/{_esc(index)}/_bulk", body)
        return m.BulkIndexResponse.model_validate(raw)

    async def search(
        self,
        index: str,
        query: m.Query | None = None,
        *,
        size: int | None = None,
        from_: int | None = None,
        sort: Sequence[Any] | None = None,
        search_after: Sequence[Any] | None = None,
        source: bool | list[str] | None = None,
        track_total_hits: bool | None = None,
        require_complete: bool = False,
        profile: bool = False,
        verify: bool = False,
        deadline_ms: int | None = None,
    ) -> SearchResult:
        if not index:
            raise ValueError("gnarl: search: empty index name")
        body = _search_body(
            query, size, from_, sort, search_after, source,
            track_total_hits, profile, verify, deadline_ms,
        )
        raw = await self._do("POST", f"/v1/indexes/{_esc(index)}/_search", body)
        result = _to_search_result(raw)
        if require_complete:
            _check_complete(result)
        return result

    async def version(self) -> NodeVersion:
        body = await self._do("GET", "/v1/node/version")
        return NodeVersion(version=body.get("version", ""), commit=body.get("commit"))

    async def status(self) -> NodeStatus:
        return NodeStatus._parse(await self._do("GET", "/v1/node/status"))

    async def ping(self) -> None:
        await self._do("GET", "/v1/node/status", want_json=False)
