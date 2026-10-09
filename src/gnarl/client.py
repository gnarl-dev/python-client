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
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, TypeVar
from urllib.parse import quote

import httpx

from . import _models as m
from ._wire import (
    BulkDoc,
    Entitlement,
    NodeStatus,
    NodeVersion,
    SearchResult,
    _activate_call,
    _bulk_body,
    _Call,
    _check_complete,
    _create_index_body,
    _decode,
    _document_body,
    _entitlement_call,
    _esc,
    _forcemerge_call,
    _get_policy_call,
    _headers,
    _normalize_base_url,
    _put_policy_call,
    _raise_for_status,
    _search_body,
    _to_search_result,
    failed_items,
)
from .errors import GnarlError, NotFound
from .memory import AsyncMemory, Memory
from .namespaces import AsyncNamespaces, Namespaces
from .snapshots import AsyncSnapshots, Snapshots

__all__ = [
    "Client",
    "AsyncClient",
    "SearchResult",
    "BulkDoc",
    "NodeStatus",
    "NodeVersion",
    "Entitlement",
    "DEFAULT_TIMEOUT",
    "failed_items",
]

T = TypeVar("T")

#: Applied when no ``timeout`` is given.
DEFAULT_TIMEOUT = 30.0

_USER_AGENT = "gnarl-python"


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

        #: Namespaces: many lightweight tenants over shared pools.
        self.namespaces = Namespaces(self)
        #: Agent memory: remember, recall, answer, ingest.
        self.memory = Memory(self)
        #: Backup and restore: repositories, snapshots, schedules, jobs.
        self.snapshots = Snapshots(self)

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

    def _call(self, call: _Call[T]) -> T:
        raw = self._do(call.method, call.path, call.body, want_json=call.want_json)
        return call.parse(raw)

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

    # -- entitlement --

    def entitlement(self) -> Entitlement:
        """What subscription this node holds: active, refused, unenforced or
        none. See :class:`Entitlement` for why refused is its own state."""
        return self._call(_entitlement_call())

    def activate_entitlement(self, key: str) -> Entitlement:
        """Activate a subscription from a pasted key.

        Takes the one-line ``gnarl-ent1.`` form from the account page, or the
        raw signed JSON. The node VERIFIES the key before storing it; a refusal
        raises :class:`~gnarl.ValidationError` whose ``reason`` says which
        failure it was — malformed, expired, or signed by a key this build does
        not trust. A node with no data directory raises
        :class:`~gnarl.Unavailable`. The node keeps its previous mesh scope
        until it restarts.
        """
        return self._call(_activate_call(key))

    # -- index maintenance --

    def forcemerge(
        self, index: str, *, max_num_segments: int | None = None
    ) -> m.V1IndexesNameForcemergePostResponse:
        """Merge an index's segments, down to ``max_num_segments`` per claim
        (node default 1). Expensive and I/O-heavy: a quiet-period operation.

        Merges only the claims THIS node holds; ``partial`` says when others
        exist elsewhere.
        """
        return self._call(_forcemerge_call(index, max_num_segments))

    def get_policy(self, index: str) -> m.IndexPolicy:
        """How far the index's data may travel: placement and replicas."""
        return self._call(_get_policy_call(index))

    def put_policy(
        self,
        index: str,
        *,
        placement: m.IndexPlacement | str | None = None,
        replication_factor: int | None = None,
    ) -> m.IndexPolicy:
        """Change the placement policy. Returns the policy now in force.

        ORIGIN ONLY: any node but the one that created the index answers
        :class:`~gnarl.Forbidden`. Omitted fields are unchanged, so this can
        narrow placement without restating a replication factor. Narrowing
        DROPS replicas already held, on the next anti-entropy cycle.
        """
        return self._call(_put_policy_call(index, placement, replication_factor))



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

        self.namespaces = AsyncNamespaces(self)
        self.memory = AsyncMemory(self)
        self.snapshots = AsyncSnapshots(self)

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

    async def _call(self, call: _Call[T]) -> T:
        raw = await self._do(call.method, call.path, call.body, want_json=call.want_json)
        return call.parse(raw)

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

    async def entitlement(self) -> Entitlement:
        return await self._call(_entitlement_call())

    async def activate_entitlement(self, key: str) -> Entitlement:
        return await self._call(_activate_call(key))

    async def forcemerge(
        self, index: str, *, max_num_segments: int | None = None
    ) -> m.V1IndexesNameForcemergePostResponse:
        return await self._call(_forcemerge_call(index, max_num_segments))

    async def get_policy(self, index: str) -> m.IndexPolicy:
        return await self._call(_get_policy_call(index))

    async def put_policy(
        self,
        index: str,
        *,
        placement: m.IndexPlacement | str | None = None,
        replication_factor: int | None = None,
    ) -> m.IndexPolicy:
        return await self._call(_put_policy_call(index, placement, replication_factor))

