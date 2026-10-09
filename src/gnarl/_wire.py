"""Wire plumbing and result types shared by every part of the client.

Nothing here talks to the network. It builds request bodies, parses response
bodies and describes calls, so the sync and async clients — and the groups
hung off them (``namespaces``, ``memory``, ``snapshots``) — run the same logic
and differ only in whether they await.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from dataclasses import field as _dc_field
from datetime import datetime, timezone
from typing import Any, Generic, TypeVar
from urllib.parse import quote, urlencode

import httpx

from . import _models as m
from .errors import GnarlError, IncompleteResult, error_from_response

T = TypeVar("T")

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




def _query_string(params: Mapping[str, Any]) -> str:
    """``?a=1&b=2`` for the parameters that are set, or ``""``."""
    present = {
        k: ("true" if v is True else "false" if v is False else v)
        for k, v in params.items()
        if v is not None
    }
    return "?" + urlencode(present) if present else ""


@dataclass(frozen=True)
class _Call(Generic[T]):
    """One request, described rather than performed.

    Every operation is built once as a ``_Call`` and executed by whichever
    client holds it, so the sync and async surfaces cannot disagree about a
    path, a body or how a response is read.
    """

    method: str
    path: str
    parse: Callable[[Any], T]
    body: Any = None
    want_json: bool = True
    #: Safe to send twice. ``None`` means "decide from the method": GET, HEAD,
    #: PUT and DELETE are; POST is not unless the operation says so — a search
    #: is a POST that changes nothing.
    idempotent: bool | None = None


def _json_body(model: Any) -> dict[str, Any]:
    """A generated model as a request body, carrying only what was set.

    The same rule as every other body here: a field with a default the caller
    never touched stays off the wire, so the node applies its own default
    rather than one this client restated.
    """
    out: dict[str, Any] = model.model_dump(
        mode="json", by_alias=True, exclude_none=True, exclude_unset=True
    )
    return out


def _as_dict(raw: Any) -> dict[str, Any]:
    """A response the description declares only as ``type: object``.

    Returned as the plain mapping rather than validated into a model with no
    fields, which would silently discard everything the node said.
    """
    return dict(raw) if isinstance(raw, Mapping) else {}


# ─── Entitlement ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Entitlement:
    """What subscription a node holds.

    THREE states, not two, and the third is easy to lose. :attr:`active` is a
    token that verifies now; :attr:`refused` is a token that is PRESENT and was
    rejected — expired, or signed by a key this build does not trust. Neither
    means nothing was ever bought. Collapsing refused into absent tells somebody
    who has paid that they have not.

    :attr:`enforced` is False on a build with no signing keys compiled in, where
    nothing is gated and there is nothing to activate. Say that, rather than
    offering an activation box that cannot do anything.
    """

    active: bool
    enforced: bool
    features: list[str]
    refused: str | None = None
    tier: str | None = None
    mesh_id: str | None = None

    #: Expiry as epoch **seconds**. Read it through :attr:`expires_at`: taken
    #: as milliseconds it lands in 1970.
    not_after: int | None = None

    #: Why a backup bucket the key carried did not register, when it did not.
    #: Separate from :attr:`refused`: the licence is fine and one setup step
    #: needs retrying.
    backup_error: str | None = None

    raw: dict[str, Any] = _dc_field(default_factory=dict, repr=False)

    @property
    def state(self) -> str:
        """``"active"``, ``"refused"``, ``"unenforced"`` or ``"none"``."""
        if self.active:
            return "active"
        if self.refused:
            return "refused"
        if not self.enforced:
            return "unenforced"
        return "none"

    @property
    def expires_at(self) -> datetime | None:
        """:attr:`not_after` as an aware UTC datetime."""
        if self.not_after is None:
            return None
        return datetime.fromtimestamp(self.not_after, tz=timezone.utc)

    def has_feature(self, name: str) -> bool:
        """Whether an ACTIVE subscription grants ``name``. A refused token's
        features grant nothing."""
        return self.active and name in self.features

    @classmethod
    def _parse(cls, body: Any) -> Entitlement:
        raw = _as_dict(body)
        # Validated against the description's shape first, so a node that
        # stopped sending a required field fails here naming it.
        typed = m.V1NodeEntitlementGetResponse.model_validate(raw)
        backup_error = raw.get("backup_error")
        return cls(
            active=typed.active,
            enforced=typed.enforced,
            features=list(typed.features),
            refused=typed.refused,
            tier=typed.tier,
            mesh_id=typed.mesh_id,
            not_after=typed.not_after,
            backup_error=backup_error if isinstance(backup_error, str) else None,
            raw=raw,
        )


def _entitlement_call() -> _Call[Entitlement]:
    return _Call("GET", "/v1/node/entitlement", Entitlement._parse)


def _activate_call(key: str) -> _Call[Entitlement]:
    if not key or not key.strip():
        raise ValueError("gnarl: activate_entitlement: empty key")
    body = _json_body(m.V1NodeEntitlementActivatePostRequest(key=key.strip()))
    # The node answers with the stored subscription — the same shape as the
    # status read — so a caller can show it without a second request.
    return _Call("POST", "/v1/node/entitlement/activate", Entitlement._parse, body=body)


# ─── Index extras ───────────────────────────────────────────────────────────


def _forcemerge_call(
    index: str, max_num_segments: int | None
) -> _Call[m.V1IndexesNameForcemergePostResponse]:
    if not index:
        raise ValueError("gnarl: forcemerge: empty index name")
    if max_num_segments is not None and max_num_segments < 1:
        raise ValueError("gnarl: forcemerge: max_num_segments must be at least 1")
    path = f"/v1/indexes/{_esc(index)}/_forcemerge" + _query_string(
        {"max_num_segments": max_num_segments}
    )
    return _Call("POST", path, m.V1IndexesNameForcemergePostResponse.model_validate)


def _get_policy_call(index: str) -> _Call[m.IndexPolicy]:
    return _Call("GET", f"/v1/indexes/{_esc(index)}/_policy", m.IndexPolicy.model_validate)


def _put_policy_call(
    index: str,
    placement: m.IndexPlacement | str | None,
    replication_factor: int | None,
) -> _Call[m.IndexPolicy]:
    fields: dict[str, Any] = {}
    if placement is not None:
        fields["placement"] = placement
    if replication_factor is not None:
        fields["replication_factor"] = replication_factor
    if not fields:
        # An empty update is a no-op the node would accept; refusing it here
        # catches the caller who meant to pass something and did not.
        raise ValueError("gnarl: put_policy: nothing to change")
    body = _json_body(m.IndexPolicyUpdate.model_validate(fields))
    return _Call(
        "PUT", f"/v1/indexes/{_esc(index)}/_policy", m.IndexPolicy.model_validate, body=body
    )


# ─── Defaults from the environment ──────────────────────────────────────────

#: Read when no address is passed.
ENV_URL = "GNARL_URL"
#: Read when no token is passed. ``token=""`` sends none even when it is set.
ENV_TOKEN = "GNARL_TOKEN"


def _resolve_addr(addr: str | None) -> str:
    if addr is None:
        addr = os.environ.get(ENV_URL)
        if not addr:
            raise ValueError(f"gnarl: no address: pass one, or set ${ENV_URL}")
    return _normalize_base_url(addr)


def _resolve_token(token: str | None) -> str | None:
    if token is None:
        return os.environ.get(ENV_TOKEN) or None
    # An explicit empty string is a decision, not an absence.
    return token or None


# ─── Retry ──────────────────────────────────────────────────────────────────

#: Methods that are safe to send twice by HTTP's own definition.
_IDEMPOTENT_METHODS = frozenset({"GET", "HEAD", "PUT", "DELETE", "OPTIONS"})


@dataclass(frozen=True)
class Retry:
    """When to send a refused request again.

    On by default, and narrow on purpose: only a status that means "not now"
    (429, 503), and only for a request that is safe to repeat — GET, PUT,
    DELETE, and the POSTs that change nothing (search, recall). A write that
    may have landed is never resent, because a second ``index_document``
    without an id is a second document.

    The node's ``Retry-After`` is honoured exactly. When it asks for longer
    than :attr:`max_delay`, the error is raised at once rather than retried
    early: asking again before the node said to is how a client gets itself
    rate-limited harder.

    Pass ``retry=None`` to a client to turn this off.
    """

    #: Total attempts, including the first. 1 means never retry.
    attempts: int = 3

    #: The longest single wait, in seconds.
    max_delay: float = 30.0

    #: The first wait when the node gave no hint; doubled on each retry.
    backoff: float = 0.5

    #: Statuses worth retrying.
    statuses: frozenset[int] = frozenset({429, 503})

    def __post_init__(self) -> None:
        if self.attempts < 1:
            raise ValueError("gnarl: Retry.attempts must be at least 1")
        if self.max_delay < 0 or self.backoff < 0:
            raise ValueError("gnarl: Retry delays cannot be negative")

    def delay(self, attempt: int, err: GnarlError) -> float | None:
        """Seconds to wait before attempt ``attempt + 1``, or ``None`` to give up.

        ``attempt`` counts from 1 — the attempt that just failed.
        """
        if err.status not in self.statuses or attempt >= self.attempts:
            return None
        if err.retry_after is not None:
            return err.retry_after if err.retry_after <= self.max_delay else None
        return float(min(self.backoff * 2.0 ** (attempt - 1), self.max_delay))


#: What a client uses when ``retry`` is not passed.
DEFAULT_RETRY = Retry()


def _is_idempotent(method: str, declared: bool | None) -> bool:
    return declared if declared is not None else method.upper() in _IDEMPOTENT_METHODS


# ─── Pagination and chunking helpers ────────────────────────────────────────


def _cursor_of(page: SearchResult, previous: list[Any] | None) -> list[Any] | None:
    """The ``search_after`` cursor for the page after this one, or ``None``
    when this page was the last.

    The end is an EMPTY page, not a short one: a page can come back short
    because a claim missed its deadline, and stopping there would end the
    iteration silently with rows unread.
    """
    if not page.hits:
        return None
    last = page.hits[-1].sort
    if not last:
        raise GnarlError(
            type="invalid_response",
            reason=(
                "a sorted search returned a hit with no `sort` values, so there "
                "is no cursor for the next page"
            ),
        )
    cursor = list(last)
    if cursor == previous:
        # The same cursor twice would request the same page forever. A node
        # that ignored `search_after` would do exactly that, and an iterator
        # that never ends is worse than one that fails.
        raise GnarlError(
            type="invalid_response",
            reason=f"search_after made no progress: the cursor stayed at {cursor!r}",
        )
    return cursor


def _check_iter_args(sort: Sequence[Any], page_size: int) -> None:
    if not sort:
        # Without an explicit sort there is no cursor: `search_after` pages
        # by the sort values each hit carries back.
        raise ValueError("gnarl: iter_search: sort is required")
    if page_size < 1:
        raise ValueError("gnarl: iter_search: page_size must be at least 1")


def _chunks(docs: Iterable[T], size: int) -> Iterator[list[T]]:
    if size < 1:
        raise ValueError("gnarl: chunk_size must be at least 1")
    chunk: list[T] = []
    for d in docs:
        chunk.append(d)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


#: Weakest first. A merged result can only claim what EVERY chunk reached.
_ACK_ORDER = [m.Ack1.accepted, m.Ack1.accepted_durably, m.Ack1.visible_for_search]


def _merge_bulk(results: Sequence[m.BulkIndexResponse]) -> m.BulkIndexResponse:
    """Several chunks' results as one, in request order.

    ``errors`` is true if any chunk had one, so :func:`failed_items` works on
    the merged result exactly as on a single one. ``ack`` is the WEAKEST level
    any chunk reached — reporting the strongest would claim visibility for
    documents that only reached the log.
    """
    if not results:
        raise ValueError("gnarl: bulk: no documents")
    ack = min((r.ack for r in results), key=_ACK_ORDER.index)
    timed_out = any(r.timed_out for r in results)
    return m.BulkIndexResponse(
        items=[item for r in results for item in r.items],
        errors=any(r.errors for r in results),
        ack=ack,
        timed_out=True if timed_out else None,
    )


_ModelT = TypeVar("_ModelT")


def _public_engine(model: _ModelT) -> _ModelT:
    """Name the native engine ``native`` whichever node answered.

    Nodes released before the rename report it by its old internal binding,
    ``tantivy``. It is the same engine; translating here means a caller never
    sees two names for it."""
    if getattr(model, "engine_binding", None) == "tantivy":
        model.engine_binding = "native"  # type: ignore[attr-defined]
    return model
