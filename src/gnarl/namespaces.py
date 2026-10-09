"""Namespaces: many lightweight tenants over bounded physical pools.

Reached as ``client.namespaces``. A namespace is created by its first write
("name it and write") and is a filtered alias over a shared pool by default;
declaring a ``dense_vector`` mapping or promoting it moves it to a dedicated
index of its own. Every read is restricted to the namespace by a filter the
node applies and the caller cannot override.

    c.namespaces.index_document("tenant-a", {"title_text": "hello"}, wait_for="visible")
    res = c.namespaces.search("tenant-a", q.match("title_text", "hello"))
"""

from __future__ import annotations

import base64
from collections.abc import AsyncIterator, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from . import _models as m
from ._wire import (
    BulkDoc,
    SearchResult,
    _as_dict,
    _bulk_body,
    _Call,
    _check_complete,
    _check_iter_args,
    _chunks,
    _cursor_of,
    _document_body,
    _esc,
    _json_body,
    _merge_bulk,
    _query_string,
    _search_body,
    _to_search_result,
)

if TYPE_CHECKING:
    from .client import AsyncClient, Client

__all__ = ["Namespaces", "AsyncNamespaces", "NamespaceList", "WaitFor"]

#: How long a write waits before it is acknowledged. ``visible`` returns once
#: the document is searchable; ``durable`` once it has reached the durable
#: sequence; ``accepted`` does not wait.
WaitFor = Literal["accepted", "durable", "visible", "visible_for_search"]


@dataclass(frozen=True)
class NamespaceList:
    """Every namespace the mesh reported, across all pages."""

    namespaces: list[m.Namespace]

    #: True when ANY page said a peer could not be reached. The catalog is not
    #: replicated — each namespace is recorded by the node that served its
    #: first write — so an unreachable peer means namespaces may be missing,
    #: and this listing is a floor rather than an answer.
    partial: bool

    def names(self) -> list[str]:
        return [n.name for n in self.namespaces]

    def __len__(self) -> int:
        return len(self.namespaces)

    def __iter__(self) -> Iterator[m.Namespace]:
        return iter(self.namespaces)


# ─── Calls, shared by both groups ───────────────────────────────────────────


def _ns(ns: str) -> str:
    if not ns:
        raise ValueError("gnarl: empty namespace name")
    return f"/v1/namespaces/{_esc(ns)}"


def _ingest_qs(wait_for: WaitFor | None, wait_for_timeout_ms: int | None) -> str:
    return _query_string(
        {"wait_for": wait_for, "wait_for_timeout_ms": wait_for_timeout_ms}
    )


def _index_document(
    ns: str,
    doc: Mapping[str, Any],
    id: str | None,
    wait_for: WaitFor | None,
    wait_for_timeout_ms: int | None,
) -> _Call[str]:
    return _Call(
        "POST",
        _ns(ns) + "/_doc" + _ingest_qs(wait_for, wait_for_timeout_ms),
        lambda raw: m.IndexDocumentResponse.model_validate(raw).field_id,
        body=_document_body(doc, id),
    )


def _bulk(
    ns: str,
    docs: Iterable[Mapping[str, Any] | BulkDoc],
    wait_for: WaitFor | None,
    wait_for_timeout_ms: int | None,
) -> _Call[m.BulkIndexResponse]:
    return _Call(
        "POST",
        _ns(ns) + "/_bulk" + _ingest_qs(wait_for, wait_for_timeout_ms),
        m.BulkIndexResponse.model_validate,
        body=_bulk_body(docs),
    )


def _search(ns: str, body: dict[str, Any], require_complete: bool) -> _Call[SearchResult]:
    def parse(raw: Any) -> SearchResult:
        result = _to_search_result(raw)
        if require_complete:
            _check_complete(result)
        return result

    return _Call("POST", _ns(ns) + "/_search", parse, body=body, idempotent=True)


def _get_document(ns: str, id: str) -> _Call[dict[str, Any]]:
    return _Call(
        "GET",
        f"{_ns(ns)}/_doc/{_esc(id)}",
        lambda raw: m.GetDocumentResponse.model_validate(raw).field_source,
    )


def _delete_document(ns: str, id: str) -> _Call[None]:
    # The node answers 204 with no body, whatever the description says.
    return _Call(
        "DELETE", f"{_ns(ns)}/_doc/{_esc(id)}", lambda _raw: None, want_json=False
    )


def _put_mapping(ns: str, schema: m.IndexSchema) -> _Call[dict[str, Any]]:
    # The body IS the schema — not wrapped under `schema` as index creation is.
    return _Call("PUT", _ns(ns) + "/_mapping", _as_dict, body=_json_body(schema))


def _promote(ns: str) -> _Call[dict[str, Any]]:
    # Idempotent by the node's own account: promoting a dedicated namespace is
    # a no-op, so a retry cannot start a second migration.
    return _Call("POST", _ns(ns) + "/_promote", _as_dict, idempotent=True)


def _encode_key(key: str | bytes) -> str:
    """A KEK as the node wants it: base64 of exactly 32 bytes.

    Raw bytes are encoded here. A string is taken to be base64 already and is
    checked, so a hex string or a passphrase fails before it reaches the node
    rather than as a 400 about a key the caller cannot see.
    """
    if isinstance(key, bytes):
        raw = key
        encoded = base64.b64encode(key).decode()
    else:
        encoded = key.strip()
        try:
            raw = base64.b64decode(encoded, validate=True)
        except ValueError as exc:
            raise ValueError("gnarl: namespace key: not valid base64") from exc
    if len(raw) != 32:
        raise ValueError(f"gnarl: namespace key: must be 32 bytes, got {len(raw)}")
    return encoded


def _set_key(ns: str, key: str | bytes) -> _Call[m.NamespaceKeyStatus]:
    body = _json_body(m.V1NamespacesNsKeyPutRequest(key=_encode_key(key)))
    return _Call("PUT", _ns(ns) + "/_key", m.NamespaceKeyStatus.model_validate, body=body)


def _key_status(ns: str) -> _Call[m.NamespaceKeyStatus]:
    return _Call("GET", _ns(ns) + "/_key", m.NamespaceKeyStatus.model_validate)


def _revoke_key(ns: str) -> _Call[m.NamespaceKeyStatus]:
    return _Call("DELETE", _ns(ns) + "/_key", m.NamespaceKeyStatus.model_validate)


def _delete(ns: str) -> _Call[dict[str, Any]]:
    return _Call("DELETE", _ns(ns), _as_dict)


def _list_page(after: str | None, limit: int | None) -> _Call[m.V1NamespacesGetResponse]:
    if limit is not None and not 1 <= limit <= 10_000:
        raise ValueError("gnarl: namespaces.list_page: limit must be 1..10000")
    return _Call(
        "GET",
        "/v1/namespaces" + _query_string({"after": after or None, "limit": limit}),
        m.V1NamespacesGetResponse.model_validate,
    )


# ─── Sync ───────────────────────────────────────────────────────────────────


class Namespaces:
    """``client.namespaces``. See the module docstring."""

    def __init__(self, client: Client) -> None:
        self._c = client

    def index_document(
        self,
        ns: str,
        doc: Mapping[str, Any],
        *,
        id: str | None = None,
        wait_for: WaitFor | None = None,
        wait_for_timeout_ms: int | None = None,
    ) -> str:
        """Write one document, creating the namespace if this is its first.

        Returns the document's id — the TENANT's id, never the internal
        namespace-scoped form. ``wait_for="visible"`` returns only once the
        document is searchable, which is what a read-your-write caller wants.
        """
        return self._c._call(_index_document(ns, doc, id, wait_for, wait_for_timeout_ms))

    def bulk(
        self,
        ns: str,
        docs: Iterable[Mapping[str, Any] | BulkDoc],
        *,
        wait_for: WaitFor | None = None,
        wait_for_timeout_ms: int | None = None,
    ) -> m.BulkIndexResponse:
        """Write many documents in one request — the path for a tenant import.

        Always 200; check :func:`~gnarl.failed_items`. A partial accept is
        reported as a failure for EVERY item, so retrying the batch (it is
        idempotent by ``_id``) is correct rather than wasteful.
        """
        return self._c._call(_bulk(ns, docs, wait_for, wait_for_timeout_ms))

    def search(
        self,
        ns: str,
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
        """Search one namespace. Same parameters as :meth:`Client.search`.

        A kNN query needs a dedicated namespace — declare a ``dense_vector``
        mapping with :meth:`put_mapping`, or :meth:`promote` — and is refused
        on a pooled one.
        """
        body = _search_body(
            query, size, from_, sort, search_after, source,
            track_total_hits, profile, verify, deadline_ms,
        )
        return self._c._call(_search(ns, body, require_complete))

    def bulk_chunked(
        self,
        ns: str,
        docs: Iterable[Mapping[str, Any] | BulkDoc],
        *,
        chunk_size: int = 500,
        wait_for: WaitFor | None = None,
        wait_for_timeout_ms: int | None = None,
    ) -> m.BulkIndexResponse:
        """A tenant import of any size, ``chunk_size`` per request, merged.
        See :meth:`Client.bulk_chunked`."""
        results = [
            self.bulk(ns, chunk, wait_for=wait_for, wait_for_timeout_ms=wait_for_timeout_ms)
            for chunk in _chunks(docs, chunk_size)
        ]
        return _merge_bulk(results)

    def iter_search(
        self,
        ns: str,
        query: m.Query | None = None,
        *,
        sort: Sequence[Any],
        page_size: int = 500,
        source: bool | list[str] | None = None,
        require_complete: bool = False,
    ) -> Iterator[m.Hit]:
        """Every hit in the namespace, via ``search_after``. See
        :meth:`Client.iter_search`."""
        _check_iter_args(sort, page_size)
        cursor: list[Any] | None = None
        while True:
            page = self.search(
                ns, query, size=page_size, sort=sort, search_after=cursor,
                source=source, require_complete=require_complete,
            )
            yield from page.hits
            cursor = _cursor_of(page, cursor)
            if cursor is None:
                return

    def get_document(self, ns: str, id: str) -> dict[str, Any]:
        """A document's stored fields. Raises :class:`~gnarl.NotFound`."""
        return self._c._call(_get_document(ns, id))

    def delete_document(self, ns: str, id: str) -> None:
        """Remove one document. Acknowledged when DURABLE, not when invisible:
        a read straight afterwards can still see it."""
        self._c._call(_delete_document(ns, id))

    def put_mapping(self, ns: str, schema: m.IndexSchema) -> dict[str, Any]:
        """Declare a ``dense_vector`` field, making the namespace dedicated.

        Needs a fresh namespace with no pooled data. Scalar-only mappings are
        refused: scalars are inferred on write.
        """
        return self._c._call(_put_mapping(ns, schema))

    def promote(self, ns: str) -> dict[str, Any]:
        """Move a pooled namespace to a dedicated index, online and without
        lost writes. Idempotent."""
        return self._c._call(_promote(ns))

    def set_key(self, ns: str, key: str | bytes) -> m.NamespaceKeyStatus:
        """Register a tenant key (BYOK), or unlock a namespace already keyed.

        ``key`` is 32 raw bytes, or their base64. ``_source`` is sealed under
        it; searchable terms stay plaintext by design. Needs ``admin`` on an
        RBAC node. A wrong key raises :class:`~gnarl.Forbidden`.
        """
        return self._c._call(_set_key(ns, key))

    def key_status(self, ns: str) -> m.NamespaceKeyStatus:
        """Whether the namespace is keyed and unlocked here. Never key material."""
        return self._c._call(_key_status(ns))

    def revoke_key(self, ns: str) -> m.NamespaceKeyStatus:
        """Crypto-erase: revoke the wrapped key and leave a fail-closed
        tombstone. Not reversible. Needs ``admin``."""
        return self._c._call(_revoke_key(ns))

    def delete(self, ns: str) -> dict[str, Any]:
        """Erase every document in the namespace. Other tenants in the same
        pool are untouched. Needs ``admin`` on an RBAC node."""
        return self._c._call(_delete(ns))

    def list(self) -> NamespaceList:
        """Every namespace, following the cursor to the end.

        Check :attr:`NamespaceList.partial` — see there for why a listing can
        be a floor.
        """
        out: list[m.Namespace] = []
        partial = False
        after: str | None = None
        while True:
            page = self.list_page(after)
            out.extend(page.namespaces or [])
            partial = partial or bool(page.partial)
            after = page.next_after
            if not after:
                return NamespaceList(out, partial)

    def list_page(
        self, after: str | None = None, *, limit: int | None = None
    ) -> m.V1NamespacesGetResponse:
        """One page; ``next_after`` is the cursor for the next (absent at the end)."""
        return self._c._call(_list_page(after, limit))


# ─── Async ──────────────────────────────────────────────────────────────────


class AsyncNamespaces:
    """:class:`Namespaces`, awaited. Same surface; see it for the documentation."""

    def __init__(self, client: AsyncClient) -> None:
        self._c = client

    async def index_document(
        self,
        ns: str,
        doc: Mapping[str, Any],
        *,
        id: str | None = None,
        wait_for: WaitFor | None = None,
        wait_for_timeout_ms: int | None = None,
    ) -> str:
        return await self._c._call(
            _index_document(ns, doc, id, wait_for, wait_for_timeout_ms)
        )

    async def bulk(
        self,
        ns: str,
        docs: Iterable[Mapping[str, Any] | BulkDoc],
        *,
        wait_for: WaitFor | None = None,
        wait_for_timeout_ms: int | None = None,
    ) -> m.BulkIndexResponse:
        return await self._c._call(_bulk(ns, docs, wait_for, wait_for_timeout_ms))

    async def search(
        self,
        ns: str,
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
        body = _search_body(
            query, size, from_, sort, search_after, source,
            track_total_hits, profile, verify, deadline_ms,
        )
        return await self._c._call(_search(ns, body, require_complete))

    async def bulk_chunked(
        self,
        ns: str,
        docs: Iterable[Mapping[str, Any] | BulkDoc],
        *,
        chunk_size: int = 500,
        wait_for: WaitFor | None = None,
        wait_for_timeout_ms: int | None = None,
    ) -> m.BulkIndexResponse:
        results = [
            await self.bulk(
                ns, chunk, wait_for=wait_for, wait_for_timeout_ms=wait_for_timeout_ms
            )
            for chunk in _chunks(docs, chunk_size)
        ]
        return _merge_bulk(results)

    async def iter_search(
        self,
        ns: str,
        query: m.Query | None = None,
        *,
        sort: Sequence[Any],
        page_size: int = 500,
        source: bool | list[str] | None = None,
        require_complete: bool = False,
    ) -> AsyncIterator[m.Hit]:
        _check_iter_args(sort, page_size)
        cursor: list[Any] | None = None
        while True:
            page = await self.search(
                ns, query, size=page_size, sort=sort, search_after=cursor,
                source=source, require_complete=require_complete,
            )
            for hit in page.hits:
                yield hit
            cursor = _cursor_of(page, cursor)
            if cursor is None:
                return

    async def get_document(self, ns: str, id: str) -> dict[str, Any]:
        return await self._c._call(_get_document(ns, id))

    async def delete_document(self, ns: str, id: str) -> None:
        await self._c._call(_delete_document(ns, id))

    async def put_mapping(self, ns: str, schema: m.IndexSchema) -> dict[str, Any]:
        return await self._c._call(_put_mapping(ns, schema))

    async def promote(self, ns: str) -> dict[str, Any]:
        return await self._c._call(_promote(ns))

    async def set_key(self, ns: str, key: str | bytes) -> m.NamespaceKeyStatus:
        return await self._c._call(_set_key(ns, key))

    async def key_status(self, ns: str) -> m.NamespaceKeyStatus:
        return await self._c._call(_key_status(ns))

    async def revoke_key(self, ns: str) -> m.NamespaceKeyStatus:
        return await self._c._call(_revoke_key(ns))

    async def delete(self, ns: str) -> dict[str, Any]:
        return await self._c._call(_delete(ns))

    async def list(self) -> NamespaceList:
        out: list[m.Namespace] = []
        partial = False
        after: str | None = None
        while True:
            page = await self.list_page(after)
            out.extend(page.namespaces or [])
            partial = partial or bool(page.partial)
            after = page.next_after
            if not after:
                return NamespaceList(out, partial)

    async def list_page(
        self, after: str | None = None, *, limit: int | None = None
    ) -> m.V1NamespacesGetResponse:
        return await self._c._call(_list_page(after, limit))
