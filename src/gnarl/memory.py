"""Agent memory: durable recall for assistants.

Reached as ``client.memory``. ``remember`` stores a fact, ``recall`` retrieves
by hybrid lexical + vector search, ``answer`` composes over what was recalled,
and the ``ingest_*`` methods take documents, chat transcripts and voice.

    c.memory.remember("the boat is moored at pier 4", namespace="agent-1")
    hits = c.memory.recall("where is the boat", namespace="agent-1")

Embedding happens on the node. A node with no on-device model and no route to
fetch one fails fast with an error naming where to install it; it does not hang.

``space`` and ``namespace`` both say where a memory lives. ``space`` is the
product-level choice — ``personal`` stays on the device, ``household`` is
REPLICATED to the mesh's peers — and ``namespace`` names one explicitly.
"""

from __future__ import annotations

import base64
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from . import _models as m
from ._wire import _as_dict, _Call, _json_body, _public_engine

if TYPE_CHECKING:
    from .client import AsyncClient, Client

__all__ = ["Memory", "AsyncMemory"]


# ─── Calls, shared by both groups ───────────────────────────────────────────


def _drop_none(**fields: Any) -> dict[str, Any]:
    return {k: v for k, v in fields.items() if v is not None}


def _remember(content: str, fields: dict[str, Any]) -> _Call[m.V1MemoryRememberPostResponse]:
    if not content or not content.strip():
        raise ValueError("gnarl: memory.remember: empty content")
    body = _json_body(
        m.V1MemoryRememberPostRequest.model_validate({"content": content, **fields})
    )
    return _Call(
        "POST",
        "/v1/memory/remember",
        lambda raw: _public_engine(m.V1MemoryRememberPostResponse.model_validate(raw)),
        body=body,
    )


def _recall(query: str, fields: dict[str, Any]) -> _Call[m.V1MemoryRecallPostResponse]:
    if not query or not query.strip():
        raise ValueError("gnarl: memory.recall: empty query")
    body = _json_body(m.V1MemoryRecallPostRequest.model_validate({"query": query, **fields}))
    # A read: recalling twice returns the same memories and stores nothing.
    return _Call(
        "POST",
        "/v1/memory/recall",
        m.V1MemoryRecallPostResponse.model_validate,
        body=body,
        idempotent=True,
    )


def _answer(query: str, fields: dict[str, Any]) -> _Call[dict[str, Any]]:
    if not query or not query.strip():
        raise ValueError("gnarl: memory.answer: empty query")
    # Not validated through the generated request model: the description
    # declares only query/namespace/k, and the node also reads space, user and
    # session — the same scoping remember and recall take. Validating would
    # silently drop them.
    return _Call(
        "POST", "/v1/memory/answer", _as_dict, body={"query": query, **fields},
        idempotent=True,
    )


def _bootstrap(fields: dict[str, Any]) -> _Call[dict[str, Any]]:
    # Always a JSON object, even an empty one: the node reads a JSON body here
    # and refuses a request without one, though the description declares none.
    return _Call("POST", "/v1/memory/bootstrap", _as_dict, body=fields, idempotent=True)


def _ingest_document(
    filename: str, content: bytes, space: str
) -> _Call[dict[str, Any]]:
    if not filename:
        raise ValueError("gnarl: memory.ingest_document: empty filename")
    if not space:
        # Required on purpose. `household` replicates to every mesh peer, so
        # a default would let a folder pick share somebody's documents with
        # their whole family or team without their choosing it.
        raise ValueError(
            "gnarl: memory.ingest_document: space is required "
            "('personal' stays on this device, 'household' is shared with mesh peers)"
        )
    if not isinstance(content, (bytes, bytearray)):
        raise TypeError("gnarl: memory.ingest_document: content must be bytes")
    body = {
        "filename": filename,
        "content_base64": base64.b64encode(bytes(content)).decode(),
        "space": space,
    }
    return _Call("POST", "/v1/memory/ingest/document", _as_dict, body=body)


def _ingest_messages(
    messages: Sequence[Mapping[str, Any]], fields: dict[str, Any]
) -> _Call[dict[str, Any]]:
    encoded: list[dict[str, Any]] = []
    for i, msg in enumerate(messages):
        if "role" not in msg or "body" not in msg:
            raise ValueError(
                f"gnarl: memory.ingest_messages: message {i} needs 'role' and 'body'"
            )
        encoded.append(dict(msg))
    if not encoded:
        raise ValueError("gnarl: memory.ingest_messages: no messages")
    return _Call(
        "POST", "/v1/memory/ingest/messages", _as_dict, body={"messages": encoded, **fields}
    )


def _ingest_voice(transcript: str, fields: dict[str, Any]) -> _Call[dict[str, Any]]:
    if not transcript or not transcript.strip():
        raise ValueError("gnarl: memory.ingest_voice: empty transcript")
    return _Call(
        "POST", "/v1/memory/ingest/voice", _as_dict, body={"transcript": transcript, **fields}
    )


# ─── Sync ───────────────────────────────────────────────────────────────────


class Memory:
    """``client.memory``. See the module docstring."""

    def __init__(self, client: Client) -> None:
        self._c = client

    def remember(
        self,
        content: str,
        *,
        namespace: str | None = None,
        space: str | None = None,
        user: str | None = None,
        session: str | None = None,
        agent: str | None = None,
        fact_type: str | None = None,
        pinned: bool | None = None,
        tags: Mapping[str, str] | None = None,
        url: str | None = None,
        title: str | None = None,
        source: str | None = None,
    ) -> m.V1MemoryRememberPostResponse:
        """Store one memory. Returns its id and which embedder produced it.

        Not retried on a 429/503: a second attempt after an ambiguous failure
        could store the memory twice.
        """
        fields = _drop_none(
            namespace=namespace, space=space, user=user, session=session, agent=agent,
            fact_type=fact_type, pinned=pinned, tags=dict(tags) if tags else None,
            url=url, title=title, source=source,
        )
        return self._c._call(_remember(content, fields))

    def recall(
        self,
        query: str,
        *,
        namespace: str | None = None,
        space: str | None = None,
        user: str | None = None,
        session: str | None = None,
        k: int | None = None,
    ) -> m.V1MemoryRecallPostResponse:
        """The memories that best match ``query``, best first.

        ``k`` above 100 is CLAMPED to 100 by the node, not refused — so getting
        100 back does not mean there were only 100. Each memory's ``tags``
        comes back as a flat ``k=v,k=v`` string, not the mapping you stored.
        """
        fields = _drop_none(namespace=namespace, space=space, user=user, session=session, k=k)
        return self._c._call(_recall(query, fields))

    def answer(
        self,
        query: str,
        *,
        namespace: str | None = None,
        space: str | None = None,
        user: str | None = None,
        session: str | None = None,
        k: int | None = None,
    ) -> dict[str, Any]:
        """Answer ``query`` from the ``k`` best memories, with the memories it
        drew on. The response shape is not pinned by the description, so it is
        returned as the node sent it."""
        fields = _drop_none(namespace=namespace, space=space, user=user, session=session, k=k)
        return self._c._call(_answer(query, fields))

    def bootstrap(
        self,
        *,
        namespace: str | None = None,
        space: str | None = None,
        user: str | None = None,
        session: str | None = None,
        query: str | None = None,
        k: int | None = None,
    ) -> dict[str, Any]:
        """Prepare a memory space, returning what an agent should start with."""
        fields = _drop_none(
            namespace=namespace, space=space, user=user, session=session, query=query, k=k
        )
        return self._c._call(_bootstrap(fields))

    def ingest_document(self, filename: str, content: bytes, *, space: str) -> dict[str, Any]:
        """Extract, chunk and remember a file.

        ``filename``'s extension selects the extractor and is the identity the
        chunks are keyed by, so sending the same document again dedups rather
        than duplicating. ``space`` is required — see the module docstring.
        """
        return self._c._call(_ingest_document(filename, content, space))

    def ingest_messages(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        namespace: str | None = None,
        space: str | None = None,
        user: str | None = None,
        thread_title: str | None = None,
        thread_id: str | None = None,
        source: str | None = None,
    ) -> dict[str, Any]:
        """Remember a chat transcript. Each message is ``{"role", "body"}``
        with an optional ``ts_ms``; ``role`` is ``me``, ``them`` or a label."""
        fields = _drop_none(
            namespace=namespace, space=space, user=user, thread_title=thread_title,
            thread_id=thread_id, source=source,
        )
        return self._c._call(_ingest_messages(messages, fields))

    def ingest_voice(
        self,
        transcript: str,
        *,
        namespace: str | None = None,
        space: str | None = None,
        user: str | None = None,
        duration_ms: int | None = None,
        audio_path: str | None = None,
        source: str | None = None,
    ) -> dict[str, Any]:
        """Remember a transcribed voice note."""
        fields = _drop_none(
            namespace=namespace, space=space, user=user, duration_ms=duration_ms,
            audio_path=audio_path, source=source,
        )
        return self._c._call(_ingest_voice(transcript, fields))


# ─── Async ──────────────────────────────────────────────────────────────────


class AsyncMemory:
    """:class:`Memory`, awaited. Same surface; see it for the documentation."""

    def __init__(self, client: AsyncClient) -> None:
        self._c = client

    async def remember(
        self,
        content: str,
        *,
        namespace: str | None = None,
        space: str | None = None,
        user: str | None = None,
        session: str | None = None,
        agent: str | None = None,
        fact_type: str | None = None,
        pinned: bool | None = None,
        tags: Mapping[str, str] | None = None,
        url: str | None = None,
        title: str | None = None,
        source: str | None = None,
    ) -> m.V1MemoryRememberPostResponse:
        fields = _drop_none(
            namespace=namespace, space=space, user=user, session=session, agent=agent,
            fact_type=fact_type, pinned=pinned, tags=dict(tags) if tags else None,
            url=url, title=title, source=source,
        )
        return await self._c._call(_remember(content, fields))

    async def recall(
        self,
        query: str,
        *,
        namespace: str | None = None,
        space: str | None = None,
        user: str | None = None,
        session: str | None = None,
        k: int | None = None,
    ) -> m.V1MemoryRecallPostResponse:
        fields = _drop_none(namespace=namespace, space=space, user=user, session=session, k=k)
        return await self._c._call(_recall(query, fields))

    async def answer(
        self,
        query: str,
        *,
        namespace: str | None = None,
        space: str | None = None,
        user: str | None = None,
        session: str | None = None,
        k: int | None = None,
    ) -> dict[str, Any]:
        fields = _drop_none(namespace=namespace, space=space, user=user, session=session, k=k)
        return await self._c._call(_answer(query, fields))

    async def bootstrap(
        self,
        *,
        namespace: str | None = None,
        space: str | None = None,
        user: str | None = None,
        session: str | None = None,
        query: str | None = None,
        k: int | None = None,
    ) -> dict[str, Any]:
        fields = _drop_none(
            namespace=namespace, space=space, user=user, session=session, query=query, k=k
        )
        return await self._c._call(_bootstrap(fields))

    async def ingest_document(
        self, filename: str, content: bytes, *, space: str
    ) -> dict[str, Any]:
        return await self._c._call(_ingest_document(filename, content, space))

    async def ingest_messages(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        namespace: str | None = None,
        space: str | None = None,
        user: str | None = None,
        thread_title: str | None = None,
        thread_id: str | None = None,
        source: str | None = None,
    ) -> dict[str, Any]:
        fields = _drop_none(
            namespace=namespace, space=space, user=user, thread_title=thread_title,
            thread_id=thread_id, source=source,
        )
        return await self._c._call(_ingest_messages(messages, fields))

    async def ingest_voice(
        self,
        transcript: str,
        *,
        namespace: str | None = None,
        space: str | None = None,
        user: str | None = None,
        duration_ms: int | None = None,
        audio_path: str | None = None,
        source: str | None = None,
    ) -> dict[str, Any]:
        fields = _drop_none(
            namespace=namespace, space=space, user=user, duration_ms=duration_ms,
            audio_path=audio_path, source=source,
        )
        return await self._c._call(_ingest_voice(transcript, fields))
