"""The one error type a caller has to handle.

A node emits a single error envelope on every route::

    {"error": {"type": "...", "reason": "...", "detail": {...}}}

That is worth stating because it was not always true. The API used to produce
four different bodies — including one where ``error`` was a plain string rather
than an object — which forced a second error type into any client generated
from the description. Because there is now one shape, this is one class.
"""

from __future__ import annotations

import json
from typing import Any

__all__ = [
    "GnarlError",
    "NotFound",
    "AlreadyExists",
    "ValidationError",
    "Unauthenticated",
    "Forbidden",
    "RateLimited",
    "Unsupported",
    "InternalError",
    "IncompleteResult",
]


class GnarlError(Exception):
    """A failure reported by a node.

    ``type`` is stable and safe to switch on — new types may be added, existing
    ones are not removed or renamed. Prefer catching the subclasses below,
    which is the same idea expressed so ``except`` does the matching.
    """

    def __init__(
        self,
        *,
        type: str = "",
        reason: str = "",
        status: int = 0,
        detail: dict[str, Any] | None = None,
        retry_after: float | None = None,
    ) -> None:
        self.type = type
        self.reason = reason
        self.status = status
        self.detail = detail
        #: Seconds the server asked you to wait, from ``Retry-After``. ``None``
        #: when it did not say.
        self.retry_after = retry_after
        super().__init__(reason or f"{type} (HTTP {status})")

    def __str__(self) -> str:
        if self.reason:
            return f"{self.type}: {self.reason}"
        return f"{self.type or 'error'} (HTTP {self.status})"

    def retry_after_or(self, default: float) -> float:
        """The server's backoff hint, or ``default`` when it gave none."""
        return self.retry_after if self.retry_after is not None else default


class NotFound(GnarlError):
    """An index, document, repository or snapshot that is not there."""


class AlreadyExists(GnarlError):
    """Creating something that already exists."""


class ValidationError(GnarlError):
    """The request was refused as malformed or contradictory."""


class Unauthenticated(GnarlError):
    """No token, an expired one, or one this node will not accept."""


class Forbidden(GnarlError):
    """Authenticated, but not permitted."""


class RateLimited(GnarlError):
    """Too many requests. ``retry_after`` carries the server's hint."""


class Unsupported(GnarlError):
    """The field, engine or capability cannot do what was asked."""


class InternalError(GnarlError):
    """The node failed in a way it does not attribute to the caller."""


class IncompleteResult(GnarlError):
    """Raised when ``require_complete`` was set and coverage fell short.

    Carries the partial :class:`~gnarl.client.SearchResponse` so a caller can
    inspect it, or degrade to it deliberately, rather than losing the work.
    """

    def __init__(self, response: Any) -> None:
        self.response = response
        cov = response.coverage
        super().__init__(
            type="incomplete",
            reason=(
                f"{cov.served_claims} of {cov.expected_claims} claims answered, "
                f"{len(cov.skipped_claims)} skipped (require_complete was set)"
            ),
        )


# Wire type -> exception class. A type absent from this map still raises a
# GnarlError with ``type`` set: an unrecognised type must never be swallowed,
# nor remapped to something more familiar, because a caller branching on that
# guess takes a path meant for a different failure.
_BY_TYPE: dict[str, type[GnarlError]] = {
    "index_not_found": NotFound,
    "document_not_found": NotFound,
    "field_not_found": NotFound,
    "repository_not_found": NotFound,
    "snapshot_not_found": NotFound,
    "index_already_exists": AlreadyExists,
    "validation_error": ValidationError,
    "schema_error": ValidationError,
    "unauthenticated": Unauthenticated,
    "unauthorized": Unauthenticated,
    "forbidden": Forbidden,
    "rate_limited": RateLimited,
    "unsupported_capability": Unsupported,
    "unsupported_engine": Unsupported,
    "namespace_not_snapshottable": Unsupported,
    "internal_error": InternalError,
    "repository_error": InternalError,
}

# For a body we could not type at all — a proxy's 404, a load balancer's 503 —
# fall back to the status so ``except NotFound`` still behaves.
_BY_STATUS: dict[int, type[GnarlError]] = {
    401: Unauthenticated,
    403: Forbidden,
    404: NotFound,
    409: AlreadyExists,
    429: RateLimited,
}


def _retry_after(headers: Any) -> float | None:
    raw = headers.get("Retry-After") if headers is not None else None
    if not raw:
        return None
    try:
        seconds = float(raw)
    except (TypeError, ValueError):
        return None
    return seconds if seconds >= 0 else None


def error_from_response(status: int, body: bytes, headers: Any = None) -> GnarlError:
    """Turn a non-2xx response into the right exception.

    Never returns ``None`` for a failure and never loses the status. A body
    that is not our envelope still produces something usable: the framework's
    own 422 for a malformed request body is ``text/plain`` and arrives before
    any handler runs, so it has no envelope at all.
    """
    retry_after = _retry_after(headers)

    try:
        parsed = json.loads(body)
        envelope = parsed["error"]
        if not isinstance(envelope, dict):
            raise TypeError
    except Exception:
        # Not our envelope. Surface the status and enough of the body to act
        # on, rather than inventing a type we did not receive.
        text = body.decode("utf-8", "replace").strip()
        if len(text) > 512:
            text = text[:512] + "…"
        cls = _BY_STATUS.get(status, GnarlError)
        return cls(type="", reason=text, status=status, retry_after=retry_after)

    etype = envelope.get("type", "") or ""
    # ``message`` is a deprecated alias for ``reason``. Read it only as a
    # fallback, so this client keeps working against a node built before the
    # envelope was unified.
    reason = envelope.get("reason") or envelope.get("message") or ""
    cls = _BY_TYPE.get(etype) or _BY_STATUS.get(status, GnarlError)
    return cls(
        type=etype,
        reason=reason,
        status=status,
        detail=envelope.get("detail"),
        retry_after=retry_after,
    )
