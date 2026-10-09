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
    "Conflict",
    "AlreadyExists",
    "ValidationError",
    "Unauthenticated",
    "Forbidden",
    "RateLimited",
    "Unsupported",
    "InternalError",
    "Unavailable",
    "IncompleteResult",
    "JobFailed",
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
    """An index, document, repository or snapshot that is not there — or no
    route at that path at all (``route_not_found``), which is how a typo or an
    endpoint this node does not have presents."""


class Conflict(GnarlError):
    """The request is valid but cannot happen in the target's current state.

    ``job_in_progress`` (one snapshot job at a time per node),
    ``namespace_not_snapshottable`` (mid-promotion), ``unverified_signer`` (a
    restore that needs explicit consent), or an untyped 409. Usually a reason to
    wait or to change the request, not a reason to give up.
    """


class AlreadyExists(Conflict):
    """Creating something that already exists. A :class:`Conflict`."""


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


class Unavailable(GnarlError):
    """503: the node cannot serve this right now — a claim mid-failover, or
    nowhere to store what was sent. ``retry_after`` carries the hint when the
    node gave one."""


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


class JobFailed(GnarlError):
    """A snapshot, restore or cleanup job finished in ``failed``.

    Raised by the job poller. ``job`` is the final job record, whose ``error``
    says why; a job that was RUNNING when its node stopped is reported as
    failed too, because whether it finished is unknown.
    """

    def __init__(self, job: Any) -> None:
        self.job = job
        kind = job.kind.value if job.kind is not None else "job"
        super().__init__(
            type="job_failed",
            reason=f"{kind} {job.id} failed: {job.error or 'no reason given'}",
        )


# Wire type -> exception class. A type absent from this map still raises a
# GnarlError with ``type`` set: an unrecognised type must never be swallowed,
# nor remapped to something more familiar, because a caller branching on that
# guess takes a path meant for a different failure.
_BY_TYPE: dict[str, type[GnarlError]] = {
    "validation_error": ValidationError,
    "schema_error": ValidationError,
    # Snapshotting a `__pool_N` index without `allow_shared_pool`: a 400 the
    # caller fixes by changing the request.
    "shared_pool": ValidationError,
    "unsupported_capability": Unsupported,
    "unsupported_engine": Unsupported,
    "index_not_found": NotFound,
    "document_not_found": NotFound,
    "route_not_found": NotFound,
    "field_not_found": NotFound,
    "repository_not_found": NotFound,
    "snapshot_not_found": NotFound,
    "index_already_exists": AlreadyExists,
    "unauthenticated": Unauthenticated,
    # NOT an authentication failure despite the name: the node emits it, as a
    # 403, when a supplied BYOK key does not unwrap the namespace's key. The
    # caller is authenticated and the KEY is wrong — `Forbidden`, not "log in".
    "unauthorized": Forbidden,
    "forbidden": Forbidden,
    "rate_limited": RateLimited,
    # The node emits this as a 409 while a namespace is mid-promotion: wait for
    # the promotion and retry. It is a state, not a missing capability.
    "namespace_not_snapshottable": Conflict,
    "job_in_progress": Conflict,
    "unverified_signer": Conflict,
    "internal_error": InternalError,
    "repository_error": InternalError,
}

# For a body we could not type at all — a proxy's 404, a load balancer's 503 —
# fall back to the status so ``except NotFound`` still behaves.
_BY_STATUS: dict[int, type[GnarlError]] = {
    401: Unauthenticated,
    403: Forbidden,
    404: NotFound,
    409: Conflict,
    429: RateLimited,
    503: Unavailable,
}

# Only for a body with NO envelope. A 400 or 422 with no type is a request the
# node refused before it could classify it — the framework's own 422 for a body
# that does not match the declared shape, or a route that answers with plain
# text, like entitlement activation's refusal. An unknown TYPE on a 400 is
# different and stays a plain GnarlError: there the node did classify it, just
# with a name this client does not know.
_BY_STATUS_UNTYPED: dict[int, type[GnarlError]] = {
    **_BY_STATUS,
    400: ValidationError,
    422: ValidationError,
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


def _retry_after_from_detail(detail: Any) -> float | None:
    """The backoff hint some routes put in the body instead of the header.

    A 503 from a claim mid-failover carries ``detail.retry_after_secs`` and no
    ``Retry-After`` header, and the rate limiter puts ``retry_after_seconds``
    in the body as well as the header. The header wins when both are present.
    """
    if not isinstance(detail, dict):
        return None
    for key in ("retry_after_secs", "retry_after_seconds"):
        raw = detail.get(key)
        if isinstance(raw, (int, float)) and not isinstance(raw, bool) and raw >= 0:
            return float(raw)
    return None


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
        cls = _BY_STATUS_UNTYPED.get(status, GnarlError)
        return cls(type="", reason=text, status=status, retry_after=retry_after)

    etype = envelope.get("type", "") or ""
    # ``message`` is a deprecated alias for ``reason``. Read it only as a
    # fallback, so this client keeps working against a node built before the
    # envelope was unified.
    reason = envelope.get("reason") or envelope.get("message") or ""
    cls = _BY_TYPE.get(etype) or _BY_STATUS.get(status, GnarlError)
    detail = envelope.get("detail")
    if retry_after is None:
        retry_after = _retry_after_from_detail(detail)
    return cls(
        type=etype,
        reason=reason,
        status=status,
        detail=detail,
        retry_after=retry_after,
    )
