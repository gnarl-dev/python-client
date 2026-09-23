"""Unit tests for error classification. No network, no node."""

from __future__ import annotations

import json

import httpx
import pytest

from gnarl.errors import (
    AlreadyExists,
    Forbidden,
    GnarlError,
    InternalError,
    NotFound,
    RateLimited,
    Unauthenticated,
    Unsupported,
    ValidationError,
    error_from_response,
)


def envelope(type_: str, reason: str = "because", **extra) -> bytes:
    body = {"type": type_, "reason": reason, **extra}
    return json.dumps({"error": body}).encode()


@pytest.mark.parametrize(
    ("type_", "status", "expected"),
    [
        ("index_not_found", 404, NotFound),
        ("document_not_found", 404, NotFound),
        ("index_already_exists", 409, AlreadyExists),
        ("validation_error", 400, ValidationError),
        ("schema_error", 400, ValidationError),
        ("unauthenticated", 401, Unauthenticated),
        ("forbidden", 403, Forbidden),
        ("rate_limited", 429, RateLimited),
        ("unsupported_capability", 400, Unsupported),
        ("unsupported_engine", 400, Unsupported),
        ("internal_error", 500, InternalError),
    ],
)
def test_each_type_maps_to_its_class(type_, status, expected):
    err = error_from_response(status, envelope(type_))
    assert isinstance(err, expected)
    assert err.type == type_
    assert err.reason == "because"
    assert err.status == status


def test_every_error_is_a_gnarl_error():
    """One `except GnarlError` catches everything this client raises."""
    err = error_from_response(404, envelope("index_not_found"))
    assert isinstance(err, GnarlError)


def test_an_unknown_type_is_not_swallowed_or_renamed():
    """An unrecognised type keeps its name and stays an error.

    Remapping it to something familiar is worse than leaving it unclassified:
    a caller branching on the guess takes the path meant for a different
    failure.
    """
    err = error_from_response(400, envelope("some_future_type"))
    assert type(err) is GnarlError
    assert err.type == "some_future_type"
    assert err.status == 400


def test_an_unknown_type_still_honours_the_status():
    """A type we do not know, on a status we do, is classified by status."""
    err = error_from_response(404, envelope("some_future_absence"))
    assert isinstance(err, NotFound)
    assert err.type == "some_future_absence"


def test_a_body_that_is_not_the_envelope_still_classifies():
    """The framework's own 422 arrives before any handler, as text/plain."""
    err = error_from_response(422, b"Failed to deserialize the JSON body")
    assert isinstance(err, GnarlError)
    assert err.status == 422
    assert "deserialize" in err.reason


def test_a_proxy_404_with_no_envelope_is_still_not_found():
    err = error_from_response(404, b"<html>404 Not Found</html>")
    assert isinstance(err, NotFound)


def test_an_error_key_that_is_a_string_does_not_crash():
    """The old fourth error shape had `error` as a bare string.

    A client that assumed an object raised a TypeError while unwrapping the
    failure, replacing the server's diagnosis with its own stack trace.
    """
    err = error_from_response(429, json.dumps({"error": "slow down"}).encode())
    assert isinstance(err, RateLimited)
    assert "slow down" in err.reason


def test_an_empty_body_does_not_produce_a_none():
    err = error_from_response(503, b"")
    assert isinstance(err, GnarlError)
    assert err.status == 503


def test_message_is_read_only_as_a_fallback_for_reason():
    """`message` is the deprecated alias; `reason` wins when both are sent."""
    both = json.dumps(
        {"error": {"type": "validation_error", "reason": "new", "message": "old"}}
    ).encode()
    assert error_from_response(400, both).reason == "new"

    only_old = json.dumps(
        {"error": {"type": "validation_error", "message": "old"}}
    ).encode()
    assert error_from_response(400, only_old).reason == "old"


def test_detail_survives():
    err = error_from_response(
        400, envelope("unsupported_capability", detail={"field": "title"})
    )
    assert err.detail == {"field": "title"}


def test_retry_after_is_read_from_the_header():
    headers = httpx.Headers({"Retry-After": "30"})
    err = error_from_response(429, envelope("rate_limited"), headers)
    assert isinstance(err, RateLimited)
    assert err.retry_after == 30.0
    assert err.retry_after_or(5.0) == 30.0


def test_a_missing_retry_after_is_none_not_zero():
    """Zero would read as "retry immediately", which is the opposite."""
    err = error_from_response(429, envelope("rate_limited"), httpx.Headers({}))
    assert err.retry_after is None
    assert err.retry_after_or(5.0) == 5.0


def test_an_unparseable_retry_after_is_none():
    """HTTP allows a date there; we do not pretend to have a number."""
    headers = httpx.Headers({"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"})
    err = error_from_response(429, envelope("rate_limited"), headers)
    assert err.retry_after is None


def test_str_names_the_type_and_the_reason():
    err = error_from_response(404, envelope("index_not_found", "no such index 'x'"))
    assert str(err) == "index_not_found: no such index 'x'"


def test_a_long_unstructured_body_is_truncated_not_dropped():
    err = error_from_response(500, b"x" * 5000)
    assert len(err.reason) < 600
    assert err.reason.startswith("xxx")
