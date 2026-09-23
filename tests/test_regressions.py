"""Regression tests.

Each one pins a defect that shipped, or a mistake that was made while writing
this client. The docstring says what went wrong, because a regression test
whose reason is lost gets deleted the next time it is inconvenient.
"""

from __future__ import annotations

import json
import math

import pytest
import respx
import yaml

from gnarl import Client, Verification, failed_items
from gnarl import query as q
from gnarl._models import BulkIndexResponse, SearchProfile
from gnarl.errors import NotFound, RateLimited, error_from_response

from .test_client import BASE, ok, search_payload

SPEC = yaml.safe_load(
    (__import__("pathlib").Path(__file__).parent.parent / "src/gnarl/openapi.yaml")
    .read_text()
)


def test_a_profile_without_a_graph_block_still_parses():
    """`SearchProfile` was declared with `graph` REQUIRED and no `fan_out`.

    The server sends `fan_out` for every profiled query and `graph` only for a
    graph traversal, so the ordinary response — any non-graph query with
    `profile: true` — contradicted the description twice: a required member
    missing, an undeclared member present. A strictly generated model rejected
    it outright; a lenient one would have discarded the fan-out, which is the
    audit trail the endpoint exists to provide.

    Found by generating this client from the description and pointing it at a
    node. Fixed in the description, not here.
    """
    parsed = SearchProfile.model_validate(
        {
            "fan_out": {
                "nodes_contacted": 2,
                "nodes_responded": 1,
                "deadline_ms": 5000,
                "claims": [
                    {
                        "claim_id": 7,
                        "served_by": "ab" * 32,
                        "source": "remote_replica",
                        "verification": "verified",
                    },
                    {"claim_id": 8, "source": "skipped", "verification": "unverified",
                     "reason": "timeout"},
                ],
                "verified_claims": 1,
                "unverified_claims": 1,
                "failed_verification_claims": 0,
                "local_claims": 0,
            }
        }
    )
    assert parsed.graph is None
    assert parsed.fan_out.nodes_responded == 1
    assert parsed.fan_out.claims[0].verification is Verification.verified
    assert parsed.fan_out.claims[1].reason == "timeout"


def test_the_spec_does_not_require_either_half_of_the_profile():
    """The shape above, asserted against the description rather than a sample,
    so re-vendoring a regressed spec fails here rather than in production."""
    profile = SPEC["components"]["schemas"]["SearchProfile"]
    assert "required" not in profile
    assert set(profile["properties"]) == {"graph", "fan_out"}


def test_unverified_is_its_own_state_and_is_not_failed():
    """"Cannot check" is not "check failed".

    In the node this mistake destroyed data four separate times, because the
    failed state withholds documents: a claim with no committed manifest, and a
    keyed BYOK document that stores no plaintext leaf by design, were both
    reported as tampered and removed from results they belonged in.

    A client that collapses the two re-creates the bug on the reading side,
    where it looks like data loss with no server-side evidence. Four states, in
    the description, distinct.
    """
    states = SPEC["components"]["schemas"]["ClaimRoute"]["properties"]["verification"]
    assert set(states["enum"]) == {"verified", "unverified", "failed", "local"}
    assert {v.value for v in Verification} == {
        "verified",
        "unverified",
        "failed",
        "local",
    }
    assert Verification.unverified is not Verification.failed


def test_a_survey_grade_coordinate_survives_the_round_trip():
    """Generating the Go client produced `Lat float32`, because the description
    said `type: number` with no `format`. float32 displaces a coordinate at
    this precision by roughly 0.23 m — enough to move a point to the wrong side
    of a boundary while every test still passed.

    Python floats are doubles natively, so the hazard is not the language here;
    it is the description saying `double`, and this asserts it still does.
    """
    lat, lon = -33.856784729, 151.215296638
    built = q.geo_distance("location", lat, lon, 1.0)
    wired = json.loads(built.model_dump_json(by_alias=True, exclude_none=True))

    assert wired["geo_distance"]["lat"] == lat
    assert wired["geo_distance"]["lon"] == lon
    # Precisely: the value is not what a float32 round trip would leave.
    as_f32 = __import__("struct").unpack("f", __import__("struct").pack("f", lat))[0]
    assert wired["geo_distance"]["lat"] != as_f32
    assert abs(as_f32 - lat) > 1e-8

    geo = SPEC["components"]["schemas"]["GeoDistanceQuery"]["properties"]
    for field in ("lat", "lon", "radius_meters"):
        assert geo[field]["format"] == "double", (
            f"{field} lost its `format: double` — a generated client is free to "
            "pick float32 again"
        )


def test_the_deprecated_message_alias_is_still_read():
    """The envelope was unified to `reason`, with `message` kept as a
    deprecated alias so callers written before it kept working.

    This client reads `message` only as a fallback. A node built before the
    unification sends `message` alone, and dropping it would leave a caller
    with a typed error carrying no diagnosis at all.
    """
    older_node = json.dumps(
        {"error": {"type": "index_not_found", "message": "no such index"}}
    ).encode()
    err = error_from_response(404, older_node)
    assert isinstance(err, NotFound)
    assert err.reason == "no such index"


def test_an_error_key_that_is_a_string_is_the_fourth_shape():
    """Before the envelope was unified there were four error bodies, and the
    rate limiter's was hand-rolled in middleware with `error` as a bare string.

    Middleware escapes the type system, so this shape can come back from any
    layer that predates the unification. Unwrapping it as an object raised a
    TypeError and replaced the server's diagnosis with a stack trace.
    """
    err = error_from_response(429, json.dumps({"error": "too many requests"}).encode())
    assert isinstance(err, RateLimited)
    assert "too many" in err.reason


@respx.mock
def test_a_bulk_200_with_failures_is_not_a_success():
    """The single most common way to lose writes silently."""
    respx.post(f"{BASE}/v1/indexes/places/_bulk").mock(
        return_value=ok(
            {
                "items": [
                    {"_id": "1", "status": 201},
                    {
                        "_id": "2",
                        "status": 400,
                        "error": {"type": "schema_error", "reason": "unknown field"},
                    },
                ],
                "errors": True,
                "ack": "accepted",
            }
        )
    )
    with Client(BASE) as c:
        result = c.bulk("places", [{"a": 1}, {"b": 2}])
    assert isinstance(result, BulkIndexResponse)
    assert [f.field_id for f in failed_items(result)] == ["2"]


def test_the_index_name_rules_match_the_server():
    """Two defects the Go client found on its first afternoon: the description
    capped names at 128 where the server caps at 64, and required `^[a-z]`
    where the server accepts a leading digit.

    A generated client validating against either would refuse a name the server
    takes, or accept one it refuses — the second being worse, because the error
    then arrives from the node with no client-side explanation.
    """
    name = SPEC["components"]["parameters"]["IndexName"]["schema"]
    assert name["maxLength"] == 64
    assert name["pattern"] == "^[a-z0-9][a-z0-9_-]*$"


def test_the_reserved_field_names_are_documented():
    """`id`, `version`, `title` and `canonical_url` belong to the document
    envelope, so declaring one is rejected at index creation — on every engine.

    `title` is the name most search schemas reach for first, and it was
    documented nowhere until a client tried it.
    """
    described = SPEC["components"]["schemas"]["IndexSchema"]["properties"]["fields"][
        "description"
    ]
    for reserved in ("`id`", "`version`", "`title`", "`canonical_url`"):
        assert reserved in described


@respx.mock
def test_require_complete_does_not_trust_the_partial_flag_alone():
    """Checking `partial` and not the claim arithmetic would pass a result
    where a claim went unserved without the flag being set.

    The point of asking for completeness is not to trust one boolean.
    """
    respx.post(f"{BASE}/v1/indexes/places/_search").mock(
        return_value=ok(
            search_payload(
                partial=False,
                coverage={
                    "expected_claims": 8,
                    "served_claims": 7,
                    "skipped_claims": [],
                },
            )
        )
    )
    from gnarl import IncompleteResult

    with Client(BASE) as c, pytest.raises(IncompleteResult):
        c.search("places", q.match_all(), require_complete=True)


def test_a_retry_after_of_zero_is_a_number_not_an_absence():
    """`retry_after` defaulting to 0.0 when the header is missing reads as
    "retry immediately", which is the opposite of what a 429 means. Absent is
    None, and `retry_after_or` is how you supply your own backoff."""
    import httpx as _httpx

    body = json.dumps({"error": {"type": "rate_limited", "reason": "slow"}}).encode()
    absent = error_from_response(429, body, _httpx.Headers({}))
    assert absent.retry_after is None
    assert absent.retry_after_or(2.0) == 2.0

    explicit_zero = error_from_response(
        429, body, _httpx.Headers({"Retry-After": "0"})
    )
    assert explicit_zero.retry_after == 0.0
    assert explicit_zero.retry_after_or(2.0) == 0.0


def test_a_nan_or_infinite_coordinate_is_refused_before_the_wire():
    """JSON has no NaN. Serializing one produces a body the node cannot parse,
    and the failure surfaces as an opaque 422 from the framework rather than as
    the caller's own bad input."""
    for bad in (math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError):
            q.geo_distance("location", bad, 0.0, 100.0)
