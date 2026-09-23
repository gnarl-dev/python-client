"""Conformance: this client against a live node.

Every test here drives real HTTP. See ``conftest.py`` for how a node is found.
"""

from __future__ import annotations

import time

import pytest

from gnarl import (
    AlreadyExists,
    AsyncClient,
    BulkDoc,
    Client,
    GnarlError,
    NotFound,
    ValidationError,
    Verification,
    failed_items,
)
from gnarl import query as q

from .conftest import until

pytestmark = pytest.mark.conformance


def searchable(client: Client, index: str, query, want: int, within: float = 30.0):
    """Poll until the query returns at least ``want`` hits.

    A write is acknowledged when it is DURABLE, not when it is SEARCHABLE.
    Asserting immediately after a write races the commit, and the failure looks
    exactly like "search is broken" rather than "we asked too early". Waiting
    on the condition is usually faster than sleeping a worst case, because the
    common case returns on the first probe.
    """
    seen: list = []

    def enough() -> bool:
        seen[:] = client.search(index, query, size=50).hits
        return len(seen) >= want

    assert until(enough, within=within), (
        f"after {within:.0f}s {index} returned {len(seen)} hits, wanted at least "
        f"{want}. This waits on the condition, so exceeding the budget means "
        "the documents genuinely did not become searchable."
    )
    return seen


def gone(client: Client, index: str, doc_id: str, within: float = 30.0) -> bool:
    """Poll until the document is no longer readable.

    A delete is acknowledged when it is durable, the same as a write, so it
    becomes invisible some time after the call returns.
    """

    def absent() -> bool:
        try:
            client.get_document(index, doc_id)
        except NotFound:
            return True
        return False

    return until(absent, within=within)


# ─── Smoke: the node answers at all ─────────────────────────────────────────


def test_the_node_reports_a_status(client: Client):
    status = client.status()
    assert status.node_id, "the node reported an empty node_id"
    assert status.mode, "the node reported an empty mode"


def test_version_is_a_different_endpoint_from_status(client: Client):
    """This client once declared `version` on the status type, where it decoded
    as empty forever without ever failing. An invented field is invisible until
    someone reads it and believes it."""
    assert client.version().version, "the node reported an empty version"


def test_a_document_survives_a_round_trip(client: Client, index):
    """The smoke test: create, write, read back, search, delete."""
    name = index(q.schema({"headline": q.text_field(), "tag": q.keyword_field()}))

    client.index_document(name, {"headline": "a king tide", "tag": "weather"}, id="d1")
    assert client.get_document(name, "d1")["headline"] == "a king tide"

    hits = searchable(client, name, q.match("headline", "tide"), 1)
    assert hits[0].field_id == "d1"

    client.delete_document(name, "d1")
    assert gone(client, name, "d1"), (
        "a deleted document was still readable after the wait. A delete is "
        "acknowledged when it is DURABLE, not when it has become invisible — "
        "the same asymmetry as a write — so this polls rather than asserting "
        "immediately."
    )


# ─── Index lifecycle ────────────────────────────────────────────────────────


def test_an_index_reports_its_own_schema(client: Client, index):
    name = index(q.schema({"headline": q.text_field(), "tag": q.keyword_field()}))
    assert client.index_exists(name) is True
    fields = client.get_schema(name).fields
    assert set(fields) >= {"headline", "tag"}


def test_a_new_index_appears_in_the_listing(client: Client, index):
    """Across every page. A caller who stops at the first page silently sees a
    prefix of their own data."""
    name = index(q.schema({"headline": q.text_field()}))
    assert name in {i.name for i in client.list_indexes()}


def test_an_absent_index_reports_absent_rather_than_failing(client: Client):
    assert client.index_exists("definitely-not-here-0000") is False


def test_creating_an_index_twice_is_refused(client: Client, index):
    """Silently accepting is how a caller destroys a live index believing they
    created a fresh one."""
    schema = q.schema({"headline": q.text_field()})
    name = index(schema)
    with pytest.raises(AlreadyExists):
        client.create_index(name, schema)


def test_a_reserved_field_name_is_refused_at_creation(client: Client):
    """`title` is the name most schemas reach for first, and the node rejects
    it on every engine — it belongs to the document envelope."""
    with pytest.raises(ValidationError):
        client.create_index(
            f"reserved-{time.time_ns() % 1_000_000}",
            q.schema({"title": q.text_field()}),
        )


def test_a_deleted_index_is_gone(client: Client):
    name = f"conf-drop-{time.time_ns() % 1_000_000_000}"
    client.create_index(name, q.schema({"headline": q.text_field()}))
    client.delete_index(name)
    assert client.index_exists(name) is False


# ─── Documents ──────────────────────────────────────────────────────────────


def test_an_id_is_generated_when_none_is_given(client: Client, index):
    name = index(q.schema({"headline": q.text_field()}))
    generated = client.index_document(name, {"headline": "no id here"})
    assert generated
    assert client.get_document(name, generated)["headline"] == "no id here"


def test_a_missing_document_reports_document_not_found(client: Client, index):
    """Not `index_not_found`. The index is right there; the document is not,
    and telling a caller their index is missing sends them to fix the wrong
    thing."""
    name = index(q.schema({"headline": q.text_field()}))
    with pytest.raises(NotFound) as caught:
        client.get_document(name, "never-written")
    assert caught.value.type == "document_not_found", (
        f"a missing document reported {caught.value.type!r}"
    )


def test_a_document_id_with_a_slash_round_trips(client: Client, index):
    """A document id is caller data. If it is not escaped into the path it
    becomes extra path segments and the request hits a different route."""
    name = index(q.schema({"headline": q.text_field()}))
    doc_id = "urn:example/2026/a b"
    client.index_document(name, {"headline": "slashes"}, id=doc_id)
    assert client.get_document(name, doc_id)["headline"] == "slashes"


def test_bulk_indexes_every_document(client: Client, index):
    name = index(q.schema({"headline": q.text_field(), "tag": q.keyword_field()}))
    result = client.bulk(
        name,
        [
            BulkDoc("b1", {"headline": "first light", "tag": "dawn"}),
            BulkDoc("b2", {"headline": "second light", "tag": "dawn"}),
            {"headline": "third light", "tag": "dawn"},
        ],
    )
    assert failed_items(result) == [], "a bulk item failed behind a 200"
    assert len(result.items) == 3
    searchable(client, name, q.term("tag", "dawn"), 3)


def test_a_type_mismatch_rejects_the_whole_batch(client: Client, index):
    """What the node actually does, which is not what the response shape
    suggests.

    `BulkIndexResponse` carries a per-item `error` and an `errors` flag, so the
    shape can express "these three failed, the rest landed". A document whose
    value does not fit the field's type never reaches that path: conversion
    happens for the whole batch before any write, and the first failure returns
    a single 400 for the request.

    Asserted as it is rather than as it should be — a conformance suite records
    the contract, it does not wish for one. The product question (one bad
    document losing 999 good ones, and the error not naming which) is filed
    separately.
    """
    name = index(q.schema({"count": q.integer_field()}))
    with pytest.raises(ValidationError) as caught:
        client.bulk(name, [{"count": 1}, {"count": "not a number"}])
    assert caught.value.reason
    assert client.count(name) == 0, "part of a rejected batch was written anyway"


def test_a_bulk_result_reports_per_item_outcomes(client: Client, index):
    """The path that does use per-item results: every item accepted."""
    name = index(q.schema({"count": q.integer_field()}))
    result = client.bulk(name, [BulkDoc(f"n{i}", {"count": i}) for i in range(3)])
    assert result.errors is False
    assert failed_items(result) == []
    assert {i.field_id for i in result.items} == {"n0", "n1", "n2"}


# ─── Search ─────────────────────────────────────────────────────────────────


def test_a_term_query_matches_a_keyword_exactly(client: Client, index):
    name = index(q.schema({"tag": q.keyword_field()}))
    client.index_document(name, {"tag": "published"}, id="t1")
    hits = searchable(client, name, q.term("tag", "published"), 1)
    assert hits[0].field_id == "t1"


def test_a_range_query_bounds_a_number(client: Client, index):
    name = index(q.schema({"depth": q.integer_field()}))
    client.bulk(name, [BulkDoc(f"d{i}", {"depth": i}) for i in range(10)])
    searchable(client, name, q.match_all(), 10)
    hits = client.search(name, q.range_("depth", gte=7), size=50).hits
    assert {h.field_id for h in hits} == {"d7", "d8", "d9"}


def test_a_bool_query_combines_clauses(client: Client, index):
    name = index(q.schema({"headline": q.text_field(), "tag": q.keyword_field()}))
    client.bulk(
        name,
        [
            BulkDoc("a", {"headline": "high tide", "tag": "weather"}),
            BulkDoc("b", {"headline": "high tide", "tag": "archive"}),
        ],
    )
    searchable(client, name, q.match_all(), 2)
    query = (
        q.bool_().must(q.match("headline", "tide")).filter(q.term("tag", "weather"))
    ).build()
    hits = client.search(name, query, size=50).hits
    assert [h.field_id for h in hits] == ["a"]


def test_every_response_carries_coverage(client: Client, index):
    """Whether or not completeness was asked for."""
    name = index(q.schema({"headline": q.text_field()}))
    res = client.search(name, q.match_all())
    assert res.coverage.expected_claims >= 1
    assert res.coverage.served_claims <= res.coverage.expected_claims


def test_a_single_node_search_is_complete(client: Client, index):
    """On one node nothing can be unreachable, so `require_complete` must not
    raise. If this fails, completeness is being computed wrongly rather than
    reporting a real gap."""
    name = index(q.schema({"headline": q.text_field()}))
    client.index_document(name, {"headline": "present"}, id="c1")
    searchable(client, name, q.match_all(), 1)
    res = client.search(name, q.match_all(), require_complete=True)
    assert res.partial is False
    assert res.coverage.served_claims == res.coverage.expected_claims


def test_size_bounds_the_hits(client: Client, index):
    name = index(q.schema({"headline": q.text_field()}))
    client.bulk(name, [BulkDoc(f"s{i}", {"headline": "many"}) for i in range(5)])
    searchable(client, name, q.match_all(), 5)
    assert len(client.search(name, q.match_all(), size=2).hits) == 2


def test_source_false_omits_the_source(client: Client, index):
    name = index(q.schema({"headline": q.text_field()}))
    client.index_document(name, {"headline": "hidden"}, id="s1")
    searchable(client, name, q.match_all(), 1)
    hits = client.search(name, q.match_all(), source=False).hits
    assert hits[0].field_source is None


def test_a_query_against_a_missing_index_reports_index_not_found(client: Client):
    with pytest.raises(NotFound) as caught:
        client.search("no-such-index-0000", q.match_all())
    assert caught.value.type == "index_not_found"


# ─── Geospatial ─────────────────────────────────────────────────────────────


def test_geo_distance_finds_a_point_inside_the_radius(client: Client, index):
    name = index(q.schema({"name": q.keyword_field(), "location": q.geo_point_field()}))
    client.index_document(
        name,
        {"name": "opera house", "location": q.geo_point(-33.8568, 151.2153)},
        id="g1",
    )
    searchable(client, name, q.match_all(), 1)

    # ~1.6 km away: the harbour bridge.
    hits = client.search(
        name, q.geo_distance("location", -33.8523, 151.2108, 5_000), size=50
    ).hits
    assert [h.field_id for h in hits] == ["g1"]


def test_geo_distance_excludes_a_point_outside_the_radius(client: Client, index):
    """The other half of the assertion. A radius query that matched everything
    would pass the test above and be completely broken."""
    name = index(q.schema({"name": q.keyword_field(), "location": q.geo_point_field()}))
    client.bulk(
        name,
        [
            BulkDoc("near", {"name": "opera house",
                             "location": q.geo_point(-33.8568, 151.2153)}),
            BulkDoc("far", {"name": "melbourne",
                            "location": q.geo_point(-37.8136, 144.9631)}),
        ],
    )
    searchable(client, name, q.match_all(), 2)
    hits = client.search(
        name, q.geo_distance("location", -33.8568, 151.2153, 10_000), size=50
    ).hits
    assert [h.field_id for h in hits] == ["near"]


def test_a_coordinate_survives_the_node_at_full_double_precision(client: Client, index):
    """Every hop keeps the double.

    A float32 anywhere in the path displaces this coordinate by roughly 0.23 m,
    which is enough to move a point across a boundary while every count-based
    assertion still passes. Pinned on the value that comes back, not on a hit
    count, because a count cannot see a displacement.
    """
    lat, lon = -33.856784729, 151.215296638
    name = index(q.schema({"name": q.keyword_field(), "location": q.geo_point_field()}))
    client.index_document(
        name, {"name": "survey mark", "location": q.geo_point(lat, lon)}, id="p1"
    )
    searchable(client, name, q.match_all(), 1)

    stored = client.get_document(name, "p1")["location"]
    assert stored["lat"] == pytest.approx(lat, abs=1e-9), (
        f"latitude came back as {stored['lat']!r}, displaced by "
        f"{abs(stored['lat'] - lat) * 111_320:.3f} m"
    )
    assert stored["lon"] == pytest.approx(lon, abs=1e-9)

    # And a radius tight enough that float32 rounding alone would miss it.
    hits = client.search(name, q.geo_distance("location", lat, lon, 0.05), size=5).hits
    assert [h.field_id for h in hits] == ["p1"], (
        "a 5 cm radius around the exact stored coordinate found nothing — "
        "something in the path is not carrying full precision"
    )


# ─── Observability ──────────────────────────────────────────────────────────


def test_a_profile_is_returned_only_when_asked_for(client: Client, index):
    """The fan-out walks every claim in the routing table. Cheap next to a
    search, but on the hot path of every query a node serves, and "cheap"
    multiplied by every request is a steady-state cost introduced by accident.
    """
    name = index(q.schema({"headline": q.text_field()}))
    client.index_document(name, {"headline": "explain me"}, id="e1")
    searchable(client, name, q.match_all(), 1)

    assert client.search(name, q.match_all()).profile is None


def test_a_profile_reports_the_fan_out(client: Client, index):
    """The response this client was generated to parse, which the description
    declared wrongly until generating this client caught it: `fan_out` present,
    `graph` absent, because this is not a graph traversal."""
    name = index(q.schema({"headline": q.text_field()}))
    client.index_document(name, {"headline": "explain me"}, id="e1")
    searchable(client, name, q.match_all(), 1)

    res = client.search(name, q.match_all(), profile=True)
    assert res.profile is not None, "profile=True returned no profile"
    fan_out = res.profile.fan_out
    assert fan_out is not None, "the profile carried no fan-out"
    assert fan_out.nodes_responded >= 1
    assert fan_out.deadline_ms > 0
    assert len(fan_out.claims) >= 1

    # The four verification counts partition the claim rows exactly.
    counted = (
        fan_out.verified_claims
        + fan_out.unverified_claims
        + fan_out.failed_verification_claims
        + fan_out.local_claims
    )
    assert counted == len(fan_out.claims), (
        f"{counted} counted against {len(fan_out.claims)} claim rows — the "
        "states do not account for every claim"
    )
    for claim in fan_out.claims:
        assert isinstance(claim.verification, Verification)


def test_a_single_node_search_verifies_or_says_it_could_not(client: Client, index):
    """`verify=True` must answer in one of the documented states, never
    silently pass. On a single node the claims are local — there is no remote
    party to attest to — so `local` is the expected outcome, and what must NOT
    appear is a claim reported as verified that nobody checked.
    """
    name = index(q.schema({"headline": q.text_field()}))
    client.index_document(name, {"headline": "prove it"}, id="v1")
    searchable(client, name, q.match_all(), 1)

    res = client.search(name, q.match_all(), verify=True, profile=True)
    assert res.profile.fan_out.failed_verification_claims == 0, (
        "a claim this node served itself failed verification"
    )
    assert [h.field_id for h in res.hits] == ["v1"], (
        "verify=True withheld a document the node served itself. "
        "'Cannot check' is not 'check failed'."
    )


# ─── Errors from a real node ────────────────────────────────────────────────


def test_a_malformed_query_is_refused_with_a_typed_error(client: Client, index):
    """Every route emits one envelope. A body this client cannot type would
    still surface, but with no `type` to branch on."""
    name = index(q.schema({"tag": q.keyword_field()}))
    with pytest.raises(GnarlError) as caught:
        # `tag` is a keyword; a full-text match needs an analyzed field.
        client.search(name, q.match("tag", "anything"), size=1)
    assert caught.value.type, (
        f"the node answered {caught.value.status} with no error type: "
        f"{caught.value.reason!r}"
    )
    assert caught.value.reason


def test_an_error_carries_a_reason_on_every_route(client: Client):
    for call in (
        lambda: client.get_schema("no-such-index-1111"),
        lambda: client.count("no-such-index-1111"),
        lambda: client.search("no-such-index-1111", q.match_all()),
    ):
        with pytest.raises(GnarlError) as caught:
            call()
        assert caught.value.reason, f"{caught.value.type} came back with no reason"


# ─── Async parity against the same node ─────────────────────────────────────


async def test_the_async_client_works_against_a_real_node(node: str, client, index):
    name = index(q.schema({"headline": q.text_field()}))
    client.index_document(name, {"headline": "async too"}, id="a1")
    searchable(client, name, q.match_all(), 1)

    async with AsyncClient(node, timeout=60.0) as ac:
        res = await ac.search(name, q.match_all())
        assert [h.field_id for h in res.hits] == ["a1"]
        assert (await ac.count(name)) == 1
        assert (await ac.status()).node_id
