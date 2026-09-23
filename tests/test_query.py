"""Unit tests for the query builders.

Each builder must produce exactly one top-level key. Setting two is accepted
by the type system and rejected by the node, so the shape is pinned here
rather than discovered in production.
"""

from __future__ import annotations

import pytest

from gnarl import query as q


def wire(query) -> dict:
    """The JSON the client would actually send."""
    return query.model_dump(mode="json", by_alias=True, exclude_none=True)


def sole_key(query) -> str:
    keys = list(wire(query))
    assert len(keys) == 1, f"a query must set exactly one key, got {keys}"
    return keys[0]


@pytest.mark.parametrize(
    ("built", "key"),
    [
        (q.match("body", "tide"), "match"),
        (q.match_phrase("body", "king tide"), "match_phrase"),
        (q.multi_match(["body", "name"], "tide"), "multi_match"),
        (q.term("status", "published"), "term"),
        (q.prefix("name", "syd"), "prefix"),
        (q.wildcard("name", "syd*"), "wildcard"),
        (q.range_("depth", gte=10), "range"),
        (q.exists("name"), "exists"),
        (q.match_all(), "match_all"),
        (q.geo_distance("location", -33.8, 151.2, 1000.0), "geo_distance"),
        (q.knn("embedding", [0.1, 0.2], 5), "knn"),
        (q.bool_().must(q.match_all()).build(), "bool"),
        (q.mmr(q.match_all(), "embedding"), "mmr"),
    ],
)
def test_each_builder_sets_exactly_one_key(built, key):
    assert sole_key(built) == key


def test_match_sends_the_bare_string_form():
    """The wire accepts a string or an object; every node understands both,
    and the string is the form that has always worked."""
    assert wire(q.match("body", "tide")) == {"match": {"body": "tide"}}


def test_match_phrase_with_slop_uses_the_object_form():
    assert wire(q.match_phrase("body", "king tide", slop=2)) == {
        "match_phrase": {"body": {"query": "king tide", "slop": 2}}
    }


def test_match_phrase_without_slop_stays_a_string():
    assert wire(q.match_phrase("body", "king tide")) == {
        "match_phrase": {"body": "king tide"}
    }


def test_geo_distance_is_flat_lat_lon_and_metres():
    """Not a nested `location` object, and not a `"5km"` string.

    Both are Elasticsearch habits that produce a request the node rejects.
    """
    assert wire(q.geo_distance("location", -33.8688, 151.2093, 5000.0)) == {
        "geo_distance": {
            "field": "location",
            "lat": -33.8688,
            "lon": 151.2093,
            "radius_meters": 5000.0,
        }
    }


def test_a_latitude_out_of_range_is_refused_before_the_wire():
    with pytest.raises(ValueError):
        q.geo_distance("location", 91.0, 0.0, 100.0)


def test_a_non_positive_radius_is_refused():
    with pytest.raises(ValueError):
        q.geo_distance("location", 0.0, 0.0, 0.0)


def test_range_refuses_contradictory_bounds():
    """gt with gte on the same side is rejected by the node; reject it here so
    the error names the mistake instead of arriving as a 400."""
    with pytest.raises(ValueError, match="gt and gte"):
        q.range_("depth", gt=1, gte=2)
    with pytest.raises(ValueError, match="lt and lte"):
        q.range_("depth", lt=1, lte=2)


def test_range_requires_at_least_one_bound():
    with pytest.raises(ValueError, match="at least one bound"):
        q.range_("depth")


def test_range_keeps_a_zero_bound():
    """`gte=0` is a real bound. A falsy check here would silently drop it."""
    assert wire(q.range_("depth", gte=0)) == {"range": {"depth": {"gte": 0}}}


def test_bool_omits_the_clauses_it_was_not_given():
    built = wire(q.bool_().filter(q.term("status", "published")).build())
    assert built == {"bool": {"filter": [{"term": {"status": "published"}}]}}
    assert "must" not in built["bool"]


def test_bool_accumulates_across_calls():
    built = q.bool_()
    built.must(q.match("body", "a"))
    built.must(q.match("body", "b"))
    built.filter(q.term("s", "x"), q.term("t", "y"))
    wired = wire(built.build())["bool"]
    assert len(wired["must"]) == 2
    assert len(wired["filter"]) == 2


def test_bool_nests():
    inner = q.bool_().should(q.match("body", "a")).build()
    outer = wire(q.bool_().must(inner).build())
    assert outer["bool"]["must"][0]["bool"]["should"][0] == {"match": {"body": "a"}}


def test_knn_omits_the_tuning_dials_when_unset():
    assert wire(q.knn("embedding", [0.1], 3)) == {
        "knn": {"field": "embedding", "vector": [0.1], "k": 3}
    }


def test_knn_rejects_k_below_one():
    with pytest.raises(ValueError):
        q.knn("embedding", [0.1], 0)


def test_mmr_uses_the_lambda_alias():
    """The wire field is `lambda`, which is a Python keyword."""
    assert wire(q.mmr(q.match_all(), "embedding", lambda_=0.3))["mmr"]["lambda"] == 0.3


def test_schema_helpers_produce_the_declared_types():
    s = q.schema(
        {
            "name": q.text_field(),
            "status": q.keyword_field(),
            "location": q.geo_point_field(),
            "embedding": q.dense_vector_field(384),
        }
    )
    wired = s.model_dump(mode="json", by_alias=True, exclude_none=True)["fields"]
    assert wired["name"]["type"] == "text"
    assert wired["status"]["type"] == "keyword"
    assert wired["location"]["type"] == "geo_point"
    assert wired["embedding"]["type"] == "dense_vector"
    assert wired["embedding"]["dimensions"] == 384


def test_dense_vector_requires_a_positive_dimensionality():
    with pytest.raises(ValueError):
        q.dense_vector_field(0)


def test_geo_point_value_is_flat_lat_lon():
    assert q.geo_point(-33.8568, 151.2153) == {"lat": -33.8568, "lon": 151.2153}
