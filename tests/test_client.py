"""Integration tests for the full request/response path.

These drive the real client over a mocked transport, so they cover everything
between a method call and the bytes on the wire: URL construction, headers,
body shape, status handling, model parsing. What they cannot prove is that the
server agrees — that is `tests/conformance/`, which drives a real node.
"""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from gnarl import (
    AsyncClient,
    BulkDoc,
    Client,
    IncompleteResult,
    InternalError,
    NotFound,
    failed_items,
)
from gnarl import query as q

BASE = "http://node.test"


@pytest.fixture
def client():
    with Client(BASE) as c:
        yield c


def ok(payload: dict) -> httpx.Response:
    return httpx.Response(200, json=payload)


def envelope(status: int, type_: str, reason: str = "nope") -> httpx.Response:
    return httpx.Response(status, json={"error": {"type": type_, "reason": reason}})


def search_payload(**over) -> dict:
    body = {
        "hits": {"total": {"value": 1, "relation": "eq"}, "hits": []},
        "took": 3,
        "partial": False,
        "coverage": {"expected_claims": 1, "served_claims": 1, "skipped_claims": []},
    }
    body.update(over)
    return body


# ─── Addressing and headers ─────────────────────────────────────────────────


def test_a_scheme_less_address_becomes_https():
    """Never a silent downgrade to plaintext for someone who wrote a hostname."""
    c = Client("node.example.com")
    assert c._base == "https://node.example.com"


def test_a_trailing_slash_does_not_double_up():
    assert Client("http://node.test/").  _base == "http://node.test"


def test_an_empty_address_is_refused():
    with pytest.raises(ValueError):
        Client("")


@respx.mock
def test_a_token_is_sent_as_a_bearer_header():
    route = respx.get(f"{BASE}/v1/node/status").mock(return_value=ok({}))
    with Client(BASE, token="s3cret") as c:
        c.ping()
    assert route.calls.last.request.headers["Authorization"] == "Bearer s3cret"


@respx.mock
def test_no_token_means_no_authorization_header(client):
    route = respx.get(f"{BASE}/v1/node/status").mock(return_value=ok({}))
    client.ping()
    assert "Authorization" not in route.calls.last.request.headers


@respx.mock
def test_a_body_carries_a_json_content_type(client):
    route = respx.put(f"{BASE}/v1/indexes/places").mock(return_value=ok({}))
    client.create_index("places", q.schema({"name": q.keyword_field()}))
    assert route.calls.last.request.headers["Content-Type"] == "application/json"


@respx.mock
def test_a_bodiless_request_sends_no_content_type(client):
    route = respx.get(f"{BASE}/v1/node/status").mock(return_value=ok({}))
    client.ping()
    assert "Content-Type" not in route.calls.last.request.headers


# ─── Indexes ────────────────────────────────────────────────────────────────


@respx.mock
def test_create_index_sends_the_schema_under_a_schema_key(client):
    route = respx.put(f"{BASE}/v1/indexes/places").mock(return_value=ok({}))
    client.create_index(
        "places", q.schema({"name": q.keyword_field(), "loc": q.geo_point_field()})
    )
    sent = json.loads(route.calls.last.request.content)
    assert sent == {
        "schema": {"fields": {"name": {"type": "keyword"}, "loc": {"type": "geo_point"}}}
    }


@respx.mock
def test_a_text_field_does_not_carry_vector_options(client):
    """Defaults that apply to one field type only are not sent for another."""
    route = respx.put(f"{BASE}/v1/indexes/docs").mock(return_value=ok({}))
    client.create_index("docs", q.schema({"body": q.text_field()}))
    sent = json.loads(route.calls.last.request.content)
    assert sent["schema"]["fields"]["body"] == {"type": "text"}


@respx.mock
def test_a_dense_vector_field_does_carry_them(client):
    route = respx.put(f"{BASE}/v1/indexes/docs").mock(return_value=ok({}))
    client.create_index("docs", q.schema({"e": q.dense_vector_field(8)}))
    field = json.loads(route.calls.last.request.content)["schema"]["fields"]["e"]
    assert field == {
        "type": "dense_vector",
        "dimensions": 8,
        "distance_metric": "cosine",
        "quantization": "none",
    }


@respx.mock
def test_an_index_name_is_escaped_into_the_path(client):
    route = respx.delete(f"{BASE}/v1/indexes/a%2Fb").mock(return_value=ok({}))
    client.delete_index("a/b")
    assert route.called


@respx.mock
def test_index_exists_is_true_on_200(client):
    respx.get(f"{BASE}/v1/indexes/places").mock(return_value=ok({}))
    assert client.index_exists("places") is True


@respx.mock
def test_index_exists_is_false_on_404(client):
    respx.get(f"{BASE}/v1/indexes/places").mock(
        return_value=envelope(404, "index_not_found")
    )
    assert client.index_exists("places") is False


@respx.mock
def test_index_exists_raises_rather_than_answering_false_on_a_500(client):
    """"Could not tell" is not "absent".

    Returning False here is how a caller ends up recreating — or deleting —
    live data because a node was briefly unwell.
    """
    respx.get(f"{BASE}/v1/indexes/places").mock(
        return_value=envelope(500, "internal_error")
    )
    with pytest.raises(InternalError):
        client.index_exists("places")


@respx.mock
def test_index_exists_raises_on_a_transport_failure(client):
    respx.get(f"{BASE}/v1/indexes/places").mock(
        side_effect=httpx.ConnectError("refused")
    )
    with pytest.raises(Exception) as caught:
        client.index_exists("places")
    assert caught.value.type == "transport_error"


@respx.mock
def test_list_indexes_follows_the_cursor_to_the_end(client):
    page1 = {
        "indexes": [{"name": "a", "schema": {"fields": {}}, "claim_count": 1}],
        "next_after": "a",
    }
    page2 = {"indexes": [{"name": "b", "schema": {"fields": {}}, "claim_count": 1}]}
    respx.get(f"{BASE}/v1/indexes", params={"after": "a"}).mock(return_value=ok(page2))
    respx.get(f"{BASE}/v1/indexes").mock(return_value=ok(page1))

    names = [i.name for i in client.list_indexes()]
    assert names == ["a", "b"]


@respx.mock
def test_count_reads_the_count(client):
    respx.get(f"{BASE}/v1/indexes/places/_count").mock(return_value=ok({"count": 42}))
    assert client.count("places") == 42


# ─── Documents ──────────────────────────────────────────────────────────────


@respx.mock
def test_index_document_puts_the_id_beside_the_fields_not_around_them(client):
    route = respx.post(f"{BASE}/v1/indexes/places/_doc").mock(
        return_value=ok({"_id": "sydney", "ack": "accepted"})
    )
    returned = client.index_document("places", {"name": "sydney"}, id="sydney")
    assert json.loads(route.calls.last.request.content) == {
        "name": "sydney",
        "_id": "sydney",
    }
    assert returned == "sydney"


@respx.mock
def test_index_document_without_an_id_sends_none(client):
    route = respx.post(f"{BASE}/v1/indexes/places/_doc").mock(
        return_value=ok({"_id": "generated", "ack": "accepted"})
    )
    assert client.index_document("places", {"name": "sydney"}) == "generated"
    assert "_id" not in json.loads(route.calls.last.request.content)


def test_index_document_refuses_an_empty_id(client):
    """An empty string is a mistake, not a request for a generated id —
    `None` is how you ask for that, and the two must not be the same."""
    with pytest.raises(ValueError, match="empty document id"):
        client.index_document("places", {"name": "x"}, id="")


def test_index_document_refuses_a_non_mapping(client):
    with pytest.raises(TypeError, match="must be a mapping"):
        client.index_document("places", ["not", "a", "document"])


@respx.mock
def test_index_document_does_not_mutate_the_caller_s_dict(client):
    respx.post(f"{BASE}/v1/indexes/places/_doc").mock(
        return_value=ok({"_id": "x", "ack": "accepted"})
    )
    doc = {"name": "sydney"}
    client.index_document("places", doc, id="x")
    assert doc == {"name": "sydney"}


@respx.mock
def test_a_document_id_with_a_slash_is_escaped(client):
    respx.get(f"{BASE}/v1/indexes/places/_doc/a%2Fb").mock(
        return_value=ok({"_id": "a/b", "_source": {"name": "x"}})
    )
    assert client.get_document("places", "a/b") == {"name": "x"}


@respx.mock
def test_a_missing_document_raises_not_found(client):
    respx.get(f"{BASE}/v1/indexes/places/_doc/nope").mock(
        return_value=envelope(404, "document_not_found")
    )
    with pytest.raises(NotFound):
        client.get_document("places", "nope")


@respx.mock
def test_bulk_wraps_the_documents(client):
    route = respx.post(f"{BASE}/v1/indexes/places/_bulk").mock(
        return_value=ok(
            {"items": [{"_id": "1", "status": 201}], "errors": False, "ack": "accepted"}
        )
    )
    client.bulk("places", [{"name": "a"}, BulkDoc("two", {"name": "b"})])
    assert json.loads(route.calls.last.request.content) == {
        "documents": [{"name": "a"}, {"name": "b", "_id": "two"}]
    }


def test_bulk_refuses_an_empty_batch(client):
    with pytest.raises(ValueError, match="no documents"):
        client.bulk("places", [])


@respx.mock
def test_failed_items_finds_the_failures_behind_a_200(client):
    """A bulk request answers 200 with individual items failed. A caller who
    checks only the status loses writes without seeing an error."""
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
    result = client.bulk("places", [{"a": 1}, {"b": 2}])
    failed = failed_items(result)
    assert [f.field_id for f in failed] == ["2"]
    assert failed[0].error.reason == "unknown field"


# ─── Search ─────────────────────────────────────────────────────────────────


@respx.mock
def test_a_plain_search_sends_only_the_query(client):
    """Anything the caller did not ask for must not appear in the body.

    `profile` and `verify` each make the node do work it otherwise skips.
    """
    route = respx.post(f"{BASE}/v1/indexes/places/_search").mock(
        return_value=ok(search_payload())
    )
    client.search("places", q.match_all())
    assert json.loads(route.calls.last.request.content) == {"query": {"match_all": {}}}


@respx.mock
def test_search_options_appear_only_when_asked_for(client):
    route = respx.post(f"{BASE}/v1/indexes/places/_search").mock(
        return_value=ok(search_payload())
    )
    client.search(
        "places",
        q.match_all(),
        size=5,
        from_=10,
        profile=True,
        verify=True,
        deadline_ms=250,
    )
    sent = json.loads(route.calls.last.request.content)
    assert sent["size"] == 5
    assert sent["from"] == 10
    assert sent["profile"] is True
    assert sent["verify"] is True
    assert sent["scope"] == {"deadline_ms": 250}


@respx.mock
def test_a_deadline_of_zero_is_not_sent(client):
    """The node clamps a deadline to its own budget, and zero is not a
    request for "no time" — it is the absence of a request."""
    route = respx.post(f"{BASE}/v1/indexes/places/_search").mock(
        return_value=ok(search_payload())
    )
    client.search("places", q.match_all(), deadline_ms=0)
    assert "scope" not in json.loads(route.calls.last.request.content)


@respx.mock
def test_search_after_and_sort_pass_through(client):
    route = respx.post(f"{BASE}/v1/indexes/places/_search").mock(
        return_value=ok(search_payload())
    )
    client.search("places", q.match_all(), sort=["depth"], search_after=[12, "abc"])
    sent = json.loads(route.calls.last.request.content)
    assert sent["sort"] == ["depth"]
    assert sent["search_after"] == [12, "abc"]


@respx.mock
def test_source_false_is_sent_rather_than_dropped_as_falsy(client):
    route = respx.post(f"{BASE}/v1/indexes/places/_search").mock(
        return_value=ok(search_payload())
    )
    client.search("places", q.match_all(), source=False)
    assert json.loads(route.calls.last.request.content)["_source"] is False


@respx.mock
def test_a_result_exposes_coverage_and_totals(client):
    respx.post(f"{BASE}/v1/indexes/places/_search").mock(
        return_value=ok(
            search_payload(
                hits={
                    "total": {"value": 2, "relation": "gte"},
                    "hits": [
                        {"_id": "a", "_score": 1.0, "_source": {"name": "a"}},
                        {"_id": "b", "_score": 0.5, "_source": {"name": "b"}},
                    ],
                }
            )
        )
    )
    res = client.search("places", q.match_all())
    assert len(res) == 2
    assert [h.field_id for h in res] == ["a", "b"]
    assert res.sources() == [{"name": "a"}, {"name": "b"}]
    assert res.total.value == 2
    assert res.total_is_exact is False
    assert res.coverage.expected_claims == 1


@respx.mock
def test_an_exact_total_reads_as_exact(client):
    respx.post(f"{BASE}/v1/indexes/places/_search").mock(
        return_value=ok(search_payload())
    )
    assert client.search("places", q.match_all()).total_is_exact is True


@respx.mock
def test_require_complete_raises_on_a_partial_result(client):
    respx.post(f"{BASE}/v1/indexes/places/_search").mock(
        return_value=ok(
            search_payload(
                partial=True,
                coverage={
                    "expected_claims": 4,
                    "served_claims": 3,
                    "skipped_claims": [{"claim_id": 2, "reason": "timeout"}],
                },
            )
        )
    )
    with pytest.raises(IncompleteResult) as caught:
        client.search("places", q.match_all(), require_complete=True)
    # The partial result survives the raise, so a caller can degrade to it
    # deliberately rather than lose the work.
    assert caught.value.response.coverage.served_claims == 3
    assert "3 of 4" in caught.value.reason


@respx.mock
def test_require_complete_catches_a_gap_even_when_partial_is_false(client):
    """The arithmetic is checked as well as the flag.

    The point of asking for completeness is not to trust one boolean.
    """
    respx.post(f"{BASE}/v1/indexes/places/_search").mock(
        return_value=ok(
            search_payload(
                partial=False,
                coverage={
                    "expected_claims": 4,
                    "served_claims": 3,
                    "skipped_claims": [],
                },
            )
        )
    )
    with pytest.raises(IncompleteResult):
        client.search("places", q.match_all(), require_complete=True)


@respx.mock
def test_a_complete_result_does_not_raise(client):
    respx.post(f"{BASE}/v1/indexes/places/_search").mock(
        return_value=ok(search_payload())
    )
    assert client.search("places", q.match_all(), require_complete=True).partial is False


@respx.mock
def test_without_require_complete_a_partial_result_is_returned_not_raised(client):
    respx.post(f"{BASE}/v1/indexes/places/_search").mock(
        return_value=ok(search_payload(partial=True))
    )
    assert client.search("places", q.match_all()).partial is True


# ─── Node ───────────────────────────────────────────────────────────────────


@respx.mock
def test_status_parses_the_documented_fields(client):
    respx.get(f"{BASE}/v1/node/status").mock(
        return_value=ok(
            {
                "node_id": "ab" * 32,
                "mode": "single-node",
                "peers": 3,
                "reachable_peers": 2,
                "claims": 8,
                "serving_ready": 8,
                "proof_verified": 8,
                "data_dir": "/var/lib/lucenia",
            }
        )
    )
    s = client.status()
    assert s.node_id == "ab" * 32
    assert s.mode == "single-node"
    assert (s.peers, s.reachable_peers) == (3, 2)
    assert s.data_dir == "/var/lib/lucenia"


@respx.mock
def test_status_keeps_the_raw_body_for_fields_this_client_does_not_name(client):
    respx.get(f"{BASE}/v1/node/status").mock(
        return_value=ok({"node_id": "x", "mode": "lan", "something_new": 7})
    )
    assert client.status().raw["something_new"] == 7


@respx.mock
def test_version_is_a_different_endpoint_from_status(client):
    respx.get(f"{BASE}/v1/node/version").mock(
        return_value=ok({"version": "0.1.0", "commit": "abc1234"})
    )
    v = client.version()
    assert (v.version, v.commit) == ("0.1.0", "abc1234")


# ─── Async parity ───────────────────────────────────────────────────────────


@respx.mock
async def test_the_async_client_reaches_the_same_endpoint():
    route = respx.post(f"{BASE}/v1/indexes/places/_search").mock(
        return_value=ok(search_payload())
    )
    async with AsyncClient(BASE) as c:
        res = await c.search("places", q.match_all())
    assert res.partial is False
    assert json.loads(route.calls.last.request.content) == {"query": {"match_all": {}}}


@respx.mock
async def test_the_async_client_raises_the_same_errors():
    respx.get(f"{BASE}/v1/indexes/places/_doc/nope").mock(
        return_value=envelope(404, "document_not_found")
    )
    async with AsyncClient(BASE) as c:
        with pytest.raises(NotFound):
            await c.get_document("places", "nope")


@respx.mock
async def test_the_async_client_enforces_completeness_the_same_way():
    respx.post(f"{BASE}/v1/indexes/places/_search").mock(
        return_value=ok(search_payload(partial=True))
    )
    async with AsyncClient(BASE) as c:
        with pytest.raises(IncompleteResult):
            await c.search("places", q.match_all(), require_complete=True)
