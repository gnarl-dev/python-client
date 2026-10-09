"""Retry, environment defaults, the search_after iterator and bulk chunking.

Over a mocked transport. The retry tests replace the client's sleep with a
recorder, so they assert the exact waits without spending them.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
import respx

import gnarl.client as client_module
from gnarl import (
    AsyncClient,
    BulkDoc,
    Client,
    GnarlError,
    IncompleteResult,
    InternalError,
    RateLimited,
    Retry,
    Unavailable,
    failed_items,
)
from gnarl import query as q

from .conftest import BASE


def ok(payload, status: int = 200) -> httpx.Response:
    return httpx.Response(status, json=payload)


def refused(status: int, retry_after: str | None = None, **detail) -> httpx.Response:
    # A 503 from a claim mid-failover carries a type outside the enum, so it
    # classifies by status — which is the case worth testing.
    etype = "rate_limited" if status == 429 else "promotion_pending"
    body: dict = {"error": {"type": etype, "reason": "not now"}}
    if detail:
        body["error"]["detail"] = detail
    headers = {"Retry-After": retry_after} if retry_after is not None else {}
    return httpx.Response(status, json=body, headers=headers)


STATUS = {"node_id": "n", "mode": "single-node"}


@pytest.fixture
def waits(monkeypatch) -> list[float]:
    """Every wait the client asked for, sync or async, in order."""
    recorded: list[float] = []

    async def asleep(seconds: float) -> None:
        recorded.append(seconds)

    monkeypatch.setattr(client_module, "_sleep", recorded.append)
    monkeypatch.setattr(client_module, "_asleep", asleep)
    return recorded


# ─── Retry ──────────────────────────────────────────────────────────────────


@respx.mock
def test_a_429_is_retried_after_exactly_the_retry_after(waits):
    route = respx.get(f"{BASE}/v1/node/status").mock(
        side_effect=[refused(429, "2"), ok(STATUS)]
    )
    with Client(BASE) as c:
        assert c.status().node_id == "n"
    assert route.call_count == 2
    assert waits == [2.0]


@respx.mock
def test_a_503_with_no_hint_backs_off_and_then_gives_up(waits):
    route = respx.get(f"{BASE}/v1/node/status").mock(return_value=refused(503))
    with Client(BASE) as c, pytest.raises(Unavailable):
        c.status()
    # Three attempts by default, so two waits, doubling from the base.
    assert route.call_count == 3
    assert waits == [0.5, 1.0]


@respx.mock
def test_the_hint_in_the_body_is_honoured_when_there_is_no_header(waits):
    """A claim mid-failover answers 503 with `detail.retry_after_secs`."""
    respx.get(f"{BASE}/v1/node/status").mock(
        side_effect=[refused(503, retry_after_secs=1), ok(STATUS)]
    )
    with Client(BASE) as c:
        c.status()
    assert waits == [1.0]


@respx.mock
def test_a_retry_after_past_the_cap_is_raised_not_retried_early(waits):
    """Asking again before the node said to is how a client gets itself
    rate-limited harder. Waiting the cap and retrying would do exactly that."""
    route = respx.get(f"{BASE}/v1/node/status").mock(return_value=refused(429, "120"))
    with Client(BASE) as c, pytest.raises(RateLimited) as caught:
        c.status()
    assert route.call_count == 1
    assert waits == []
    assert caught.value.retry_after == 120.0


@respx.mock
def test_the_cap_and_attempts_are_configurable(waits):
    route = respx.get(f"{BASE}/v1/node/status").mock(return_value=refused(429, "120"))
    with Client(BASE, retry=Retry(attempts=2, max_delay=300)) as c, pytest.raises(RateLimited):
        c.status()
    assert route.call_count == 2
    assert waits == [120.0]


@respx.mock
def test_a_backoff_is_capped(waits):
    respx.get(f"{BASE}/v1/node/status").mock(return_value=refused(503))
    policy = Retry(attempts=4, backoff=10, max_delay=15)
    with Client(BASE, retry=policy) as c, pytest.raises(Unavailable):
        c.status()
    assert waits == [10.0, 15.0, 15.0]


@respx.mock
def test_retry_none_turns_it_off(waits):
    route = respx.get(f"{BASE}/v1/node/status").mock(return_value=refused(429, "1"))
    with Client(BASE, retry=None) as c, pytest.raises(RateLimited):
        c.status()
    assert route.call_count == 1
    assert waits == []


@respx.mock
def test_a_500_is_not_retried(waits):
    """Not "not now" — a failure. Retrying it repeats the failure."""
    route = respx.get(f"{BASE}/v1/node/status").mock(
        return_value=httpx.Response(
            500, json={"error": {"type": "internal_error", "reason": "boom"}}
        )
    )
    with Client(BASE) as c, pytest.raises(InternalError):
        c.status()
    assert route.call_count == 1


@respx.mock
def test_a_write_that_may_have_landed_is_never_resent(waits):
    """A second `index_document` without an id is a second document."""
    route = respx.post(f"{BASE}/v1/indexes/places/_doc").mock(return_value=refused(503))
    with Client(BASE) as c, pytest.raises(Unavailable):
        c.index_document("places", {"name": "x"})
    assert route.call_count == 1
    assert waits == []


@respx.mock
def test_bulk_and_remember_are_not_retried(waits):
    bulk = respx.post(f"{BASE}/v1/indexes/places/_bulk").mock(return_value=refused(429, "1"))
    remember = respx.post(f"{BASE}/v1/memory/remember").mock(return_value=refused(429, "1"))
    with Client(BASE) as c:
        with pytest.raises(RateLimited):
            c.bulk("places", [{"n": 1}])
        with pytest.raises(RateLimited):
            c.memory.remember("x")
    assert (bulk.call_count, remember.call_count) == (1, 1)


SEARCH = {
    "hits": {"total": {"value": 0, "relation": "eq"}, "hits": []},
    "took": 1,
    "partial": False,
    "coverage": {"expected_claims": 1, "served_claims": 1, "skipped_claims": []},
}


@respx.mock
def test_a_search_is_a_post_that_is_retried(waits):
    route = respx.post(f"{BASE}/v1/indexes/places/_search").mock(
        side_effect=[refused(429, "0"), ok(SEARCH)]
    )
    with Client(BASE) as c:
        c.search("places", q.match_all())
    assert route.call_count == 2
    # The same body both times.
    assert route.calls[0].request.content == route.calls[1].request.content


@respx.mock
def test_recall_and_namespace_search_are_retried(waits):
    recall = respx.post(f"{BASE}/v1/memory/recall").mock(
        side_effect=[refused(503), ok({"namespace": "a", "count": 0, "embedder": "e",
                                       "memories": []})]
    )
    ns = respx.post(f"{BASE}/v1/namespaces/t/_search").mock(
        side_effect=[refused(503), ok(SEARCH)]
    )
    with Client(BASE) as c:
        c.memory.recall("q")
        c.namespaces.search("t")
    assert (recall.call_count, ns.call_count) == (2, 2)


@respx.mock
def test_put_and_delete_are_retried(waits):
    put = respx.put(f"{BASE}/v1/indexes/places").mock(side_effect=[refused(503), ok({})])
    delete = respx.delete(f"{BASE}/v1/indexes/places").mock(
        side_effect=[refused(503), ok({})]
    )
    with Client(BASE) as c:
        c.create_index("places", q.schema({"n": q.integer_field()}))
        c.delete_index("places")
    assert (put.call_count, delete.call_count) == (2, 2)


@respx.mock
def test_the_async_client_retries_the_same_way(waits):
    route = respx.get(f"{BASE}/v1/node/status").mock(
        side_effect=[refused(429, "3"), refused(503), ok(STATUS)]
    )

    async def go():
        async with AsyncClient(BASE) as c:
            return await c.status()

    assert asyncio.run(go()).node_id == "n"
    assert route.call_count == 3
    assert waits == [3.0, 1.0]


@respx.mock
def test_the_async_client_does_not_resend_a_write(waits):
    route = respx.post(f"{BASE}/v1/indexes/places/_doc").mock(return_value=refused(503))

    async def go():
        async with AsyncClient(BASE) as c:
            await c.index_document("places", {"n": 1})

    with pytest.raises(Unavailable):
        asyncio.run(go())
    assert route.call_count == 1


@pytest.mark.parametrize("bad", [{"attempts": 0}, {"max_delay": -1}, {"backoff": -1}])
def test_a_nonsense_policy_is_refused(bad):
    with pytest.raises(ValueError):
        Retry(**bad)


# ─── Environment defaults ───────────────────────────────────────────────────


@respx.mock
def test_the_address_and_token_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("GNARL_URL", BASE)
    monkeypatch.setenv("GNARL_TOKEN", "from-env")
    route = respx.get(f"{BASE}/v1/node/status").mock(return_value=ok(STATUS))
    with Client() as c:
        c.ping()
    assert route.calls.last.request.headers["Authorization"] == "Bearer from-env"


def test_no_address_anywhere_says_how_to_give_one(monkeypatch):
    monkeypatch.delenv("GNARL_URL", raising=False)
    with pytest.raises(ValueError, match="GNARL_URL"):
        Client()
    with pytest.raises(ValueError, match="GNARL_URL"):
        AsyncClient()


def test_an_explicit_address_wins(monkeypatch):
    monkeypatch.setenv("GNARL_URL", "https://elsewhere.test")
    assert Client(BASE)._base == BASE


@respx.mock
def test_an_explicit_token_wins_and_an_empty_one_sends_none(monkeypatch):
    monkeypatch.setenv("GNARL_TOKEN", "from-env")
    route = respx.get(f"{BASE}/v1/node/status").mock(return_value=ok(STATUS))
    with Client(BASE, token="explicit") as c:
        c.ping()
    assert route.calls.last.request.headers["Authorization"] == "Bearer explicit"
    with Client(BASE, token="") as c:
        c.ping()
    assert "Authorization" not in route.calls.last.request.headers


def test_the_async_client_reads_the_environment_too(monkeypatch):
    monkeypatch.setenv("GNARL_URL", BASE)
    monkeypatch.setenv("GNARL_TOKEN", "t")
    c = AsyncClient()
    assert (c._base, c._token) == (BASE, "t")
    asyncio.run(c.aclose())


# ─── iter_search ────────────────────────────────────────────────────────────


def page(*ids: str, partial: bool = False) -> httpx.Response:
    hits = [{"_id": i, "sort": [int(i[1:]), i]} for i in ids]
    body = {**SEARCH, "hits": {"total": {"value": 9, "relation": "gte"}, "hits": hits}}
    if partial:
        body["partial"] = True
    return ok(body)


@respx.mock
def test_iter_search_follows_the_cursor_to_an_empty_page(call):
    route = respx.post(f"{BASE}/v1/indexes/places/_search").mock(
        side_effect=[page("d1", "d2"), page("d3"), page()]
    )

    def collect(c):
        it = c.iter_search("places", q.match_all(), sort=[{"n": {"order": "asc"}}], page_size=2)
        if isinstance(c, AsyncClient):
            async def drain():
                return [h.field_id async for h in it]
            return drain()
        return [h.field_id for h in it]

    assert call(collect) == ["d1", "d2", "d3"]
    bodies = [json.loads(r.request.content) for r in route.calls]
    assert [b.get("search_after") for b in bodies] == [None, [2, "d2"], [3, "d3"]]
    assert all(b["size"] == 2 and b["sort"] == [{"n": {"order": "asc"}}] for b in bodies)


@respx.mock
def test_a_short_page_does_not_end_the_iteration():
    """A page can be short because a claim missed its deadline; stopping there
    would lose the rows after it without a word."""
    respx.post(f"{BASE}/v1/indexes/places/_search").mock(
        side_effect=[page("d1"), page("d2", "d3"), page()]
    )
    with Client(BASE) as c:
        got = [h.field_id for h in c.iter_search("places", sort=["n"], page_size=2)]
    assert got == ["d1", "d2", "d3"]


@respx.mock
def test_iter_search_can_demand_complete_pages():
    respx.post(f"{BASE}/v1/indexes/places/_search").mock(
        side_effect=[page("d1", "d2", partial=True)]
    )
    with Client(BASE) as c, pytest.raises(IncompleteResult):
        list(c.iter_search("places", sort=["n"], page_size=2, require_complete=True))


def test_iter_search_needs_a_sort():
    with Client(BASE) as c, pytest.raises(ValueError, match="sort"):
        next(iter(c.iter_search("places", sort=[])))


@respx.mock
def test_a_hit_with_no_cursor_is_an_error_not_an_end():
    respx.post(f"{BASE}/v1/indexes/places/_search").mock(
        return_value=ok({**SEARCH, "hits": {"total": {"value": 1, "relation": "eq"},
                                            "hits": [{"_id": "d1"}]}})
    )
    with Client(BASE) as c, pytest.raises(GnarlError, match="cursor"):
        list(c.iter_search("places", sort=["n"]))


@respx.mock
def test_a_cursor_that_does_not_move_is_an_error_not_a_loop():
    """A node that ignored `search_after` would return the first page forever."""
    respx.post(f"{BASE}/v1/indexes/places/_search").mock(return_value=page("d1", "d2"))
    with Client(BASE) as c, pytest.raises(GnarlError, match="no progress"):
        list(c.iter_search("places", sort=["n"], page_size=2))


@respx.mock
def test_namespace_iter_search(call):
    route = respx.post(f"{BASE}/v1/namespaces/t/_search").mock(
        side_effect=[page("d1", "d2"), page()]
    )

    def collect(c):
        it = c.namespaces.iter_search("t", sort=["n"], page_size=2)
        if isinstance(c, AsyncClient):
            async def drain():
                return [h.field_id async for h in it]
            return drain()
        return [h.field_id for h in it]

    assert call(collect) == ["d1", "d2"]
    assert json.loads(route.calls.last.request.content)["search_after"] == [2, "d2"]


# ─── bulk_chunked ───────────────────────────────────────────────────────────


def bulk_result(ids, ack="visible_for_search", failed=()):
    items = [
        {"_id": i, "status": 400, "error": {"type": "validation_error", "reason": "bad"}}
        if i in failed else {"_id": i, "status": 201}
        for i in ids
    ]
    return ok({"items": items, "errors": bool(failed), "ack": ack})


@respx.mock
def test_bulk_chunked_splits_and_merges_in_order(call):
    route = respx.post(f"{BASE}/v1/indexes/places/_bulk").mock(
        side_effect=[
            bulk_result(["a", "b"]),
            bulk_result(["c", "d"], ack="accepted", failed=("d",)),
            bulk_result(["e"]),
        ]
    )
    docs = (BulkDoc(i, {"n": n}) for n, i in enumerate("abcde"))  # a generator
    res = call(lambda c: c.bulk_chunked("places", docs, chunk_size=2))
    sizes = [len(json.loads(r.request.content)["documents"]) for r in route.calls]
    assert sizes == [2, 2, 1]
    assert [i.field_id for i in res.items] == list("abcde")
    assert res.errors is True
    assert [i.field_id for i in failed_items(res)] == ["d"]
    # The weakest level any chunk reached, never the strongest.
    assert res.ack.value == "accepted"


@respx.mock
def test_namespace_bulk_chunked_carries_wait_for(call):
    route = respx.post(f"{BASE}/v1/namespaces/t/_bulk").mock(
        side_effect=[bulk_result(["a"]), bulk_result(["b"])]
    )
    res = call(
        lambda c: c.namespaces.bulk_chunked(
            "t", [{"n": 1}, {"n": 2}], chunk_size=1, wait_for="visible"
        )
    )
    assert route.call_count == 2
    assert all(r.request.url.params["wait_for"] == "visible" for r in route.calls)
    assert res.errors is False


def test_bulk_chunked_refuses_nothing_to_send(call):
    with pytest.raises(ValueError):
        call(lambda c: c.bulk_chunked("places", []))


def test_bulk_chunked_refuses_a_zero_chunk(call):
    with pytest.raises(ValueError):
        call(lambda c: c.bulk_chunked("places", [{"n": 1}], chunk_size=0))
