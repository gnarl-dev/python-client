"""Conformance for the GA surface: entitlement, namespaces, memory, index
maintenance, snapshots, and the iterator and chunking helpers — against a real
node. See ``conftest.py`` for how one is found.
"""

from __future__ import annotations

import asyncio
import hashlib
import tempfile
import time

import httpx
import pytest

from gnarl import (
    AsyncClient,
    BulkDoc,
    Client,
    GnarlError,
    IncompleteResult,
    InternalError,
    NotFound,
    ValidationError,
    failed_items,
)
from gnarl import query as q

from .conftest import _verify_for, until

pytestmark = pytest.mark.conformance


@pytest.fixture
def ns(request, client: Client):
    """A namespace unique to this test, erased afterwards."""
    digest = hashlib.sha256(request.node.nodeid.encode()).hexdigest()[:8]
    names: list[str] = []

    def make(suffix: str = "") -> str:
        name = f"conf-ns-{digest}{suffix}-{time.time_ns() % 1_000_000_000}"
        names.append(name)
        return name

    yield make
    for name in names:
        try:
            client.namespaces.delete(name)
        except GnarlError:
            pass


def _is_local(addr: str) -> bool:
    return httpx.URL(addr).host in ("127.0.0.1", "::1", "localhost")


# ─── Entitlement ────────────────────────────────────────────────────────────


def test_entitlement_status_is_one_of_four_coherent_states(client: Client):
    ent = client.entitlement()
    assert ent.state in {"active", "refused", "unenforced", "none"}
    if not ent.enforced:
        # Nothing is gated on a build with no keys, so nothing can be active.
        assert ent.active is False
    if ent.active:
        assert ent.refused is None
    if ent.not_after is not None:
        # SECONDS: after 2001 and before 5138. A millisecond value is 1000x
        # larger and fails the upper bound.
        assert 1_000_000_000 < ent.not_after < 100_000_000_000, ent.not_after


def test_a_junk_activation_key_is_refused_with_a_reason(client: Client):
    """Refused, never stored: on an unenforced build because there is nothing
    to activate, on an enforced one because the key does not parse. Either
    way the caller gets a ValidationError that says why."""
    before = client.entitlement()
    with pytest.raises(ValidationError) as caught:
        client.activate_entitlement("gnarl-ent1.this-is-not-a-key")
    assert caught.value.status == 400
    assert caught.value.reason
    assert client.entitlement().state == before.state, "a refused key changed state"


# ─── Namespaces ─────────────────────────────────────────────────────────────


def test_a_namespace_write_is_searchable_once_visible(client: Client, ns):
    name = ns()
    doc_id = client.namespaces.index_document(
        name, {"subject": "invoice overdue", "amount": 12}, id="n1", wait_for="visible"
    )
    assert doc_id == "n1", "the tenant's own id must come back, not the scoped form"
    # wait_for=visible means the very next search sees it — no polling.
    res = client.namespaces.search(name, q.match("subject", "invoice"))
    assert [h.field_id for h in res] == ["n1"]
    assert client.namespaces.get_document(name, "n1")["subject"] == "invoice overdue"


def test_a_namespace_cannot_see_another_namespaces_documents(client: Client, ns):
    a, b = ns("a"), ns("b")
    client.namespaces.index_document(a, {"subject": "alpha only"}, id="x", wait_for="visible")
    client.namespaces.index_document(b, {"subject": "beta only"}, id="x", wait_for="visible")
    hits = client.namespaces.search(a, q.match_all(), size=50).hits
    assert [h.field_source["subject"] for h in hits] == ["alpha only"]


def test_a_namespace_bulk_reports_per_item_and_lands(client: Client, ns):
    name = ns()
    result = client.namespaces.bulk(
        name, [BulkDoc("b1", {"n": 1}), BulkDoc("b2", {"n": 2})], wait_for="visible"
    )
    assert failed_items(result) == []
    assert [i.field_id for i in result.items] == ["b1", "b2"]
    assert len(client.namespaces.search(name, q.match_all())) == 2


def test_a_namespace_appears_in_the_listing(client: Client, ns):
    name = ns()
    client.namespaces.index_document(name, {"n": 1}, wait_for="visible")
    listing = client.namespaces.list()
    assert name in listing.names()
    assert listing.partial is False, "a single node cannot have an unreachable peer"


def test_a_namespace_key_status_reports_unkeyed(client: Client, ns):
    name = ns()
    client.namespaces.index_document(name, {"n": 1}, wait_for="visible")
    status = client.namespaces.key_status(name)
    assert status.namespace == name
    assert (status.encrypted, status.unlocked) == (False, False)


def test_a_deleted_namespace_document_goes(client: Client, ns):
    name = ns()
    client.namespaces.index_document(name, {"n": 1}, id="gone", wait_for="visible")
    client.namespaces.delete_document(name, "gone")

    def absent() -> bool:
        try:
            client.namespaces.get_document(name, "gone")
        except NotFound:
            return True
        return False

    assert until(absent)


def test_promoting_a_namespace_keeps_its_documents(client: Client, ns):
    name = ns()
    client.namespaces.index_document(name, {"subject": "keep me"}, id="p1", wait_for="visible")
    out = client.namespaces.promote(name)
    assert out, "promotion returned an empty body"
    assert until(
        lambda: [h.field_id for h in client.namespaces.search(name, q.match_all())] == ["p1"]
    ), "a document was lost across promotion"


@pytest.mark.xfail(
    strict=True,
    raises=IncompleteResult,
    reason=(
        "SERVER DEFECT: in a namespace whose first document lacked field `n`, "
        "a sort or range on `n` — added later by dynamic mapping — fails on "
        "the claim that took that first document ('phase2_execution_failed'). "
        "The node reports it honestly as partial, so require_complete raises; "
        "without it the documents in that claim are silently absent from a "
        "sorted read. Strict, so this goes red the day it is fixed."
    ),
)
def test_a_field_added_after_the_first_write_sorts_on_every_claim(client: Client, ns):
    name = ns()
    client.namespaces.index_document(name, {"subject": "no n here"}, wait_for="visible")
    # Forty documents, so the first one's claim almost surely receives some
    # (a miss is (3/4)^40, about 1 in 100,000).
    client.namespaces.bulk(
        name, [BulkDoc(f"s{i:02}", {"n": i}) for i in range(40)], wait_for="visible"
    )
    res = client.namespaces.search(name, sort=["n"], size=100, require_complete=True)
    assert len(res) == 40


# ─── Memory ─────────────────────────────────────────────────────────────────


def test_remember_then_recall(client: Client):
    space = f"conf-mem-{time.time_ns() % 1_000_000_000}"
    marker = f"the skiff is moored at jetty {time.time_ns() % 100_000}"
    stored = client.memory.remember(marker, namespace=space, tags={"suite": "conformance"})
    assert stored.id
    assert stored.embedder, "the node did not say which embedder produced the vector"

    found: list = []

    def recalled() -> bool:
        found[:] = client.memory.recall("where is the skiff moored", namespace=space).memories
        return any(mem.id == stored.id for mem in found)

    assert until(recalled), f"a remembered fact was not recalled: {found}"
    hit = next(mem for mem in found if mem.id == stored.id)
    assert hit.content == marker
    # Flattened on the way out.
    assert hit.tags is None or "suite=conformance" in hit.tags


def test_answer_draws_on_recalled_memories(client: Client):
    space = f"conf-ans-{time.time_ns() % 1_000_000_000}"
    stored = client.memory.remember("the spare key is under the blue pot", namespace=space)
    assert until(
        lambda: any(
            mem.id == stored.id
            for mem in client.memory.recall("spare key", namespace=space).memories
        )
    )
    answer = client.memory.answer("where is the spare key", namespace=space)
    assert answer.get("count", 0) >= 1, answer


def test_bootstrap_answers(client: Client):
    """The description declares no body; the node requires one. The client
    sends `{}`, and this proves that is what makes it work."""
    assert isinstance(client.memory.bootstrap(), dict)


def test_ingest_messages_stores_the_transcript(client: Client):
    space = f"conf-ing-{time.time_ns() % 1_000_000_000}"
    out = client.memory.ingest_messages(
        [{"role": "me", "body": "pick up the tide tables"}], namespace=space
    )
    assert out.get("ids"), out


# ─── Index maintenance ──────────────────────────────────────────────────────


def test_forcemerge_merges_and_reports(client: Client, index):
    name = index(q.schema({"n": q.integer_field()}))
    # Enough documents that every claim holds some: see the test below for
    # what happens when one does not. With 64 documents over 4 claims, the
    # chance a claim is left empty is about 1 in 25 million.
    result = client.bulk(name, [{"n": i} for i in range(64)])
    assert failed_items(result) == []
    out = client.forcemerge(name, max_num_segments=1)
    assert isinstance(out.segments, int) and out.segments >= 0
    assert out.partial is False


@pytest.mark.xfail(
    strict=True,
    raises=InternalError,
    reason=(
        "SERVER DEFECT: _forcemerge answers 500 'no engine for local claim N' "
        "when any locally-held claim of the index has not been written yet. "
        "Engines are created on a claim's first write, and force_merge_index "
        "treats a claim with no engine as an error rather than as nothing to "
        "merge — so a new index, or a small one whose documents did not reach "
        "every claim, cannot be merged at all. Strict, so this goes red the "
        "day it is fixed."
    ),
)
def test_forcemerge_on_an_index_with_an_unwritten_claim(client: Client, index):
    name = index(q.schema({"n": q.integer_field()}))
    out = client.forcemerge(name)
    assert out.partial is False


def test_forcemerge_on_a_missing_index_is_not_found(client: Client):
    with pytest.raises(NotFound):
        client.forcemerge("definitely-not-here-0000")


def test_an_index_policy_round_trips(client: Client, index):
    name = index(q.schema({"n": q.integer_field()}))
    assert client.get_policy(name).placement.value == "mesh", "the default is mesh"
    narrowed = client.put_policy(name, placement="local")
    assert narrowed.placement.value == "local"
    assert client.get_policy(name).placement.value == "local"


def test_an_unknown_route_is_not_found_rather_than_a_200(client: Client):
    """A mistyped path once answered 200 with the Console's HTML. The node
    this suite starts is headless, and there it currently answers a bare 404
    with no body rather than `route_not_found` — the status still classifies
    it, which is what this pins."""
    with pytest.raises(NotFound):
        client._do("GET", "/v1/indices")


# ─── Iterating and chunking ─────────────────────────────────────────────────


def test_iter_search_reads_every_document_exactly_once(client: Client, index):
    name = index(q.schema({"n": q.integer_field()}))
    result = client.bulk_chunked(
        name, (BulkDoc(f"d{i:03}", {"n": i}) for i in range(57)), chunk_size=20
    )
    assert failed_items(result) == []
    assert len(result.items) == 57
    assert until(lambda: client.count(name) == 57)

    seen = [h.field_id for h in client.iter_search(name, sort=["n"], page_size=10)]
    assert seen == [f"d{i:03}" for i in range(57)], "rows skipped, repeated or reordered"


def test_namespace_bulk_chunked_and_iter_search(client: Client, ns):
    name = ns()
    result = client.namespaces.bulk_chunked(
        name, [BulkDoc(f"k{i:02}", {"n": i}) for i in range(13)], chunk_size=5,
        wait_for="visible",
    )
    assert failed_items(result) == []
    got = [h.field_id for h in client.namespaces.iter_search(name, sort=["n"], page_size=4)]
    assert got == [f"k{i:02}" for i in range(13)]


# ─── Snapshots ──────────────────────────────────────────────────────────────


def test_a_snapshot_round_trip(client: Client, node: str, index):
    """Register a directory repository, snapshot an index into it, read the
    signed descriptor back, schedule and unschedule it, and clean up.

    Needs the node's filesystem to be this machine's, so it runs only against
    a node on loopback."""
    if not _is_local(node):
        pytest.skip("a filesystem repository needs the node on this machine")
    name = index(q.schema({"n": q.integer_field()}))
    client.bulk(name, [{"n": i} for i in range(5)])
    repo = f"conf-repo-{time.time_ns() % 1_000_000_000}"
    location = tempfile.mkdtemp(prefix="gnarl-repo-")
    registered = client.snapshots.register_repository(
        repo, {"type": "fs", "location": location}
    )
    try:
        assert registered.repository == repo
        assert repo in [r.repository for r in client.snapshots.list_repositories()]

        job = client.snapshots.create(repo, "s1", index=name)
        done = client.snapshots.wait(job, timeout=120, interval=0.2)
        assert done.state.value == "succeeded"
        assert client.snapshots.list(repo) == ["s1"]
        descriptor = client.snapshots.get(repo, "s1")
        assert descriptor.index_name == name
        assert descriptor.signature_verified is True, descriptor.signature_note
        assert any(j.id == job.id for j in client.snapshots.jobs())

        assert client.snapshots.get_schedule(repo) is None
        sched = client.snapshots.set_schedule(repo, name, every_hours=24)
        assert (sched.target, sched.everyHours) == (name, 24)
        assert client.snapshots.get_schedule(repo).nextSnapshotName
        assert client.snapshots.clear_schedule(repo) is True

        assert client.snapshots.delete(repo, "s1").deleted is True
    finally:
        client.snapshots.unregister_repository(repo)
    with pytest.raises(NotFound):
        client.snapshots.get_repository(repo)


# ─── Async parity ───────────────────────────────────────────────────────────


async def test_the_async_groups_work_against_a_real_node(node: str, ns, index):
    name = ns()
    async with AsyncClient(node, timeout=60.0, verify=_verify_for(node)) as c:
        ent = await c.entitlement()
        assert ent.state in {"active", "refused", "unenforced", "none"}
        await c.namespaces.index_document(name, {"subject": "async"}, wait_for="visible")
        res = await c.namespaces.search(name, q.match("subject", "async"))
        assert len(res) == 1
        # A namespace of its own: sorting `name` by a field its first document
        # lacked trips the server defect pinned below.
        sorted_ns = ns("sorted")
        await c.namespaces.bulk_chunked(
            sorted_ns, [BulkDoc(f"a{i}", {"n": i}) for i in range(5)], chunk_size=2,
            wait_for="visible",
        )
        pages = c.namespaces.iter_search(sorted_ns, sort=["n"], page_size=2)
        assert [h.field_id async for h in pages] == [f"a{i}" for i in range(5)]

        space = f"conf-amem-{time.time_ns() % 1_000_000_000}"
        for i in range(3):
            await c.memory.remember(f"async memory number {i}", namespace=space)

        async def recalled() -> int:
            return len((await c.memory.recall("async memory", namespace=space, k=2)).memories)

        deadline = time.monotonic() + 30
        while await recalled() < 2 and time.monotonic() < deadline:
            await asyncio.sleep(0.2)
        assert await recalled() == 2, "k=2 must bound the recall to two memories"

        indexed = index(q.schema({"n": q.integer_field()}))
        assert (await c.get_policy(indexed)).placement.value == "mesh"
