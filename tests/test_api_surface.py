"""The GA surface over a mocked transport: entitlement, index maintenance,
namespaces, memory, and snapshots.

Every test takes the ``call`` fixture and so runs through BOTH clients. Each one
pins the method, the path, the body on the wire and how the response is read —
the four places a hand-written client can be wrong while still type-checking.
"""

from __future__ import annotations

import base64
import json

import httpx
import pytest
import respx

from gnarl import (
    Conflict,
    Entitlement,
    Forbidden,
    GnarlError,
    IncompleteResult,
    JobFailed,
    NamespaceList,
    NotFound,
    SnapshotJob,
    Unavailable,
    ValidationError,
    failed_items,
)
from gnarl import query as q

from .conftest import BASE


def ok(payload, status: int = 200) -> httpx.Response:
    return httpx.Response(status, json=payload)


def sent(route) -> dict:
    return json.loads(route.calls.last.request.content)


def search_payload(**over) -> dict:
    body = {
        "hits": {"total": {"value": 1, "relation": "eq"}, "hits": [{"_id": "d1"}]},
        "took": 3,
        "partial": False,
        "coverage": {"expected_claims": 1, "served_claims": 1, "skipped_claims": []},
    }
    body.update(over)
    return body


# ─── Entitlement ────────────────────────────────────────────────────────────

ACTIVE = {
    "active": True,
    "refused": None,
    "tier": "personal",
    "features": ["private-mesh", "hosted-backup"],
    "mesh_id": "m-123",
    "not_after": 1_900_000_000,
    "enforced": True,
}


@respx.mock
def test_entitlement_reads_the_status(call):
    route = respx.get(f"{BASE}/v1/node/entitlement").mock(return_value=ok(ACTIVE))
    ent = call(lambda c: c.entitlement())
    assert route.called
    assert isinstance(ent, Entitlement)
    assert ent.state == "active"
    assert (ent.tier, ent.mesh_id) == ("personal", "m-123")
    assert ent.has_feature("hosted-backup")


@respx.mock
def test_not_after_is_epoch_seconds_not_milliseconds(call):
    """Read as milliseconds, 1_900_000_000 is 22 January 1970."""
    respx.get(f"{BASE}/v1/node/entitlement").mock(return_value=ok(ACTIVE))
    ent = call(lambda c: c.entitlement())
    assert ent.expires_at is not None
    assert ent.expires_at.year == 2030
    assert ent.expires_at.utcoffset() is not None, "expiry must be timezone-aware"


@respx.mock
def test_a_refused_token_is_its_own_state_not_absence(call):
    """Collapsing refused into absent tells somebody who has paid that they
    have not, and sends them to buy a second subscription."""
    body = {
        "active": False, "refused": "expired at 1700000000", "features": ["private-mesh"],
        "enforced": True,
    }
    respx.get(f"{BASE}/v1/node/entitlement").mock(return_value=ok(body))
    ent = call(lambda c: c.entitlement())
    assert ent.state == "refused"
    assert ent.refused == "expired at 1700000000"
    # A refused token's features grant nothing.
    assert not ent.has_feature("private-mesh")


@respx.mock
def test_an_unenforced_build_says_so(call):
    respx.get(f"{BASE}/v1/node/entitlement").mock(
        return_value=ok({"active": False, "features": [], "enforced": False})
    )
    ent = call(lambda c: c.entitlement())
    assert ent.state == "unenforced"
    assert ent.expires_at is None


@respx.mock
def test_no_subscription_on_an_enforced_build_is_none(call):
    respx.get(f"{BASE}/v1/node/entitlement").mock(
        return_value=ok({"active": False, "features": [], "enforced": True})
    )
    assert call(lambda c: c.entitlement()).state == "none"


@respx.mock
def test_a_status_missing_a_required_field_fails_loudly(call):
    """`enforced` is required. A model that defaulted it to False would report
    every node as ungated."""
    respx.get(f"{BASE}/v1/node/entitlement").mock(
        return_value=ok({"active": False, "features": []})
    )
    with pytest.raises(Exception, match="enforced"):
        call(lambda c: c.entitlement())


@respx.mock
def test_backup_error_is_kept_though_the_description_omits_it(call):
    respx.get(f"{BASE}/v1/node/entitlement").mock(
        return_value=ok({**ACTIVE, "backup_error": "bucket unreachable"})
    )
    assert call(lambda c: c.entitlement()).backup_error == "bucket unreachable"


@respx.mock
def test_activate_posts_the_key_and_returns_the_stored_subscription(call):
    route = respx.post(f"{BASE}/v1/node/entitlement/activate").mock(return_value=ok(ACTIVE))
    ent = call(lambda c: c.activate_entitlement("  gnarl-ent1.abc  "))
    # Trimmed: a key pasted with its trailing newline is still the key.
    assert sent(route) == {"key": "gnarl-ent1.abc"}
    assert ent.active is True


@respx.mock
def test_a_refused_key_is_a_validation_error_carrying_the_reason(call):
    """The node refuses with a PLAIN-TEXT 400 naming the failure."""
    respx.post(f"{BASE}/v1/node/entitlement/activate").mock(
        return_value=httpx.Response(400, text="key signed by an untrusted key 'k9'")
    )
    with pytest.raises(ValidationError) as caught:
        call(lambda c: c.activate_entitlement("gnarl-ent1.abc"))
    assert "untrusted" in caught.value.reason


@respx.mock
def test_activation_without_a_data_dir_is_unavailable(call):
    respx.post(f"{BASE}/v1/node/entitlement/activate").mock(
        return_value=httpx.Response(503, text="this node has no data directory")
    )
    with pytest.raises(Unavailable):
        call(lambda c: c.activate_entitlement("gnarl-ent1.abc"))


def test_an_empty_key_is_refused_before_the_wire(call):
    with pytest.raises(ValueError):
        call(lambda c: c.activate_entitlement("   "))


# ─── Index maintenance ──────────────────────────────────────────────────────


@respx.mock
def test_forcemerge_posts_with_the_target_in_the_query(call):
    route = respx.post(f"{BASE}/v1/indexes/places/_forcemerge").mock(
        return_value=ok({"segments": 1, "partial": False})
    )
    res = call(lambda c: c.forcemerge("places", max_num_segments=2))
    req = route.calls.last.request
    assert req.url.params["max_num_segments"] == "2"
    assert not req.content, "forcemerge takes no body"
    assert (res.segments, res.partial) == (1, False)


@respx.mock
def test_forcemerge_without_a_target_sends_no_query(call):
    route = respx.post(f"{BASE}/v1/indexes/places/_forcemerge").mock(
        return_value=ok({"segments": 3, "partial": True})
    )
    res = call(lambda c: c.forcemerge("places"))
    assert route.calls.last.request.url.query == b""
    assert res.partial is True


def test_forcemerge_refuses_a_target_below_one(call):
    with pytest.raises(ValueError):
        call(lambda c: c.forcemerge("places", max_num_segments=0))


@respx.mock
def test_get_policy(call):
    respx.get(f"{BASE}/v1/indexes/places/_policy").mock(
        return_value=ok({"placement": "mesh", "replication_factor": 2})
    )
    policy = call(lambda c: c.get_policy("places"))
    assert policy.placement.value == "mesh"
    assert policy.replication_factor == 2


@respx.mock
def test_put_policy_sends_only_what_changes(call):
    """Omitted fields are left unchanged by the node, so sending a default
    would change something the caller did not mention."""
    route = respx.put(f"{BASE}/v1/indexes/places/_policy").mock(
        return_value=ok({"placement": "local"})
    )
    policy = call(lambda c: c.put_policy("places", placement="local"))
    assert sent(route) == {"placement": "local"}
    assert policy.placement.value == "local"


@respx.mock
def test_put_policy_on_a_non_origin_node_is_forbidden(call):
    respx.put(f"{BASE}/v1/indexes/places/_policy").mock(
        return_value=httpx.Response(
            403, json={"error": {"type": "forbidden", "reason": "not the origin"}}
        )
    )
    with pytest.raises(Forbidden):
        call(lambda c: c.put_policy("places", replication_factor=1))


def test_put_policy_with_nothing_to_change_is_refused(call):
    with pytest.raises(ValueError):
        call(lambda c: c.put_policy("places"))


def test_put_policy_rejects_an_unknown_placement(call):
    with pytest.raises(Exception, match="placement"):
        call(lambda c: c.put_policy("places", placement="everywhere"))


# ─── Namespaces ─────────────────────────────────────────────────────────────


@respx.mock
def test_ns_index_document_puts_the_id_beside_the_fields(call):
    route = respx.post(f"{BASE}/v1/namespaces/tenant-a/_doc").mock(
        return_value=ok({"_id": "d1", "result": "created", "ack": "visible_for_search"}, 201)
    )
    got = call(
        lambda c: c.namespaces.index_document(
            "tenant-a", {"title": "x"}, id="d1", wait_for="visible"
        )
    )
    assert got == "d1"
    assert sent(route) == {"title": "x", "_id": "d1"}
    assert route.calls.last.request.url.params["wait_for"] == "visible"


@respx.mock
def test_ns_bulk(call):
    route = respx.post(f"{BASE}/v1/namespaces/tenant-a/_bulk").mock(
        return_value=ok(
            {
                "items": [
                    {"_id": "a", "status": 201},
                    {"_id": "b", "status": 400,
                     "error": {"type": "validation_error", "reason": "bad"}},
                ],
                "errors": True,
                "ack": "accepted",
            }
        )
    )
    res = call(
        lambda c: c.namespaces.bulk(
            "tenant-a", [{"n": 1}, {"n": "x"}], wait_for="durable", wait_for_timeout_ms=500
        )
    )
    assert sent(route) == {"documents": [{"n": 1}, {"n": "x"}]}
    params = route.calls.last.request.url.params
    assert (params["wait_for"], params["wait_for_timeout_ms"]) == ("durable", "500")
    assert [i.field_id for i in failed_items(res)] == ["b"]


@respx.mock
def test_ns_search_is_scoped_by_the_path_and_reads_a_search_result(call):
    route = respx.post(f"{BASE}/v1/namespaces/tenant-a/_search").mock(
        return_value=ok(search_payload())
    )
    res = call(lambda c: c.namespaces.search("tenant-a", q.match_all(), size=5))
    assert sent(route) == {"query": {"match_all": {}}, "size": 5}
    assert [h.field_id for h in res] == ["d1"]


@respx.mock
def test_ns_search_honours_require_complete(call):
    respx.post(f"{BASE}/v1/namespaces/tenant-a/_search").mock(
        return_value=ok(search_payload(partial=True))
    )
    with pytest.raises(IncompleteResult):
        call(lambda c: c.namespaces.search("tenant-a", require_complete=True))


@respx.mock
def test_ns_get_and_delete_document(call):
    get = respx.get(f"{BASE}/v1/namespaces/tenant-a/_doc/a%2Fb").mock(
        return_value=ok({"_id": "a/b", "_source": {"n": 1}})
    )
    # The node answers 204 with no body.
    delete = respx.delete(f"{BASE}/v1/namespaces/tenant-a/_doc/a%2Fb").mock(
        return_value=httpx.Response(204)
    )
    assert call(lambda c: c.namespaces.get_document("tenant-a", "a/b")) == {"n": 1}
    assert call(lambda c: c.namespaces.delete_document("tenant-a", "a/b")) is None
    # A caller's id may hold a slash; unescaped it names a different path.
    assert get.calls.last.request.url.raw_path.endswith(b"/_doc/a%2Fb")
    assert delete.calls.last.request.url.raw_path.endswith(b"/_doc/a%2Fb")


@respx.mock
def test_ns_get_document_not_found(call):
    respx.get(f"{BASE}/v1/namespaces/tenant-a/_doc/nope").mock(
        return_value=httpx.Response(
            404, json={"error": {"type": "document_not_found", "reason": "no"}}
        )
    )
    with pytest.raises(NotFound):
        call(lambda c: c.namespaces.get_document("tenant-a", "nope"))


@respx.mock
def test_ns_put_mapping_sends_the_schema_unwrapped(call):
    """Index creation wraps the schema under `schema`; the namespace mapping
    route takes the schema itself. Wrapping it here is a 422."""
    route = respx.put(f"{BASE}/v1/namespaces/tenant-a/_mapping").mock(
        return_value=ok({"namespace": "tenant-a", "tier": "dedicated"})
    )
    schema = q.schema({"emb": q.dense_vector_field(3)})
    res = call(lambda c: c.namespaces.put_mapping("tenant-a", schema))
    body = sent(route)
    assert "schema" not in body
    assert body["fields"]["emb"]["type"] == "dense_vector"
    assert res == {"namespace": "tenant-a", "tier": "dedicated"}


@respx.mock
def test_ns_promote_returns_what_the_node_said(call):
    """The description declares a bare object; validating it into an empty
    model would discard every field."""
    route = respx.post(f"{BASE}/v1/namespaces/tenant-a/_promote").mock(
        return_value=ok({"namespace": "tenant-a", "tier": "dedicated", "copied": 3})
    )
    assert call(lambda c: c.namespaces.promote("tenant-a"))["copied"] == 3
    assert not route.calls.last.request.content


@respx.mock
def test_ns_set_key_base64_encodes_raw_bytes(call):
    route = respx.put(f"{BASE}/v1/namespaces/tenant-a/_key").mock(
        return_value=ok(
            {"namespace": "tenant-a", "encrypted": True, "unlocked": True, "registered": True}
        )
    )
    key = bytes(range(32))
    status = call(lambda c: c.namespaces.set_key("tenant-a", key))
    assert sent(route) == {"key": base64.b64encode(key).decode()}
    assert status.registered is True


@respx.mock
def test_ns_set_key_passes_base64_through(call):
    route = respx.put(f"{BASE}/v1/namespaces/tenant-a/_key").mock(
        return_value=ok({"namespace": "tenant-a", "encrypted": True, "unlocked": True})
    )
    encoded = base64.b64encode(b"k" * 32).decode()
    call(lambda c: c.namespaces.set_key("tenant-a", encoded))
    assert sent(route) == {"key": encoded}


@pytest.mark.parametrize(
    "bad", [b"short", "not base64!!", base64.b64encode(b"x" * 16).decode()]
)
def test_ns_set_key_refuses_anything_but_32_bytes(call, bad):
    with pytest.raises(ValueError):
        call(lambda c: c.namespaces.set_key("tenant-a", bad))


@respx.mock
def test_ns_wrong_key_is_forbidden(call):
    respx.put(f"{BASE}/v1/namespaces/tenant-a/_key").mock(
        return_value=httpx.Response(
            403, json={"error": {"type": "unauthorized", "reason": "wrong KEK"}}
        )
    )
    with pytest.raises(Forbidden):
        call(lambda c: c.namespaces.set_key("tenant-a", b"k" * 32))


@respx.mock
def test_ns_key_status_and_revoke(call):
    status = {"namespace": "tenant-a", "encrypted": True, "unlocked": False}
    get = respx.get(f"{BASE}/v1/namespaces/tenant-a/_key").mock(return_value=ok(status))
    rev = respx.delete(f"{BASE}/v1/namespaces/tenant-a/_key").mock(
        return_value=ok({**status, "encrypted": False})
    )
    assert call(lambda c: c.namespaces.key_status("tenant-a")).encrypted is True
    assert call(lambda c: c.namespaces.revoke_key("tenant-a")).encrypted is False
    assert get.called and rev.called


@respx.mock
def test_ns_delete(call):
    route = respx.delete(f"{BASE}/v1/namespaces/tenant-a").mock(
        return_value=ok({"namespace": "tenant-a", "deleted": 4})
    )
    assert call(lambda c: c.namespaces.delete("tenant-a")) == {
        "namespace": "tenant-a", "deleted": 4
    }
    assert route.called


@respx.mock
def test_ns_list_follows_every_page_and_remembers_any_partial_one(call):
    """A floor on ANY page makes the whole listing a floor."""
    route = respx.get(f"{BASE}/v1/namespaces").mock(
        side_effect=[
            ok({"namespaces": [{"name": "a", "promotion": "pooled", "keyed": False}],
                "next_after": "a", "partial": True}),
            ok({"namespaces": [{"name": "b", "promotion": "dedicated", "keyed": True,
                                "unlocked": True}]}),
        ]
    )
    listing = call(lambda c: c.namespaces.list())
    assert isinstance(listing, NamespaceList)
    assert listing.names() == ["a", "b"]
    assert listing.partial is True
    assert route.calls[0].request.url.query == b""
    assert route.calls[1].request.url.params["after"] == "a"


@respx.mock
def test_ns_list_page_passes_the_limit(call):
    route = respx.get(f"{BASE}/v1/namespaces").mock(return_value=ok({"namespaces": []}))
    page = call(lambda c: c.namespaces.list_page(limit=50))
    assert route.calls.last.request.url.params["limit"] == "50"
    assert page.next_after is None


def test_an_empty_namespace_name_is_refused(call):
    with pytest.raises(ValueError):
        call(lambda c: c.namespaces.get_document("", "d1"))


# ─── Memory ─────────────────────────────────────────────────────────────────


@respx.mock
def test_remember_sends_only_what_was_given(call):
    route = respx.post(f"{BASE}/v1/memory/remember").mock(
        return_value=ok({"id": "m1", "namespace": "agent", "user": "u", "embedder": "minilm"})
    )
    res = call(
        lambda c: c.memory.remember(
            "the boat is at pier 4", namespace="agent", tags={"topic": "boats"}, pinned=True
        )
    )
    assert sent(route) == {
        "content": "the boat is at pier 4",
        "namespace": "agent",
        "tags": {"topic": "boats"},
        "pinned": True,
    }
    assert (res.id, res.embedder) == ("m1", "minilm")


@respx.mock
def test_the_engine_is_called_native_whichever_node_answers(call):
    # A node released before the rename reports the native engine as
    # `tantivy`. Each response that carries the engine must say `native`.
    respx.post(f"{BASE}/v1/memory/remember").mock(
        return_value=ok(
            {
                "id": "m1",
                "namespace": "agent",
                "user": "u",
                "embedder": "minilm",
                "engine_binding": "tantivy",
            }
        )
    )
    assert call(lambda c: c.memory.remember("x")).engine_binding == "native"

    respx.get(f"{BASE}/v1/indexes").mock(
        return_value=ok(
            {
                "indexes": [
                    {
                        "name": "a",
                        "schema": {"fields": {}},
                        "claim_count": 4,
                        "engine_binding": "tantivy",
                    },
                    {
                        "name": "b",
                        "schema": {"fields": {}},
                        "claim_count": 4,
                        "engine_binding": "lucene",
                    },
                ]
            }
        )
    )
    page, _ = call(lambda c: c.list_indexes_page())
    assert [i.engine_binding for i in page] == ["native", "lucene"]


def test_remember_refuses_empty_content(call):
    with pytest.raises(ValueError):
        call(lambda c: c.memory.remember("  "))


@respx.mock
def test_recall_parses_ranked_memories(call):
    route = respx.post(f"{BASE}/v1/memory/recall").mock(
        return_value=ok(
            {
                "namespace": "agent",
                "count": 1,
                "embedder": "minilm",
                "memories": [
                    {"id": "m1", "score": 0.7, "content": "pier 4", "tags": "topic=boats",
                     "created": "2026-10-09T19:07:57.374Z"}
                ],
            }
        )
    )
    res = call(lambda c: c.memory.recall("where is the boat", namespace="agent", k=3))
    assert sent(route) == {"query": "where is the boat", "namespace": "agent", "k": 3}
    assert res.memories[0].content == "pier 4"
    # Flattened on the way out: a string, not the mapping that was stored.
    assert res.memories[0].tags == "topic=boats"


def test_recall_refuses_a_k_the_description_bounds(call):
    with pytest.raises(Exception, match="k"):
        call(lambda c: c.memory.recall("q", k=101))


@respx.mock
def test_answer_carries_the_scoping_the_node_reads(call):
    route = respx.post(f"{BASE}/v1/memory/answer").mock(
        return_value=ok({"answer": "pier 4", "count": 1, "memories": []})
    )
    res = call(lambda c: c.memory.answer("where?", space="personal", k=2))
    assert sent(route) == {"query": "where?", "space": "personal", "k": 2}
    assert res["answer"] == "pier 4"


@respx.mock
def test_bootstrap_always_sends_a_json_object(call):
    """The node reads a JSON body here and refuses a request without one."""
    route = respx.post(f"{BASE}/v1/memory/bootstrap").mock(
        return_value=ok({"active": True})
    )
    assert call(lambda c: c.memory.bootstrap())["active"] is True
    req = route.calls.last.request
    assert req.headers["Content-Type"] == "application/json"
    assert json.loads(req.content) == {}


@respx.mock
def test_ingest_document_base64s_the_bytes(call):
    route = respx.post(f"{BASE}/v1/memory/ingest/document").mock(
        return_value=ok({"id": "doc", "memories_written": 2})
    )
    res = call(lambda c: c.memory.ingest_document("notes.txt", b"\x00hello", space="personal"))
    assert sent(route) == {
        "filename": "notes.txt",
        "content_base64": base64.b64encode(b"\x00hello").decode(),
        "space": "personal",
    }
    assert res["memories_written"] == 2


def test_ingest_document_requires_a_space(call):
    """`household` replicates to every mesh peer, so there is no default."""
    with pytest.raises(ValueError, match="space"):
        call(lambda c: c.memory.ingest_document("notes.txt", b"x", space=""))


@respx.mock
def test_ingest_messages(call):
    route = respx.post(f"{BASE}/v1/memory/ingest/messages").mock(
        return_value=ok({"chunks": 1, "ids": ["m1"]})
    )
    msgs = [{"role": "me", "body": "buy milk", "ts_ms": 1}]
    res = call(lambda c: c.memory.ingest_messages(msgs, thread_title="family"))
    assert sent(route) == {"messages": msgs, "thread_title": "family"}
    assert res["ids"] == ["m1"]


def test_ingest_messages_needs_role_and_body(call):
    with pytest.raises(ValueError, match="role"):
        call(lambda c: c.memory.ingest_messages([{"text": "hi"}]))


@respx.mock
def test_ingest_voice(call):
    route = respx.post(f"{BASE}/v1/memory/ingest/voice").mock(
        return_value=ok({"chunks": 1, "ids": ["v1"]})
    )
    call(lambda c: c.memory.ingest_voice("call mum", duration_ms=1200))
    assert sent(route) == {"transcript": "call mum", "duration_ms": 1200}


# ─── Snapshots ──────────────────────────────────────────────────────────────

JOB = {
    "id": "snapshot-1", "kind": "snapshot", "state": "running",
    "repository": "r1", "snapshot": "s1", "index": "places",
}


@respx.mock
def test_register_repository_validates_and_sends_the_spec(call):
    route = respx.put(f"{BASE}/v1/repositories/r1").mock(
        return_value=ok({"repository": "r1", "spec": {"type": "fs", "location": "/b"}}, 201)
    )
    repo = call(
        lambda c: c.snapshots.register_repository("r1", {"type": "fs", "location": "/b"})
    )
    assert sent(route) == {"type": "fs", "location": "/b"}
    assert repo.repository == "r1"


def test_an_s3_spec_missing_its_bucket_fails_before_the_wire(call):
    with pytest.raises(Exception, match="bucket"):
        call(
            lambda c: c.snapshots.register_repository(
                "r1",
                {"type": "s3", "endpoint": "https://s3", "region": "auto",
                 "access_key_id": "a", "secret_access_key": "s"},
            )
        )


@respx.mock
def test_repository_reads_and_unregister(call):
    reg = {"repository": "r1", "spec": {"type": "fs", "location": "/b"}}
    respx.get(f"{BASE}/v1/repositories").mock(return_value=ok({"repositories": [reg]}))
    respx.get(f"{BASE}/v1/repositories/r1").mock(return_value=ok(reg))
    respx.delete(f"{BASE}/v1/repositories/r1").mock(
        return_value=ok({"repository": "r1", "unregistered": True})
    )
    assert [r.repository for r in call(lambda c: c.snapshots.list_repositories())] == ["r1"]
    assert call(lambda c: c.snapshots.get_repository("r1")).spec.location == "/b"
    assert call(lambda c: c.snapshots.unregister_repository("r1")).unregistered is True


@respx.mock
def test_cleanup_sends_the_grace_period(call):
    route = respx.post(f"{BASE}/v1/repositories/r1/_cleanup").mock(
        return_value=ok({**JOB, "kind": "cleanup"}, 202)
    )
    job = call(lambda c: c.snapshots.cleanup("r1", grace_seconds=60))
    assert sent(route) == {"grace_seconds": 60}
    assert job.kind.value == "cleanup"


@respx.mock
def test_set_schedule_uses_the_wire_names(call):
    route = respx.put(f"{BASE}/v1/repositories/r1/schedule").mock(
        return_value=ok(
            {"repository": "r1",
             "schedule": {"target": "places", "everyHours": 6, "enabled": True}}
        )
    )
    sched = call(lambda c: c.snapshots.set_schedule("r1", "places", 6))
    assert sent(route) == {"target": "places", "everyHours": 6}
    assert sched.everyHours == 6


def test_set_schedule_refuses_a_cadence_past_a_week(call):
    with pytest.raises(Exception, match="everyHours"):
        call(lambda c: c.snapshots.set_schedule("r1", "places", 169))


@respx.mock
def test_no_schedule_is_none_not_an_error(call):
    respx.get(f"{BASE}/v1/repositories/r1/schedule").mock(
        return_value=ok({"repository": "r1", "schedule": None})
    )
    assert call(lambda c: c.snapshots.get_schedule("r1")) is None


@respx.mock
def test_a_schedule_on_a_missing_repository_is_not_found(call):
    respx.get(f"{BASE}/v1/repositories/nope/schedule").mock(
        return_value=httpx.Response(
            404, json={"error": {"type": "repository_not_found", "reason": "no"}}
        )
    )
    with pytest.raises(NotFound):
        call(lambda c: c.snapshots.get_schedule("nope"))


@respx.mock
def test_clear_schedule_reports_whether_one_was_removed(call):
    respx.delete(f"{BASE}/v1/repositories/r1/schedule").mock(
        return_value=ok({"repository": "r1", "removed": False})
    )
    assert call(lambda c: c.snapshots.clear_schedule("r1")) is False


@respx.mock
def test_create_snapshot_of_an_index(call):
    route = respx.put(f"{BASE}/v1/repositories/r1/snapshots/s1").mock(
        return_value=ok(JOB, 202)
    )
    job = call(lambda c: c.snapshots.create("r1", "s1", index="places"))
    assert sent(route) == {"index": "places"}
    assert isinstance(job, SnapshotJob)
    assert job.state.value == "running"


@respx.mock
def test_create_snapshot_of_a_namespace_with_pool_consent(call):
    route = respx.put(f"{BASE}/v1/repositories/r1/snapshots/s1").mock(
        return_value=ok(JOB, 202)
    )
    call(lambda c: c.snapshots.create("r1", "s1", namespace="t", allow_shared_pool=True))
    assert sent(route) == {"namespace": "t", "allow_shared_pool": True}


@pytest.mark.parametrize("kwargs", [{}, {"index": "a", "namespace": "b"}])
def test_create_needs_exactly_one_target(call, kwargs):
    with pytest.raises(ValueError):
        call(lambda c: c.snapshots.create("r1", "s1", **kwargs))


@respx.mock
def test_a_second_job_is_a_conflict(call):
    respx.put(f"{BASE}/v1/repositories/r1/snapshots/s1").mock(
        return_value=httpx.Response(
            409, json={"error": {"type": "job_in_progress", "reason": "busy"}}
        )
    )
    with pytest.raises(Conflict):
        call(lambda c: c.snapshots.create("r1", "s1", index="places"))


@respx.mock
def test_list_get_and_delete_snapshots(call):
    respx.get(f"{BASE}/v1/repositories/r1/snapshots").mock(
        return_value=ok({"repository": "r1", "snapshots": ["s1", "s2"]})
    )
    respx.get(f"{BASE}/v1/repositories/r1/snapshots/s1").mock(
        return_value=ok({"snapshot": "s1", "index_name": "places", "signature_verified": True})
    )
    respx.delete(f"{BASE}/v1/repositories/r1/snapshots/s1").mock(
        return_value=ok({"repository": "r1", "snapshot": "s1", "deleted": True})
    )
    assert call(lambda c: c.snapshots.list("r1")) == ["s1", "s2"]
    assert call(lambda c: c.snapshots.get("r1", "s1")).signature_verified is True
    assert call(lambda c: c.snapshots.delete("r1", "s1")).deleted is True


@respx.mock
def test_restore_sends_only_the_consents_given(call):
    route = respx.post(f"{BASE}/v1/repositories/r1/snapshots/s1/_restore").mock(
        return_value=ok({**JOB, "kind": "restore"}, 202)
    )
    call(lambda c: c.snapshots.restore("r1", "s1"))
    assert sent(route) == {}
    call(
        lambda c: c.snapshots.restore(
            "r1", "s1", signer_public_key="ab" * 32, allow_overwrite_live_index=True
        )
    )
    assert sent(route) == {
        "signer_public_key": "ab" * 32, "allow_overwrite_live_index": True
    }


@respx.mock
def test_jobs_and_job(call):
    respx.get(f"{BASE}/v1/snapshot_jobs").mock(return_value=ok({"jobs": [JOB]}))
    respx.get(f"{BASE}/v1/snapshot_jobs/snapshot-1").mock(return_value=ok(JOB))
    assert [j.id for j in call(lambda c: c.snapshots.jobs())] == ["snapshot-1"]
    assert call(lambda c: c.snapshots.job("snapshot-1")).index == "places"


@respx.mock
def test_wait_polls_until_the_job_settles(call):
    route = respx.get(f"{BASE}/v1/snapshot_jobs/snapshot-1").mock(
        side_effect=[ok(JOB), ok(JOB), ok({**JOB, "state": "succeeded"})]
    )
    done = call(lambda c: c.snapshots.wait(SnapshotJob.model_validate(JOB), interval=0))
    assert done.state.value == "succeeded"
    assert route.call_count == 3


@respx.mock
def test_wait_raises_job_failed_with_the_job(call):
    respx.get(f"{BASE}/v1/snapshot_jobs/snapshot-1").mock(
        return_value=ok({**JOB, "state": "failed", "error": "disk full"})
    )
    with pytest.raises(JobFailed) as caught:
        call(lambda c: c.snapshots.wait("snapshot-1", interval=0))
    assert caught.value.job.error == "disk full"
    assert "disk full" in caught.value.reason


@respx.mock
def test_wait_can_return_a_failed_job_instead(call):
    respx.get(f"{BASE}/v1/snapshot_jobs/snapshot-1").mock(
        return_value=ok({**JOB, "state": "failed", "error": "disk full"})
    )
    job = call(lambda c: c.snapshots.wait("snapshot-1", interval=0, raise_on_failure=False))
    assert job.state.value == "failed"


@respx.mock
def test_wait_gives_up_after_the_timeout_without_pretending(call):
    respx.get(f"{BASE}/v1/snapshot_jobs/snapshot-1").mock(return_value=ok(JOB))
    with pytest.raises(GnarlError) as caught:
        call(lambda c: c.snapshots.wait("snapshot-1", timeout=0, interval=0))
    assert caught.value.type == "job_timeout"
    assert not isinstance(caught.value, JobFailed)


@respx.mock
def test_a_job_with_no_state_is_not_treated_as_finished(call):
    """Stopping on a missing state would report a result nobody had."""
    respx.get(f"{BASE}/v1/snapshot_jobs/snapshot-1").mock(
        side_effect=[ok({"id": "snapshot-1"}), ok({**JOB, "state": "succeeded"})]
    )
    assert call(lambda c: c.snapshots.wait("snapshot-1", interval=0)).state.value == "succeeded"
