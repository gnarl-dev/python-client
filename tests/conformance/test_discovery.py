"""``Client()`` with no address reaches the node on this machine.

The node records where it answers in ``<data_dir>/runtime/endpoint.json`` when
it starts. Released nodes may predate that, so this writes the file the way the
node does — pointing at the live test node — and asserts that a client given no
address talks to it, end to end.
"""

from __future__ import annotations

import json

import httpx
import pytest

from gnarl import Client


@pytest.mark.conformance
def test_a_client_with_no_address_reaches_the_recorded_node(node, monkeypatch, tmp_path):
    url = httpx.URL(node)
    data_dir = tmp_path / "data"
    (data_dir / "runtime").mkdir(parents=True)
    (data_dir / "runtime" / "endpoint.json").write_text(
        json.dumps({"scheme": url.scheme, "host": url.host, "port": url.port, "pid": 1})
    )
    monkeypatch.delenv("GNARL_URL", raising=False)
    monkeypatch.setenv("LUCENIA_DATA_DIR", str(data_dir))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    # The harness's own node serves a self-signed certificate; discovery keeps
    # verification on, so a TLS node needs the same opt-out a reader would use.
    loopback = url.host in ("127.0.0.1", "::1", "localhost")
    with Client(verify=not loopback) as c:
        assert c._base == str(url).rstrip("/")
        c.ping()


@pytest.mark.conformance
def test_recall_in_a_namespace_nobody_has_written_to(client):
    res = client.memory.recall("what do I remember?", namespace="conf-never-written-ns")
    assert res.count == 0
    assert res.memories == []
