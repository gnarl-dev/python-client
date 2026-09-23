"""Conformance against a REAL node.

Importing this package proves the types line up with the description. It does
not prove the description matches the server, and that gap is where client bugs
live: a field renamed on the wire, an error shape that differs by route, a
query the node rejects for a reason the description never mentions. Every test
here drives real HTTP against a real node and asserts on what comes back.

Running it::

    pytest                                        # conformance skips
    LUCENIA_BIN=/path/to/lucenia pytest           # starts a node, runs it all
    GNARL_TEST_NODE=http://localhost:8080 pytest  # uses a node you have

The harness finds a node in this order, and says exactly what to do if it
cannot:

1. ``$GNARL_TEST_NODE`` — a node you already have running.
2. ``$LUCENIA_BIN`` — a ``lucenia`` binary the harness starts and stops itself.
3. a ``lucenia`` binary in the sibling lucenia checkout's target directory.

A node runs without a JVM on the native engine, so no Java toolchain is needed.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import signal
import socket
import subprocess
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from gnarl import Client, GnarlError

#: How long a node gets to come up. Generous because a cold CI runner is slow
#: and the harness POLLS — it does not sleep this long.
BOOT_TIMEOUT = 60.0


def _candidate_binaries() -> list[Path]:
    explicit = os.environ.get("LUCENIA_BIN")
    if explicit:
        return [Path(explicit)]
    here = Path(__file__).resolve().parents[2]
    return [
        here.parent / "lucenia/rust/target/release/lucenia",
        here.parent / "lucenia/rust/target/debug/lucenia",
    ]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_ready(addr: str, within: float) -> str | None:
    """Poll until the node answers, rather than sleeping a guessed interval.

    A fixed sleep is a bet on how fast the machine is: it passes on a laptop
    and fails on a loaded runner, and the failure then looks like a product
    defect rather than a slow start.
    """
    deadline = time.monotonic() + within
    last = "never tried"
    with Client(addr, timeout=2.0) as probe:
        while time.monotonic() < deadline:
            try:
                probe.ping()
                return None
            except GnarlError as exc:
                last = str(exc)
            time.sleep(0.2)
    return f"after {within:.0f}s: {last}"


@pytest.fixture(scope="session")
def node() -> Iterator[str]:
    """The address of a live node, or a skip explaining how to get one."""
    existing = os.environ.get("GNARL_TEST_NODE")
    if existing:
        problem = _wait_ready(existing, 10.0)
        if problem:
            pytest.fail(f"$GNARL_TEST_NODE={existing} did not answer: {problem}")
        yield existing
        return

    binary = next((p for p in _candidate_binaries() if p.is_file()), None)
    if binary is None:
        pytest.skip(
            "no node available. Either point $GNARL_TEST_NODE at a running "
            "node, or set $LUCENIA_BIN to a `lucenia` binary "
            "(cargo build -p luceniad --bin lucenia)."
        )

    port = _free_port()
    data_dir = tempfile.mkdtemp(prefix="gnarl-conformance-")
    # The node's log goes to a FILE, not a pipe. A pipe nobody reads fills its
    # buffer and blocks the node mid-test, and leaving it open past the run
    # raises a ResourceWarning that this suite treats as an error. The file is
    # read only when something goes wrong, which is when it is worth having.
    log = open(Path(data_dir) / "node.log", "w+b")  # noqa: SIM115
    proc = subprocess.Popen(
        [
            str(binary), "start",
            "--port", str(port),
            "--data-dir", data_dir,
            # Keeps this node off any real mesh: a conformance run must not
            # discover a developer's cluster, join it, and then assert on data
            # it does not own.
            "--single-node",
            # No certificate handling in the harness.
            "--no-tls",
            # `lucenia start` enables the production rate limiter: 600
            # requests/minute per client IP with a burst of 60. That is an
            # abuse brake for an open mesh, and a conformance suite is not
            # abuse — it is one client issuing a few hundred requests in a few
            # seconds, which trips the burst and turns every later test into a
            # 429 that reads as a product defect. The limiter has its own tests
            # in the node; this suite is measuring the API.
            "--no-http-rate-limit",
            "--headless",
        ],
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    addr = f"http://127.0.0.1:{port}"
    try:
        problem = _wait_ready(addr, BOOT_TIMEOUT)
        if problem:
            proc.send_signal(signal.SIGTERM)
            log.flush()
            log.seek(0)
            tail = log.read().decode("utf-8", "replace")[-2000:]
            pytest.fail(f"node at {addr} never became ready: {problem}\n{tail}")
        yield addr
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        log.close()
        shutil.rmtree(data_dir, ignore_errors=True)


@pytest.fixture
def client(node: str) -> Iterator[Client]:
    with Client(node, timeout=60.0) as c:
        yield c


@pytest.fixture
def index(request, client: Client):
    """An index unique to this test, removed afterwards.

    Unique per test because the session shares one node: a shared index name
    turns an unrelated test's cleanup into this test's missing data. Hashed
    rather than spelled out because the server caps a name at 64 characters and
    a pytest node id overruns that easily.
    """

    def make(schema) -> str:
        digest = hashlib.sha256(request.node.nodeid.encode()).hexdigest()[:8]
        name = f"conf-{digest}-{time.time_ns() % 1_000_000_000}"
        client.create_index(name, schema)
        request.addfinalizer(lambda: _drop(client, name))
        return name

    return make


def _drop(client: Client, name: str) -> None:
    try:
        client.delete_index(name)
    except GnarlError:
        # Cleanup failing must not mask the test's own verdict.
        pass


def until(predicate, within: float = 30.0, every: float = 0.1) -> bool:
    """Poll ``predicate`` until it is true, or give up.

    Indexing is asynchronous: a document is acknowledged before it is
    searchable. Waiting for a specific condition is the honest way to express
    that — a fixed sleep either wastes time or fails on a slow machine, and the
    failure reads as a product defect.

    Note this proves PRESENCE only. Proving something is absent means spending
    the whole window, because an early return just means it has not arrived
    yet; the tests that need that say so.
    """
    deadline = time.monotonic() + within
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(every)
    return False
