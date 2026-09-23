"""The README's examples are executed, not just read.

Documentation with untested examples rots silently: a rename lands, every test
stays green, and the first person to hit the stale snippet is a user. Every
fenced ``python`` block in README.md is compiled here, and every block that can
be run against a node is run against one.

To exempt a block, put this immediately before its fence::

    <!-- doctest: skip because <reason> -->

A reason is required. An exemption with no reason is how a suite quietly stops
covering anything.
"""

from __future__ import annotations

import re
import textwrap
from pathlib import Path

import pytest

import gnarl
from gnarl import query as q

README = Path(__file__).resolve().parents[2] / "README.md"

#: The placeholder address every runnable example uses. It is rewritten to the
#: live node's address before the block is executed, so the README can show a
#: realistic URL and the test can still run.
PLACEHOLDER = "https://localhost:8080"

FENCE = re.compile(
    r"(?:(?P<directive><!--\s*doctest:\s*(?P<action>\w+)(?P<rest>[^>]*?)-->)\s*\n)?"
    r"```python\n(?P<code>.*?)```",
    re.DOTALL,
)


class Block:
    def __init__(self, code: str, line: int, skip: str | None):
        self.code = code
        self.line = line
        self.skip = skip

    @property
    def id(self) -> str:
        return f"README.md:{self.line}"

    @property
    def runnable(self) -> bool:
        # A block that never constructs a client has nothing to drive.
        return PLACEHOLDER in self.code and self.skip is None


def _blocks() -> list[Block]:
    text = README.read_text()
    out: list[Block] = []
    for match in FENCE.finditer(text):
        skip = None
        if match.group("action") == "skip":
            reason = match.group("rest").strip()
            assert reason.startswith("because "), (
                f"README.md: a doctest skip must say why: "
                f"`<!-- doctest: skip because ... -->`, got {match.group('directive')!r}"
            )
            skip = reason
        out.append(
            Block(
                code=match.group("code"),
                line=text[: match.start("code")].count("\n") + 1,
                skip=skip,
            )
        )
    return out


BLOCKS = _blocks()


def test_the_readme_has_examples_to_check():
    """A scan that silently matches nothing proves nothing."""
    assert len(BLOCKS) >= 8, f"only found {len(BLOCKS)} python blocks in README.md"


@pytest.mark.parametrize("block", BLOCKS, ids=[b.id for b in BLOCKS])
def test_every_example_compiles(block: Block):
    """Syntax and indentation, for every block including the skipped ones.

    Compiling is cheap and catches the most common rot — a snippet edited by
    hand into something that is not valid Python.
    """
    compile(textwrap.dedent(block.code), block.id, "exec")


RUNNABLE = [b for b in BLOCKS if b.runnable]


@pytest.mark.conformance
@pytest.mark.parametrize("block", RUNNABLE, ids=[b.id for b in RUNNABLE])
def test_every_runnable_example_runs(block: Block, node: str, client):
    """Executed against a live node, with the placeholder address rewritten.

    The examples are a NARRATIVE — the quick start creates `places` and
    everything after it uses what that left behind — so they are run in README
    order and the preconditions are set up per block rather than wiped between
    them. Wiping would make every search example return nothing and pass
    anyway, which is the failure mode a doc test exists to prevent.
    """
    _prepare(client, first=block is RUNNABLE[0])
    source = block.code.replace(PLACEHOLDER, node)
    scope: dict = {"gnarl": gnarl, "q": q}
    try:
        exec(compile(textwrap.dedent(source), block.id, "exec"), scope)  # noqa: S102
    except Exception as exc:  # noqa: BLE001
        pytest.fail(
            f"{block.id} failed against a real node: "
            f"{type(exc).__name__}: {exc}\n\n{source}"
        )
    finally:
        _close_clients(scope)


def _close_clients(scope: dict) -> None:
    """Close any client an example left open.

    Several examples construct a bare `Client(...)` rather than a `with`
    block, because the point being made is about errors and a context manager
    is noise there. That is fine for a reader and untidy for a test process:
    the connection pool is closed whenever the object is collected, which lands
    in the middle of some later, unrelated test and is reported against it. Two
    of them failed that way before this existed.
    """
    for value in scope.values():
        if isinstance(value, gnarl.Client):
            value.close()


def _prepare(client, *, first: bool) -> None:
    """Put the node in the state a reader would be in at this point.

    The first runnable block is the quick start, which CREATES `places` — so
    for that one the index must be absent, or the example fails on a conflict
    that no reader would ever hit. For every later block it must exist, so that
    running one block on its own still works.
    """
    exists = client.index_exists("places")
    if first and exists:
        client.delete_index("places")
        return
    if not first and not exists:
        client.create_index(
            "places",
            q.schema({"name": q.keyword_field(), "location": q.geo_point_field()}),
        )
