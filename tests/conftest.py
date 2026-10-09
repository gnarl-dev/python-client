"""Fixtures shared by the mocked-transport tests."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from typing import Any

import pytest

from gnarl import AsyncClient, Client

BASE = "http://node.test"


@pytest.fixture(params=["sync", "async"])
def call(request) -> Callable[[Callable[[Any], Any]], Any]:
    """Run one operation through BOTH clients.

    Every test that takes this fixture runs twice, once per client, so the
    async surface cannot quietly diverge from the sync one: a method missing,
    a path built differently, a parse skipped. Use as
    ``call(lambda c: c.memory.recall("q"))`` — the lambda's result is awaited
    when the client is async.
    """

    def run(fn: Callable[[Any], Any]) -> Any:
        if request.param == "sync":
            with Client(BASE) as c:
                return fn(c)

        async def go() -> Any:
            async with AsyncClient(BASE) as c:
                out = fn(c)
                return await out if inspect.isawaitable(out) else out

        return asyncio.run(go())

    return run
