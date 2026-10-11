# Changelog

## 0.2.1 — 2026-10-11

A patch release for the first five minutes.

### Changed

- **`Client()` finds the Gnarly app on this machine.** With no address, the
  client tries `$GNARL_URL`, then the endpoint the local node recorded in
  `$LUCENIA_DATA_DIR/runtime/endpoint.json` or
  `~/.lucenia/runtime/endpoint.json`, then `http://127.0.0.1:43300`. It used to
  raise. A recorded plain-`http` endpoint is used only on loopback, and a
  configured `https` address is never downgraded.
- **Port 8080 is gone from the README and docstrings.** No node defaults to
  it: the Gnarly app and `gnarl start` both listen on 43300. The examples now
  construct `Client()`.

### Fixed

- **`memory.recall` in a namespace nobody has written to** raised
  `pydantic.ValidationError`, because the node's empty answer carries no
  `embedder`. It now returns `RecallResult` with `embedder=None`.
- **The wheel ships `py.typed`.** It declared `Typing :: Typed` without the
  PEP 561 marker, so type checkers read the package as untyped.

## 0.2.0 — 2026-10-09

The GA surface: entitlement, agent memory, namespaces, placement and
maintenance, backup and restore, retries, and iterators.
