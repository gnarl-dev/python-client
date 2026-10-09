# python-client

The Python client for [Gnarl](https://gnarl.dev) — a decentralized search fabric.

A node is a peer, not a coordinator, so there is no cluster endpoint to point
at. You talk to a node and it answers for the mesh. Any node will do.

```bash
pip install gnarl-client
```

The package is **`gnarl-client`** and the import is **`gnarl`**. The bare
name `gnarl` on PyPI belongs to an unrelated project — installing it gets you
somebody else's code.

## Quick start

```python
from gnarl import Client, query as q

with Client("https://localhost:8080", verify=False) as c:
    c.create_index("places", q.schema({
        "name":     q.keyword_field(),
        "location": q.geo_point_field(),
    }))

    c.index_document("places", {
        "name":     "sydney",
        "location": q.geo_point(-33.8688, 151.2093),
    }, id="sydney")

    res = c.search("places", q.geo_distance("location", -33.8688, 151.2093, 1_000_000))
    for hit in res.hits:
        print(hit.field_id)
```

There is an `AsyncClient` with the same surface:

```python
import asyncio
from gnarl import AsyncClient, query as q

async def main():
    async with AsyncClient("https://localhost:8080", verify=False) as c:
        res = await c.search("places", q.match_all())
        print(len(res))

asyncio.run(main())
```

## Connecting

**A node serves TLS by default.** `lucenia start` listens on **8080** over
**https**, using a self-signed certificate it generates on first run. `--no-tls`
turns that off, and the desktop build uses it, but a node you started with the
plain command speaks https — which is why every example above says so.

That certificate cannot be verified, because nothing signed it. `verify=False`
is the right answer for a node **you started yourself** and the wrong answer
for anything else: it turns off the protection TLS exists to provide, and a
client that keeps it on by habit will happily talk to whoever answers the
address. Against a node with a real certificate, pass nothing:

<!-- doctest: skip because it needs a deployment with a real certificate -->
```python
from gnarl import Client

c = Client("https://search.example.com")          # verified, the normal case
c = Client("https://search.example.com", verify="/etc/ssl/internal-ca.pem")
```

An address with no scheme becomes **https**, so `Client("search.example.com")`
is never silently downgraded to plaintext.

## Errors

Every failure is a `GnarlError`. Catch the subclass you care about:

```python
from gnarl import Client, AlreadyExists, ValidationError, query as q

c = Client("https://localhost:8080", verify=False)
try:
    c.create_index("places", q.schema({"name": q.keyword_field()}))
except AlreadyExists:
    pass                       # fine, it was already there
except ValidationError as e:
    print("bad schema:", e.reason)
```

Each one carries `type`, `reason`, `status`, an optional `detail`, and
`retry_after` on a 429:

```python
import time
from gnarl import Client, RateLimited, query as q

c = Client("https://localhost:8080", verify=False)
try:
    c.search("places", q.match_all())
except RateLimited as e:
    time.sleep(e.retry_after_or(2.0))
```

| class | raised for |
|---|---|
| `NotFound` | a missing index, document, field, repository or snapshot — and `route_not_found`, a path this node has no route for |
| `Conflict` | a 409: `job_in_progress`, `namespace_not_snapshottable` (mid-promotion), `unverified_signer` |
| `AlreadyExists` | `index_already_exists`; a `Conflict` |
| `ValidationError` | `validation_error`, `schema_error`, `shared_pool`, and an untyped 400 or 422 |
| `Unauthenticated` / `Forbidden` | 401 / 403 — including `unauthorized`, which is a wrong BYOK key, not a login problem |
| `RateLimited` / `Unavailable` | 429 / 503, with `retry_after` when the node gave a hint |
| `Unsupported` | `unsupported_capability`, `unsupported_engine` |
| `InternalError` | `internal_error`, `repository_error` |

An error type this client does not recognise still raises a `GnarlError` with
`type` set, never something more familiar — a caller branching on a guess takes
the path meant for a different failure.

## Completeness

A search spans claims held by many peers, and a node answers with whatever it
could reach. For interactive search that is the right default. For anything
auditable — a compliance export, a reconciliation job — a quietly partial
answer is a wrong answer:

```python
from gnarl import Client, IncompleteResult, query as q

c = Client("https://localhost:8080", verify=False)
try:
    res = c.search("places", q.match_all(), require_complete=True)
    print(f"complete: {len(res)} hits from {res.coverage.served_claims} claims")
except IncompleteResult as e:
    # The partial result is still here, so you can degrade to it deliberately
    # rather than lose the work.
    cov = e.response.coverage
    print(f"{cov.served_claims} of {cov.expected_claims} claims answered")
```

Every response carries `coverage` whether you ask for completeness or not.

## Tamper evidence

`verify=True` requires every served claim to be **proven** against an anchor
the node holds independently of whoever served it:

```python
from gnarl import Client, query as q

c = Client("https://localhost:8080", verify=False)
res = c.search("places", q.match_all(), verify=True)
if not res.partial:
    print("every row returned was proven")
```

A claim nobody could check counts against completeness exactly as an
unanswered one does. "Cannot check" is never reported as "checked" — those are
different states and conflating them either invents trust or destroys data.

This is distinct from `require_complete`: that asks whether every claim
**answered**, `verify` asks whether every answer was **proven**. Set both to
demand both.

## Seeing where a query went

`profile=True` returns the fan-out the node would otherwise discard — which
peer served each claim, which were skipped and why, and what each claim's proof
came to:

```python
from gnarl import Client, query as q

c = Client("https://localhost:8080", verify=False)
res = c.search("places", q.match_all(), profile=True)
fan_out = res.profile.fan_out
print(f"{fan_out.nodes_responded}/{fan_out.nodes_contacted} nodes answered")
for claim in fan_out.claims:
    print(claim.claim_id, claim.source, claim.verification.value, claim.reason)
```

The node computes none of this unless you ask. `profile.graph` is present only
for a graph traversal, so the usual response carries `fan_out` alone.

## Two things that surprise people

**`geo_distance` is flat, and the radius is metres.** Not the Elasticsearch
shape — there is no nested `location` object and no `"10km"` string. The
builder makes the right one:

```python
from gnarl import query as q

query = q.geo_distance("location", -33.8688, 151.2093, 10_000)  # ten kilometres
```

Coordinates stay Python floats, which are IEEE-754 doubles, all the way to the
wire. Passing one through float32 anywhere displaces a survey-grade coordinate
by about 23 cm.

**Four field names are reserved**: `id`, `version`, `title` and
`canonical_url`. They belong to the document envelope, so declaring one is
rejected at index creation. `title` is the one people hit — use `name`,
`headline` or `subject`.

## Bulk writes

A bulk request can return 200 with individual items failed, which is the most
common way to lose writes silently. `failed_items` makes the check a one-liner:

```python
from gnarl import Client, failed_items

c = Client("https://localhost:8080", verify=False)
result = c.bulk("places", [
    {"name": "sydney"},
    {"name": "melbourne"},
])
for item in failed_items(result):
    print("failed:", item.field_id, item.error.reason)
```

## Namespaces

Many lightweight tenants over shared pools. A namespace exists from its first
write, and every read is fenced to it by a filter the node applies and the
caller cannot override:

```python
from gnarl import Client, query as q

with Client("https://localhost:8080", verify=False) as c:
    c.namespaces.index_document(
        "tenant-a", {"subject": "invoice overdue"}, id="t1", wait_for="visible"
    )
    res = c.namespaces.search("tenant-a", q.match("subject", "invoice"))
    print([h.field_id for h in res])
```

`wait_for="visible"` returns once the write is searchable, so the search right
after it sees it. Without it a write is acknowledged when it is DURABLE, which
comes first.

`c.namespaces` also has `bulk`, `get_document`, `delete_document`, `list`,
`put_mapping` (declare a `dense_vector` field, which gives the namespace a
dedicated index), `promote`, and the bring-your-own-key trio `set_key`,
`key_status` and `revoke_key`. `list()` follows every page and reports
`partial` when a peer did not answer: the catalog is not replicated, so an
unreachable peer means a namespace may be missing.

## Agent memory

```python
from gnarl import Client

with Client("https://localhost:8080", verify=False) as c:
    c.memory.remember("the boat is moored at pier 4", namespace="deckhand")
    found = c.memory.recall("where is the boat", namespace="deckhand", k=3)
    for mem in found.memories:
        print(f"{mem.score:.2f}  {mem.content}")
```

Embedding happens on the node. `answer()` composes over what `recall` finds,
and `ingest_document`, `ingest_messages` and `ingest_voice` take files, chat
transcripts and voice notes. `ingest_document` requires `space`: `personal`
stays on the device and `household` is replicated to the mesh's peers, so there
is deliberately no default.

## Subscription

```python
from gnarl import Client

with Client("https://localhost:8080", verify=False) as c:
    ent = c.entitlement()
    print(ent.state)            # active, refused, unenforced or none
    if ent.expires_at:
        print("renews by", ent.expires_at.date())
```

`refused` is not `none`: a key is present and was rejected — expired, or signed
by a key this build does not trust — and the person holding it has paid.
`not_after` on the wire is epoch **seconds**; `expires_at` reads it as such.
Activate a key from the account page with `c.activate_entitlement(key)`; a
refusal raises `ValidationError` whose `reason` says which failure it was.

## Backup and restore

<!-- doctest: skip because it writes a repository to a directory on the node's host -->
```python
from gnarl import Client

with Client("https://localhost:8080", verify=False) as c:
    c.snapshots.register_repository("local", {"type": "fs", "location": "/var/backups/gnarl"})
    job = c.snapshots.create("local", "places-1", index="places")
    done = c.snapshots.wait(job)              # raises JobFailed if it failed
    print(done.result)

    c.snapshots.set_schedule("local", "places", every_hours=24)
```

Snapshot, restore and cleanup run in the background, one at a time per node;
each returns a job and `wait()` polls it to the end. A second job while one is
running raises `Conflict`. A restore refuses a snapshot this node cannot
attribute to itself or a known peer until you pass the signer's key or consent
with `allow_unverified_signer=True`, and refuses to roll a live index back
without `allow_overwrite_live_index=True`.

## Index maintenance

`c.forcemerge(index)` merges segments (expensive; for a quiet period), and
`c.get_policy(index)` / `c.put_policy(index, placement="local")` read and narrow
how far an index's data may travel. Only the node that created an index may
change its policy; any other answers `Forbidden`.

## Authentication

A node with RBAC enabled exempts loopback callers, so a local node usually
needs no token. A remote one always does:

```python
import os
from gnarl import Client

c = Client("https://node.example.com", token=os.environ["GNARL_TOKEN"])
```

See [Connecting](#connecting) for what the node's certificate means for the
`verify` argument.

## How this package is built

`gnarl/_models.py` is **generated** from the node's OpenAPI description and is
never edited by hand, so the payload types cannot drift from the server.
Everything else is written by hand, so it can be idiomatic. Regenerate with:

```bash
make models          # = scripts/regen-models.sh
make vendor          # copy ../lucenia/rust/api/openapi.yaml in, then regenerate
```

The script is the only regeneration command: CI's spec-drift job runs it and
compares the result byte for byte. The generator **and the formatters it runs**
are pinned exactly in `requirements-codegen.txt` and installed into their own
`.venv-codegen`, because an unpinned black reformats the file on a runner and
fails a pull request that changed nothing. To move to a newer generator, bump
the pins, regenerate, and commit both together.

Several flags are not cosmetic. Without `--deserialize-default-values enum`, an
enum-typed field with a default holds the raw string and pydantic emits a
serialization warning on every request — which under `filterwarnings =
["error"]` is how the test suite found it. Without `--use-default-kwarg`,
`Field(None, ...)` passes the default positionally, mypy cannot see it through
`dataclass_transform`, and every optional field reads as required: 177 spurious
strict-mode errors. `--openapi-scopes schemas paths` gives the inline request
and response shapes (entitlement, memory, namespaces, schedules) models of their
own, and `--disable-timestamp` makes the output a pure function of the input.

The description is vendored at `src/gnarl/openapi.yaml` and ships in the wheel:
a user debugging a response should be able to read the contract out of the
installed package.

## Tests

Four layers:

| layer | where | what it proves |
|---|---|---|
| unit | `tests/test_query.py`, `tests/test_errors.py` | builders emit the exact wire shape; every error body parses |
| regression | `tests/test_regressions.py` | defects that shipped once stay fixed |
| integration | `tests/test_client.py` | the full request/response path over a mocked transport |
| smoke + conformance | `tests/conformance/` | a real node boots and answers real HTTP |

Compiling — or in Python, importing — proves the types match the description.
It does not prove the description matches the server, and that gap is where
client bugs live. So the conformance suite starts a real node and drives it:

```bash
pytest                                            # everything but conformance
LUCENIA_BIN=/path/to/lucenia pytest               # starts a node, runs it all
GNARL_TEST_NODE=https://localhost:8080 pytest     # uses a node you have
```

The README's examples are executed by `tests/test_readme_examples.py` against a
live node, so a snippet here that does not work is a failing test rather than a
bug report.

Writing this client found a defect in the API description before it had a
single test: `SearchProfile` was declared with `graph` required and no `fan_out`
at all, while the server sends `fan_out` for every profiled query and `graph`
only for a graph traversal. A strictly generated model rejected the ordinary
response outright. That is what generating from the description, and then
running it against a node, is for.

## License

Apache-2.0. The node itself is AGPL-3.0-or-later; the client is permissive so
it can be embedded freely.
