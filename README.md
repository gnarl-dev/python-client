# gnarl-python

The Python client for [Gnarl](https://gnarl.dev) — a decentralized search fabric.

A node is a peer, not a coordinator, so there is no cluster endpoint to point
at. You talk to a node and it answers for the mesh. Any node will do.

```bash
pip install gnarl
```

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

## Authentication

A node with RBAC enabled exempts loopback callers, so a local node usually
needs no token. A remote one always does:

```python
import os
from gnarl import Client

c = Client("https://node.example.com", token=os.environ["GNARL_TOKEN"])
```

An address without a scheme becomes **https**, so `Client("node.example.com")`
is not silently downgraded to plaintext.

A node generates a self-signed certificate on first run. Against a node you
started yourself, `verify=False` skips certificate verification — never against
one you did not.

## How this package is built

`gnarl/_models.py` is **generated** from the node's OpenAPI description and is
never edited by hand, so the payload types cannot drift from the server.
Everything else is written by hand, so it can be idiomatic. Regenerate with:

```bash
datamodel-codegen \
  --input src/gnarl/openapi.yaml --input-file-type openapi \
  --output src/gnarl/_models.py --output-model-type pydantic_v2.BaseModel \
  --target-python-version 3.10 --use-standard-collections --use-union-operator \
  --field-constraints --use-schema-description --collapse-root-models \
  --deserialize-default-values enum --use-default-kwarg
```

The last two flags are not cosmetic. Without `--deserialize-default-values
enum`, an enum-typed field with a default holds the raw string and pydantic
emits a serialization warning on every request — which under
`filterwarnings = ["error"]` is how the test suite found it. Without
`--use-default-kwarg`, `Field(None, ...)` passes the default positionally,
mypy cannot see it through `dataclass_transform`, and every optional field
reads as required: 177 spurious strict-mode errors.

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
