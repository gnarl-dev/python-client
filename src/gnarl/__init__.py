"""gnarl — the Python client for a Gnarl node.

A node is a peer in a decentralized search fabric rather than a coordinator,
so there is no cluster endpoint to point at: you talk to a node, and it answers
for the mesh. Any node will do.

    from gnarl import Client, query as q

    with Client("http://localhost:8080") as c:
        c.create_index("places", q.schema({
            "name":     q.text_field(),
            "location": q.geo_point_field(),
        }))
        c.index_document("places", {
            "name": "Sydney Opera House",
            "location": q.geo_point(-33.8568, 151.2153),
        })
        res = c.search("places", q.geo_distance("location", -33.8688, 151.2093, 5_000))
        for hit in res.hits:
            print(hit.field_source["name"])

The payload types in ``gnarl._models`` are generated from the node's OpenAPI
description and cannot drift from it. Everything else is written by hand, so
it can be idiomatic.

Two things worth knowing before you read further, because each has a default
that is right for interactive search and wrong for an audit:

``search(..., require_complete=True)`` turns a partial answer into an error. A
search spans claims held by many peers and a node answers with whatever it
reached; a compliance export must not quietly read a subset.

``search(..., verify=True)`` demands tamper evidence. A claim nobody could
check counts against completeness exactly as an unanswered one does — "cannot
check" is never reported as "checked".
"""

from __future__ import annotations

from . import query
from ._models import (
    BulkIndexResponse,
    BulkItemResult,
    ClaimRoute,
    FanOutProfile,
    FieldDefinition,
    FieldType,
    Hit,
    IndexMetadata,
    IndexSchema,
    Query,
    QueryScope,
    SearchCoverage,
    SearchProfile,
    SkippedClaim,
    TotalHits,
    Verification,
)
from .client import (
    DEFAULT_TIMEOUT,
    AsyncClient,
    BulkDoc,
    Client,
    NodeStatus,
    NodeVersion,
    SearchResult,
    failed_items,
)
from .errors import (
    AlreadyExists,
    Forbidden,
    GnarlError,
    IncompleteResult,
    InternalError,
    NotFound,
    RateLimited,
    Unauthenticated,
    Unsupported,
    ValidationError,
)

__version__ = "0.1.0"

__all__ = [
    # client
    "Client",
    "AsyncClient",
    "SearchResult",
    "BulkDoc",
    "NodeStatus",
    "NodeVersion",
    "DEFAULT_TIMEOUT",
    "failed_items",
    # queries and schemas
    "query",
    "Query",
    "QueryScope",
    "IndexSchema",
    "FieldDefinition",
    "FieldType",
    # responses
    "Hit",
    "TotalHits",
    "SearchCoverage",
    "SkippedClaim",
    "SearchProfile",
    "FanOutProfile",
    "ClaimRoute",
    "Verification",
    "IndexMetadata",
    "BulkIndexResponse",
    "BulkItemResult",
    # errors
    "GnarlError",
    "NotFound",
    "AlreadyExists",
    "ValidationError",
    "Unauthenticated",
    "Forbidden",
    "RateLimited",
    "Unsupported",
    "InternalError",
    "IncompleteResult",
    "__version__",
]
