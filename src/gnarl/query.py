"""Query builders.

The wire DSL is one object with a single top-level key set per query type.
That is faithful to the description and awkward to write by hand — and easy to
get wrong, because setting two keys type-checks and is then rejected by the
node. Each function here produces exactly one valid query.

    from gnarl import query as q

    q.bool_().filter(q.term("status", "published")).must(q.match("body", "tide")).build()
"""

from __future__ import annotations

from typing import Any

from ._models import (
    BoolQuery,
    DistanceMetric,
    Exists,
    FieldDefinition,
    FieldType,
    GeoDistanceQuery,
    IndexSchema,
    KnnQuery,
    MatchPhrase,
    MmrQuery,
    MultiMatch,
    Quantization,
    Query,
    RangeBounds,
)

__all__ = [
    "match",
    "match_phrase",
    "multi_match",
    "term",
    "prefix",
    "wildcard",
    "range_",
    "exists",
    "match_all",
    "geo_distance",
    "knn",
    "mmr",
    "bool_",
    "Bool",
    "schema",
    "text_field",
    "keyword_field",
    "integer_field",
    "long_field",
    "float_field",
    "double_field",
    "date_field",
    "boolean_field",
    "geo_point_field",
    "dense_vector_field",
    "geo_point",
]


def match(field: str, text: str) -> Query:
    """Full-text search. The text is run through the field's analyzer."""
    return Query(match={field: text})


def match_phrase(field: str, text: str, slop: int | None = None) -> Query:
    """Phrase match — the tokens must appear consecutively.

    ``slop`` allows that many intervening positions; the default of 0 is an
    exact phrase.
    """
    if slop is None:
        return Query(match_phrase={field: text})
    return Query(match_phrase={field: MatchPhrase(query=text, slop=slop)})


def multi_match(fields: list[str], text: str) -> Query:
    """Full-text search across several fields at once."""
    return Query(multi_match=MultiMatch(query=text, fields=fields))


def term(field: str, value: str) -> Query:
    """Exact, un-analyzed match.

    On an analyzed ``text`` field this usually matches nothing: the indexed
    terms are the analyzer's output, not the string you wrote. That is the most
    common surprise in any search API — use :func:`match` for text, and
    ``term`` for ``keyword``.

    The value is a string because a term lookup compares against an indexed
    term, and indexed terms are text. Use :func:`range_` for numbers and dates.
    """
    return Query(term={field: value})


def prefix(field: str, value: str) -> Query:
    """Matches terms starting with ``value``. Not analyzed."""
    return Query(prefix={field: value})


def wildcard(field: str, pattern: str) -> Query:
    """Pattern match with ``*`` (any run) and ``?`` (one character)."""
    return Query(wildcard={field: pattern})


def range_(
    field: str,
    *,
    gt: Any = None,
    gte: Any = None,
    lt: Any = None,
    lte: Any = None,
) -> Query:
    """Range query on a numeric, date or keyword field.

    Named ``range_`` because ``range`` is a builtin. Exclusive and inclusive
    bounds cannot be combined on the same side; the node rejects ``gt`` with
    ``gte``, and so does this.
    """
    if gt is not None and gte is not None:
        raise ValueError("range_: gt and gte cannot both be set")
    if lt is not None and lte is not None:
        raise ValueError("range_: lt and lte cannot both be set")
    if gt is None and gte is None and lt is None and lte is None:
        raise ValueError("range_: at least one bound is required")
    return Query(range={field: RangeBounds(gt=gt, gte=gte, lt=lt, lte=lte)})


def exists(field: str) -> Query:
    """Matches documents where the field has at least one value."""
    return Query(exists=Exists(field=field))


def match_all() -> Query:
    """Matches every document."""
    return Query(match_all={})


def geo_distance(field: str, lat: float, lon: float, radius_meters: float) -> Query:
    """Points within ``radius_meters`` of (``lat``, ``lon``) on a geo_point field.

    The radius is METRES — a number, not a unit-suffixed string like ``"5km"``
    — and the centre is flat ``lat``/``lon`` rather than a nested object. Both
    are places an Elasticsearch habit produces a request the node rejects.

    Coordinates are Python floats, which are IEEE-754 doubles, and they stay
    doubles all the way to the wire. Passing a survey-grade coordinate through
    float32 anywhere would displace it by roughly 0.23 m.
    """
    return Query(
        geo_distance=GeoDistanceQuery(
            field=field, lat=lat, lon=lon, radius_meters=radius_meters
        )
    )


def knn(
    field: str,
    vector: list[float],
    k: int,
    *,
    ef: int | None = None,
    oversample: float | None = None,
) -> Query:
    """Approximate nearest-neighbour search over a dense_vector field.

    ``ef`` is where the query sits on the recall/latency curve — the same dial
    as ``ef_search`` elsewhere. Omitted means the engine's heuristic.
    """
    return Query(knn=KnnQuery(field=field, vector=vector, k=k, ef=ef, oversample=oversample))


def mmr(
    inner: Query,
    field: str,
    *,
    lambda_: float | None = None,
    candidates: int | None = None,
) -> Query:
    """Maximal Marginal Relevance diversity rerank over ``inner``.

    ``lambda_`` is 1.0 for pure relevance and 0.0 for pure diversity.
    """
    kwargs: dict[str, Any] = {"query": inner, "field": field}
    if lambda_ is not None:
        kwargs["lambda"] = lambda_
    if candidates is not None:
        kwargs["candidates"] = candidates
    return Query(mmr=MmrQuery(**kwargs))


class Bool:
    """Accumulates clauses for a boolean query. Chainable; call ``build()``.

    ``must`` is required and scores, ``filter`` is required and does not (and
    is the cheaper choice), ``should`` is optional, ``must_not`` excludes.
    """

    def __init__(self) -> None:
        self._must: list[Query] = []
        self._filter: list[Query] = []
        self._should: list[Query] = []
        self._must_not: list[Query] = []

    def must(self, *qs: Query) -> Bool:
        """Required, scoring clauses."""
        self._must.extend(qs)
        return self

    def filter(self, *qs: Query) -> Bool:
        """Required, non-scoring clauses.

        Prefer this over :meth:`must` whenever the clause is a yes/no
        restriction rather than part of relevance.
        """
        self._filter.extend(qs)
        return self

    def should(self, *qs: Query) -> Bool:
        """Optional clauses."""
        self._should.extend(qs)
        return self

    def must_not(self, *qs: Query) -> Bool:
        """Excluding clauses."""
        self._must_not.extend(qs)
        return self

    def build(self) -> Query:
        """Finish the boolean query."""
        return Query(
            bool=BoolQuery(
                must=self._must or None,
                filter=self._filter or None,
                should=self._should or None,
                must_not=self._must_not or None,
            )
        )


def bool_() -> Bool:
    """Start a boolean query. Named with a trailing underscore; ``bool`` is a
    builtin."""
    return Bool()


# ─── Schema helpers ─────────────────────────────────────────────────────────


def schema(fields: dict[str, FieldDefinition]) -> IndexSchema:
    """Build an index schema from field definitions.

    Four field names are reserved and rejected at creation: ``id``,
    ``version``, ``title`` and ``canonical_url``. ``title`` is the one most
    schemas reach for first — use ``name``, ``headline`` or ``subject``.
    """
    return IndexSchema(fields=fields)


def text_field() -> FieldDefinition:
    """Analyzed, full-text-searchable."""
    return FieldDefinition(type=FieldType.text)


def keyword_field() -> FieldDefinition:
    """Exact-match: not analyzed, suitable for term, filtering and sorting."""
    return FieldDefinition(type=FieldType.keyword)


def integer_field() -> FieldDefinition:
    return FieldDefinition(type=FieldType.integer)


def long_field() -> FieldDefinition:
    return FieldDefinition(type=FieldType.long)


def float_field() -> FieldDefinition:
    return FieldDefinition(type=FieldType.float)


def double_field() -> FieldDefinition:
    return FieldDefinition(type=FieldType.double)


def date_field() -> FieldDefinition:
    return FieldDefinition(type=FieldType.date)


def boolean_field() -> FieldDefinition:
    return FieldDefinition(type=FieldType.boolean)


def geo_point_field() -> FieldDefinition:
    """A lat/lon point, queryable with :func:`geo_distance`."""
    return FieldDefinition(type=FieldType.geo_point)


def dense_vector_field(
    dimensions: int,
    *,
    distance_metric: str | DistanceMetric = DistanceMetric.cosine,
    quantization: str | Quantization = Quantization.none,
) -> FieldDefinition:
    """An embedding of the given dimensionality."""
    return FieldDefinition(
        type=FieldType.dense_vector,
        dimensions=dimensions,
        distance_metric=DistanceMetric(distance_metric),
        quantization=Quantization(quantization),
    )


def geo_point(lat: float, lon: float) -> dict[str, float]:
    """The wire shape of a geo_point value: flat ``lat``/``lon``, doubles."""
    return {"lat": lat, "lon": lon}
