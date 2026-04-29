"""Integration tests for :mod:`scripts.production_api`.

These tests cover every deliverable for the Week 10 REST API:

* 9 endpoints exist and are wired to their underlying NLP components.
* Pydantic validation rejects malformed payloads.
* The in-memory response cache hits on repeat requests.
* The rate limiter returns HTTP 429 beyond 10 requests / second / IP.
* The OpenAPI schema is generated and lists every endpoint.

The semantic search dependency is substituted with an in-memory fake so the
suite runs without faiss / sentence-transformers installed.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from scripts.production_api import (  # noqa: E402
    TTLCache,
    _cache,
    aggregate_feedback,
    append_feedback_event,
    apply_filters,
    apply_metadata_filters,
    app,
    candidate_listing_ids,
    feedback_log_path,
    get_bm25_searcher,
    get_metadata_store,
    get_semantic_searcher,
    listing_matches_metadata_filters,
    limiter as rate_limiter,
    metrics as metrics_registry,
    set_feedback_log_path,
)
from scripts.listing_metadata import ListingMetadataStore  # noqa: E402


@dataclass(frozen=True)
class _StubResult:
    listing_id: str
    remark: str
    score: float


class _StubSearcher:
    """Minimal stand-in for :class:`SemanticSearcher` used in tests."""

    def __init__(self) -> None:
        self.corpus = [
            _StubResult(
                "L-1",
                "Charming 3 bed home with pool and garage in Irvine.",
                0.91,
            ),
            _StubResult(
                "L-2",
                "Cozy 2 bed condo near downtown, hardwood floors.",
                0.81,
            ),
            _StubResult(
                "L-3",
                "Spacious 4 bed estate with pool, no HOA, move-in ready.",
                0.76,
            ),
        ]

    def search(  # noqa: ARG002
        self,
        query: str,
        top_k: int = 10,
        *,
        allowed_listing_ids=None,
    ):
        rows = list(self.corpus)
        if allowed_listing_ids is not None:
            allowed = {str(v) for v in allowed_listing_ids}
            rows = [row for row in rows if row.listing_id in allowed]
        return rows[:top_k]


class _StubBM25Searcher:
    """Minimal stand-in for :class:`BM25Searcher` used in tests."""

    def __init__(self) -> None:
        self.corpus = [
            _StubResult(
                "L-1",
                "Charming 3 bed home with pool and garage in Irvine.",
                4.5,
            ),
            _StubResult(
                "L-3",
                "Spacious 4 bed estate with pool, no HOA, move-in ready.",
                3.8,
            ),
            _StubResult(
                "L-2",
                "Cozy 2 bed condo near downtown, hardwood floors.",
                3.1,
            ),
        ]

    def search(  # noqa: ARG002
        self,
        query: str,
        top_k: int = 10,
        *,
        allowed_listing_ids=None,
    ):
        rows = list(self.corpus)
        if allowed_listing_ids is not None:
            allowed = {str(v) for v in allowed_listing_ids}
            rows = [row for row in rows if row.listing_id in allowed]
        return rows[:top_k]


@pytest.fixture()
def stub_metadata_store(tmp_path) -> ListingMetadataStore:  # noqa: ANN001
    csv_path = tmp_path / "listings.csv"
    csv_path.write_text(
        "L_ListingID,L_Address,L_City,beds,baths,price,sqft,remarks,remarks_clean\n"
        "L-1,42 Pine St,Irvine,3,2,725000,"
        "1850,"
        "\"Charming 3 bed home with pool and garage in Irvine.\","
        "\"Charming 3 bed home with pool and garage in Irvine.\"\n"
        "L-2,99 Oak Ave,Austin,2,1,499000,"
        "960,"
        "\"Cozy 2 bed condo near downtown.\","
        "\"Cozy 2 bed condo near downtown.\"\n"
        "L-3,7 Elm Ct,Los Angeles,4,3,1200000,"
        "2700,"
        "\"Spacious 4 bed estate with pool.\","
        "\"Spacious 4 bed estate with pool.\"\n",
        encoding="utf-8",
    )
    return ListingMetadataStore(csv_path)


@pytest.fixture()
def feedback_log(tmp_path):  # noqa: ANN001
    """Redirect the feedback log to a tmp file for the duration of a test."""
    original = feedback_log_path()
    target = tmp_path / "demo_event_log.jsonl"
    set_feedback_log_path(target)
    try:
        yield target
    finally:
        set_feedback_log_path(original)


@pytest.fixture()
def client(stub_metadata_store, feedback_log) -> TestClient:  # noqa: ANN001
    app.dependency_overrides[get_semantic_searcher] = lambda: _StubSearcher()
    app.dependency_overrides[get_bm25_searcher] = lambda: _StubBM25Searcher()
    app.dependency_overrides[get_metadata_store] = lambda: stub_metadata_store
    _cache.clear()
    metrics_registry.reset()
    rate_limiter.reset()
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()
    _cache.clear()
    metrics_registry.reset()
    rate_limiter.reset()


# ---------------------------------------------------------------------------
# Sanity / meta
# ---------------------------------------------------------------------------


def test_health_endpoint(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "version" in body
    assert "semantic_search_ready" in body


def test_openapi_lists_all_endpoints(client: TestClient) -> None:
    resp = client.get("/openapi.json")
    assert resp.status_code == 200
    paths = resp.json()["paths"]
    expected = {
        "/health",
        "/search",
        "/search/bm25",
        "/search/compare",
        "/parse-query",
        "/extract-entities",
        "/summarize",
        "/check-compliance",
        "/submit-listing",
        "/cache/stats",
        "/cache",
        "/feedback",
        "/metrics/dashboard",
        "/listings/{listing_id}",
    }
    assert expected.issubset(paths.keys())


def test_docs_endpoint_renders(client: TestClient) -> None:
    resp = client.get("/docs")
    assert resp.status_code == 200
    assert "Swagger" in resp.text or "swagger" in resp.text


# ---------------------------------------------------------------------------
# Per-endpoint behaviour
# ---------------------------------------------------------------------------


def test_parse_query_returns_filters_and_sql(client: TestClient) -> None:
    resp = client.post(
        "/parse-query",
        json={"query": "3 bedroom home in Irvine under $700k with pool"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["query"].startswith("3 bedroom")
    assert body["filters"].get("bedrooms_min") == 3
    assert body["filters"].get("price_max") == 700_000
    assert body["filters"].get("city") == "Irvine"
    assert "pool" in body["filters"].get("amenities_in", [])
    assert "%s" in body["where_sql"]
    assert isinstance(body["params"], list)


def test_extract_entities_returns_structured_fields(client: TestClient) -> None:
    text = (
        "Listed at $850,000 this 4 bed 2.5 bath home offers approximately "
        "2,450 square feet of living space with a pool and garage."
    )
    resp = client.post("/extract-entities", json={"text": text})
    assert resp.status_code == 200
    body = resp.json()
    assert body["bedrooms"] == 4
    assert body["bathrooms"] == 2.5
    assert body["price"] == 850_000
    assert 2450 in body["sqft"]
    assert isinstance(body["amenities"], list)


def test_summarize_extractive_includes_price_and_location(client: TestClient) -> None:
    resp = client.post(
        "/summarize",
        json={
            "remarks": "Bright corner unit with hardwood floors and a pool in Irvine.",
            "bedrooms": 3,
            "bathrooms": 2,
            "price": 825_000,
            "city": "Irvine",
            "num_sentences": 2,
            "mode": "extractive",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["mode"] == "extractive"
    assert "Irvine" in body["summary"]
    assert "$825,000" in body["summary"]


def test_summarize_abstractive_mode(client: TestClient) -> None:
    resp = client.post(
        "/summarize",
        json={
            "remarks": "Spacious backyard and updated kitchen.",
            "bedrooms": 2,
            "bathrooms": 1,
            "price": 500_000,
            "city": "Austin",
            "mode": "abstractive",
        },
    )
    assert resp.status_code == 200
    assert resp.json()["mode"] == "abstractive"


def test_check_compliance_flags_errors(client: TestClient) -> None:
    resp = client.post(
        "/check-compliance",
        json={"text": "Adults only, no children. Perfect Christian home."},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["compliant"] is False
    assert body["error_count"] >= 1
    severities = {v["severity"] for v in body["violations"]}
    assert "error" in severities


def test_check_compliance_clean_listing(client: TestClient) -> None:
    resp = client.post(
        "/check-compliance",
        json={"text": "3 bed 2 bath craftsman with hardwood floors and a garage."},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["compliant"] is True
    assert body["error_count"] == 0


def test_submit_listing_publishes_clean_description(client: TestClient) -> None:
    resp = client.post(
        "/submit-listing",
        json={
            "listing_id": "L-100",
            "address": "1 Pine St, Irvine, CA",
            "price_usd": 825_000,
            "description": "3 bed 2 bath home with hardwood floors and garage.",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "published"
    assert body["compliance"]["compliant"] is True


def test_submit_listing_blocks_discriminatory_description(client: TestClient) -> None:
    resp = client.post(
        "/submit-listing",
        json={
            "listing_id": "L-101",
            "address": "2 Elm St, Austin, TX",
            "price_usd": 500_000,
            "description": "Adults only, no children. No Section 8.",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "blocked"
    assert body["compliance"]["error_count"] >= 1


def test_search_endpoint_uses_stub_and_applies_filters(client: TestClient) -> None:
    resp = client.post(
        "/search",
        json={"query": "3 bedroom home with pool in Irvine", "top_k": 5},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] >= 1
    for hit in body["results"]:
        assert "pool" in hit["remark"].lower()
    assert body["filters"].get("city") == "Irvine"


def test_search_excludes_amenities(client: TestClient) -> None:
    resp = client.post(
        "/search",
        json={"query": "home without pool", "top_k": 5},
    )
    assert resp.status_code == 200
    for hit in resp.json()["results"]:
        assert "pool" not in hit["remark"].lower()


def test_search_applies_metadata_filters_before_returning_results(
    client: TestClient,
) -> None:
    resp = client.post(
        "/search",
        json={
            "query": "3 bedroom home with pool in Irvine under 800k",
            "top_k": 5,
            "mode": "hybrid",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["filters"]["city"] == "Irvine"
    assert body["filters"]["price_max"] == 800_000
    assert [hit["listing_id"] for hit in body["results"]] == ["L-1"]


def test_search_metadata_filters_can_remove_all_results(client: TestClient) -> None:
    resp = client.post(
        "/search",
        json={
            "query": "3 bedroom home with pool in Irvine under 600k",
            "top_k": 5,
            "mode": "hybrid",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["filters"]["city"] == "Irvine"
    assert body["filters"]["price_max"] == 600_000
    assert body["results"] == []
    assert body["count"] == 0


def test_search_applies_amenity_constraints_before_ranking(client: TestClient) -> None:
    resp = client.post(
        "/search",
        json={
            "query": "home with pool without hoa",
            "top_k": 5,
            "mode": "hybrid",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "pool" in body["filters"]["amenities_in"]
    assert "hoa" in body["filters"]["amenities_out"]
    assert body["results"]
    for hit in body["results"]:
        remark_lower = hit["remark"].lower()
        assert "pool" in remark_lower
        assert "hoa" not in remark_lower


def test_raw_bm25_search_skips_parser_and_filters(client: TestClient) -> None:
    resp = client.post(
        "/search/bm25",
        json={"query": "3 bedroom home with pool in Austin under 600k", "top_k": 5},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["mode"] == "bm25_raw"
    assert body["filters"] == {}
    # The filtered /search endpoint would remove these non-Austin matches.
    assert [hit["listing_id"] for hit in body["results"]] == ["L-1", "L-3", "L-2"]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_pydantic_rejects_empty_query(client: TestClient) -> None:
    resp = client.post("/parse-query", json={"query": ""})
    assert resp.status_code == 422


def test_pydantic_rejects_top_k_out_of_range(client: TestClient) -> None:
    resp = client.post("/search", json={"query": "pool", "top_k": 0})
    assert resp.status_code == 422


def test_pydantic_rejects_unknown_summary_mode(client: TestClient) -> None:
    resp = client.post(
        "/summarize",
        json={"remarks": "x", "mode": "invalid"},
    )
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


def test_parse_query_response_is_cached(client: TestClient) -> None:
    _cache.clear()
    payload = {"query": "3 bed home under $600k"}
    r1 = client.post("/parse-query", json=payload)
    r2 = client.post("/parse-query", json=payload)
    assert r1.status_code == 200 and r2.status_code == 200
    stats = client.get("/cache/stats").json()
    assert stats["hits"] >= 1
    assert stats["size"] >= 1


def test_search_response_reports_cache_flag(client: TestClient) -> None:
    _cache.clear()
    payload = {"query": "pool", "top_k": 3}
    first = client.post("/search", json=payload).json()
    second = client.post("/search", json=payload).json()
    assert first["cached"] is False
    assert second["cached"] is True
    assert first["results"] == second["results"]


def test_cache_clear_resets_state(client: TestClient) -> None:
    client.post("/parse-query", json={"query": "4 bed home"})
    assert client.get("/cache/stats").json()["size"] >= 1
    resp = client.delete("/cache")
    assert resp.status_code == 200
    assert client.get("/cache/stats").json()["size"] == 0


def test_ttl_cache_expires_entries() -> None:
    cache = TTLCache(maxsize=4, ttl_seconds=0.05)
    cache.set("k", 1)
    assert cache.get("k") == 1
    time.sleep(0.1)
    assert cache.get("k") is None


def test_ttl_cache_evicts_lru() -> None:
    cache = TTLCache(maxsize=2, ttl_seconds=60.0)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.set("c", 3)
    assert cache.get("a") is None
    assert cache.get("b") == 2
    assert cache.get("c") == 3


# ---------------------------------------------------------------------------
# Rate limiting (10 req/s per IP)
# ---------------------------------------------------------------------------


def test_rate_limit_returns_429_after_burst() -> None:
    """Fresh client so the limiter bucket is empty; burst >10 must trip 429."""
    app.dependency_overrides[get_semantic_searcher] = lambda: _StubSearcher()
    try:
        with TestClient(app) as c:
            statuses = [c.get("/health").status_code for _ in range(25)]
    finally:
        app.dependency_overrides.clear()
    assert 429 in statuses, f"expected a 429 within {len(statuses)} requests, got {statuses}"
    # First batch should still succeed.
    assert statuses[:5] == [200, 200, 200, 200, 200]


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_apply_filters_respects_includes_and_excludes() -> None:
    results = [
        _StubResult("1", "home with pool and garage", 0.9),
        _StubResult("2", "home with garage only", 0.8),
        _StubResult("3", "home with pool and updated kitchen", 0.7),
    ]
    kept = apply_filters(
        results,
        {"amenities_in": ["pool"], "amenities_out": ["garage"]},
    )
    assert [r.listing_id for r in kept] == ["3"]


def test_apply_filters_noop_without_amenities() -> None:
    results = [_StubResult("1", "anything", 0.5)]
    assert apply_filters(results, {"city": "Irvine"}) == results


def test_apply_filters_handles_dict_hits() -> None:
    results = [
        {"listing_id": "1", "remark": "home with pool", "score": 0.9},
        {"listing_id": "2", "remark": "home without amenities", "score": 0.5},
    ]
    kept = apply_filters(results, {"amenities_in": ["pool"]})
    assert [r["listing_id"] for r in kept] == ["1"]


def test_listing_matches_metadata_filters() -> None:
    record = {
        "listing_id": "L-1",
        "city": "Irvine",
        "beds": 3,
        "baths": 2,
        "price": 725_000,
        "sqft": 1850,
    }
    assert listing_matches_metadata_filters(
        record,
        {
            "city": "Irvine",
            "bedrooms_min": 3,
            "bedrooms_max": 3,
            "bathrooms_min": 2,
            "price_max": 800_000,
        },
    )
    assert not listing_matches_metadata_filters(record, {"city": "Austin"})
    assert not listing_matches_metadata_filters(record, {"price_max": 700_000})
    assert not listing_matches_metadata_filters(record, {"sqft_min": 2000})
    assert not listing_matches_metadata_filters(None, {"city": "Irvine"})


def test_apply_metadata_filters_uses_listing_metadata(
    stub_metadata_store: ListingMetadataStore,
) -> None:
    results = [
        {"listing_id": "L-1", "remark": "pool", "score": 0.9},
        {"listing_id": "L-2", "remark": "condo", "score": 0.8},
        {"listing_id": "L-3", "remark": "estate", "score": 0.7},
    ]
    kept = apply_metadata_filters(
        results,
        {"city": "Irvine", "price_max": 800_000, "bedrooms_min": 3},
        stub_metadata_store,
    )
    assert [hit["listing_id"] for hit in kept] == ["L-1"]


def test_candidate_listing_ids_applies_full_sql_like_filters(
    stub_metadata_store: ListingMetadataStore,
) -> None:
    ids = candidate_listing_ids(
        {
            "city": "Irvine",
            "price_max": 800_000,
            "bedrooms_min": 3,
            "sqft_min": 1500,
            "amenities_in": ["pool"],
            "amenities_out": ["hoa"],
        },
        stub_metadata_store,
    )
    assert ids == {"L-1"}


# ---------------------------------------------------------------------------
# New Week 11 endpoints
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["hybrid", "semantic", "keyword"])
def test_search_dispatches_each_mode(client: TestClient, mode: str) -> None:
    resp = client.post(
        "/search",
        json={"query": "3 bedroom home with pool", "top_k": 5, "mode": mode},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["mode"] == mode
    assert body["count"] >= 1
    assert body["latency_ms"] >= 0
    for hit in body["results"]:
        assert "listing_id" in hit
        assert "score" in hit


def test_search_rejects_invalid_mode(client: TestClient) -> None:
    resp = client.post(
        "/search",
        json={"query": "pool", "top_k": 3, "mode": "bogus"},
    )
    assert resp.status_code == 422


def test_search_compare_returns_three_modes(client: TestClient) -> None:
    resp = client.post(
        "/search/compare",
        json={"query": "3 bedroom home with pool in Irvine", "top_k": 3},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert set(body["modes"].keys()) == {"semantic", "keyword", "hybrid", "bm25_raw"}
    for mode_data in body["modes"].values():
        assert mode_data["available"] is True
        assert isinstance(mode_data["latency_ms"], (int, float))
    overlap = body["overlap"]
    for key in (
        "semantic_vs_keyword",
        "semantic_vs_hybrid",
        "keyword_vs_hybrid",
        "all_three",
    ):
        assert key in overlap
        assert overlap[key] >= 0


def test_search_compare_applies_metadata_filters(client: TestClient) -> None:
    resp = client.post(
        "/search/compare",
        json={"query": "3 bedroom home with pool in Irvine under 800k", "top_k": 3},
    )
    assert resp.status_code == 200
    body = resp.json()
    for mode in ("semantic", "keyword", "hybrid"):
        ids = [hit["listing_id"] for hit in body["modes"][mode]["results"]]
        assert ids == ["L-1"]
    raw_ids = [hit["listing_id"] for hit in body["modes"]["bm25_raw"]["results"]]
    assert raw_ids == ["L-1", "L-3", "L-2"]


def test_search_compare_marks_keyword_unavailable_when_bm25_missing(
    stub_metadata_store, feedback_log,  # noqa: ANN001
) -> None:
    """If BM25 isn't wired up, keyword + hybrid should report unavailable
    rather than 500."""
    app.dependency_overrides[get_semantic_searcher] = lambda: _StubSearcher()
    app.dependency_overrides[get_bm25_searcher] = lambda: None
    app.dependency_overrides[get_metadata_store] = lambda: stub_metadata_store
    _cache.clear()
    metrics_registry.reset()
    try:
        with TestClient(app) as c:
            resp = c.post(
                "/search/compare",
                json={"query": "pool", "top_k": 3},
            )
            assert resp.status_code == 200
            body = resp.json()
            assert body["modes"]["semantic"]["available"] is True
            assert body["modes"]["keyword"]["available"] is False
            assert body["modes"]["bm25_raw"]["available"] is False
            # Hybrid still works because it tolerates a missing BM25 searcher.
            assert body["modes"]["hybrid"]["available"] is True
    finally:
        app.dependency_overrides.clear()


def test_listing_detail_enriches_hit(client: TestClient) -> None:
    resp = client.get("/listings/L-1")
    assert resp.status_code == 200
    body = resp.json()
    assert body["found"] is True
    assert body["listing_id"] == "L-1"
    assert body["address"] == "42 Pine St"
    assert body["city"] == "Irvine"
    assert body["beds"] == 3
    assert body["baths"] == 2
    assert body["price"] == 725000
    assert body["sqft"] == 1850
    assert body["summary"] is not None and len(body["summary"]) > 0
    assert body["compliance_ok"] is True
    assert body["compliance_error_count"] == 0


def test_listing_detail_missing_returns_not_found_payload(
    client: TestClient,
) -> None:
    resp = client.get("/listings/does-not-exist")
    assert resp.status_code == 200
    body = resp.json()
    assert body["found"] is False
    assert body["address"] is None


def test_feedback_persists_to_jsonl(
    client: TestClient, feedback_log,  # noqa: ANN001
) -> None:
    resp = client.post(
        "/feedback",
        json={
            "listing_id": "L-1",
            "query": "3 bedroom home with pool",
            "mode": "hybrid",
            "rating": 1,
            "latency_ms": 12.3,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "recorded"
    assert body["total_events"] == 1
    assert feedback_log.exists()
    lines = feedback_log.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    import json as _json

    record = _json.loads(lines[0])
    assert record["listing_id"] == "L-1"
    assert record["rating"] == 1
    assert record["mode"] == "hybrid"


def test_feedback_rejects_invalid_rating(client: TestClient) -> None:
    resp = client.post(
        "/feedback",
        json={
            "listing_id": "L-1",
            "query": "pool",
            "mode": "hybrid",
            "rating": 5,
        },
    )
    assert resp.status_code == 422


def test_metrics_dashboard_aggregates_search_and_feedback(
    client: TestClient, feedback_log,  # noqa: ANN001
) -> None:
    client.post("/search", json={"query": "pool", "top_k": 3, "mode": "hybrid"})
    client.post("/search", json={"query": "pool", "top_k": 3, "mode": "keyword"})
    client.post(
        "/feedback",
        json={
            "listing_id": "L-1",
            "query": "pool",
            "mode": "hybrid",
            "rating": 1,
        },
    )
    client.post(
        "/feedback",
        json={
            "listing_id": "L-2",
            "query": "pool",
            "mode": "hybrid",
            "rating": -1,
        },
    )
    client.post(
        "/feedback",
        json={
            "listing_id": "L-3",
            "query": "pool",
            "mode": "hybrid",
            "rating": 1,
        },
    )

    resp = client.get("/metrics/dashboard")
    assert resp.status_code == 200
    body = resp.json()
    assert body["search_requests_total"] >= 2
    assert body["feedback_events_total"] == 3
    assert body["thumbs_up"] == 2
    assert body["thumbs_down"] == 1
    assert body["satisfaction_proxy"] == pytest.approx(2 / 3)
    assert body["latency_ms"]["count"] >= 5
    assert body["mode_distribution"]["hybrid"] >= 1
    assert body["mode_distribution"]["keyword"] >= 1
    assert "size" in body["cache"]


def test_aggregate_feedback_pure_helper(tmp_path) -> None:  # noqa: ANN001
    log = tmp_path / "events.jsonl"
    original = feedback_log_path()
    set_feedback_log_path(log)
    try:
        append_feedback_event({"rating": 1, "listing_id": "a"})
        append_feedback_event({"rating": -1, "listing_id": "b"})
        append_feedback_event({"rating": 0, "listing_id": "c"})
        agg = aggregate_feedback()
    finally:
        set_feedback_log_path(original)
    assert agg["feedback_events_total"] == 3
    assert agg["thumbs_up"] == 1
    assert agg["thumbs_down"] == 1
    assert agg["thumbs_neutral"] == 1
    assert agg["satisfaction_proxy"] == pytest.approx(0.5)
