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
    apply_filters,
    app,
    get_semantic_searcher,
)


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

    def search(self, query: str, top_k: int = 10):  # noqa: ARG002
        return list(self.corpus[:top_k])


@pytest.fixture()
def client() -> TestClient:
    app.dependency_overrides[get_semantic_searcher] = lambda: _StubSearcher()
    _cache.clear()
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()
    _cache.clear()


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
        "/parse-query",
        "/extract-entities",
        "/summarize",
        "/check-compliance",
        "/submit-listing",
        "/cache/stats",
        "/cache",
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
