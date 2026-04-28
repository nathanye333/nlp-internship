"""Unit + integration tests for the Week 11 demo client and UI helpers.

We don't try to render Streamlit (that would require a browser session); we
test that:

* :class:`DemoApiClient` correctly speaks to the FastAPI app via
  ``httpx.ASGITransport`` and surfaces typed dataclasses.
* The metadata store and listing enrichment helpers behave correctly.
* The pure UI helpers in :mod:`scripts.product_demo` (filter chips, result
  enrichment, overlap summary) produce the right shapes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.demo_api_client import (  # noqa: E402
    ApiError,
    CompareResult,
    DemoApiClient,
    SearchResult,
)
from scripts.listing_metadata import ListingMetadataStore  # noqa: E402
from scripts.product_demo import (  # noqa: E402
    aggregate_mode_distribution,
    compare_listing_ids,
    comparison_mode_label,
    enrich_results,
    filter_chips,
    format_beds_baths,
    format_price,
    overlap_summary,
    render_compliance_flag,
)
from scripts.production_api import (  # noqa: E402
    _cache,
    app,
    feedback_log_path,
    get_bm25_searcher,
    get_metadata_store,
    get_semantic_searcher,
    limiter as rate_limiter,
    metrics as metrics_registry,
    set_feedback_log_path,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


from dataclasses import dataclass


@dataclass(frozen=True)
class _Hit:
    listing_id: str
    remark: str
    score: float


class _StubSearcher:
    def __init__(self) -> None:
        self.corpus = [
            _Hit("L-1", "3 bed pool home in Irvine", 0.9),
            _Hit("L-2", "downtown condo with hardwood", 0.7),
            _Hit("L-3", "4 bed estate with pool", 0.6),
        ]

    def search(self, query: str, top_k: int = 10):  # noqa: ARG002
        return list(self.corpus[:top_k])


class _StubBM25:
    def __init__(self) -> None:
        self.corpus = [
            _Hit("L-1", "3 bed pool home in Irvine", 5.4),
            _Hit("L-3", "4 bed estate with pool", 3.2),
        ]

    def search(self, query: str, top_k: int = 10):  # noqa: ARG002
        return list(self.corpus[:top_k])


@pytest.fixture()
def stub_metadata_store(tmp_path) -> ListingMetadataStore:  # noqa: ANN001
    csv_path = tmp_path / "listings.csv"
    csv_path.write_text(
        "L_ListingID,L_Address,L_City,beds,baths,price,remarks,remarks_clean\n"
        "L-1,42 Pine St,Irvine,3,2,725000,"
        "\"3 bed pool home in Irvine.\","
        "\"3 bed pool home in Irvine.\"\n"
        "L-2,99 Oak Ave,Austin,2,1,499000,"
        "\"Downtown condo with hardwood.\","
        "\"Downtown condo with hardwood.\"\n",
        encoding="utf-8",
    )
    return ListingMetadataStore(csv_path)


@pytest.fixture()
def feedback_log(tmp_path):  # noqa: ANN001
    original = feedback_log_path()
    target = tmp_path / "events.jsonl"
    set_feedback_log_path(target)
    try:
        yield target
    finally:
        set_feedback_log_path(original)


@pytest.fixture()
def api_client(stub_metadata_store, feedback_log):  # noqa: ANN001
    """Wire DemoApiClient to the FastAPI app via FastAPI TestClient.

    ``TestClient`` is an :class:`httpx.Client` subclass, so we hand it to
    :class:`DemoApiClient` as the underlying transport for in-process tests.
    """
    app.dependency_overrides[get_semantic_searcher] = lambda: _StubSearcher()
    app.dependency_overrides[get_bm25_searcher] = lambda: _StubBM25()
    app.dependency_overrides[get_metadata_store] = lambda: stub_metadata_store
    _cache.clear()
    metrics_registry.reset()
    rate_limiter.reset()
    test_client = TestClient(app)
    api = DemoApiClient(base_url=str(test_client.base_url), client=test_client)
    try:
        yield api
    finally:
        api.close()
        test_client.close()
        app.dependency_overrides.clear()
        _cache.clear()
        metrics_registry.reset()
        rate_limiter.reset()


# ---------------------------------------------------------------------------
# DemoApiClient
# ---------------------------------------------------------------------------


def test_client_search_returns_typed_result(api_client: DemoApiClient) -> None:
    result = api_client.search("3 bedroom home with pool", top_k=3, mode="semantic")
    assert isinstance(result, SearchResult)
    assert result.mode == "semantic"
    assert result.hits and result.hits[0].listing_id == "L-1"
    assert result.client_latency_ms >= 0
    assert result.server_latency_ms >= 0


def test_client_search_modes_dispatch(api_client: DemoApiClient) -> None:
    for mode in ("hybrid", "semantic", "keyword"):
        result = api_client.search("pool", top_k=3, mode=mode)
        assert result.mode == mode
        assert isinstance(result.hits, list)


def test_client_raw_bm25_search_skips_filters(api_client: DemoApiClient) -> None:
    result = api_client.search_bm25_raw(
        "3 bedroom home with pool in Austin under 600k",
        top_k=3,
    )
    assert result.mode == "bm25_raw"
    assert result.filters == {}
    assert [hit.listing_id for hit in result.hits] == ["L-1", "L-3"]


def test_client_compare_returns_three_modes(api_client: DemoApiClient) -> None:
    result = api_client.compare("3 bed pool home", top_k=3)
    assert isinstance(result, CompareResult)
    assert set(result.modes.keys()) == {"semantic", "keyword", "hybrid", "bm25_raw"}
    assert result.overlap.get("semantic_vs_hybrid", 0) >= 0


def test_client_listing_returns_metadata(api_client: DemoApiClient) -> None:
    res = api_client.listing("L-1")
    assert res.ok()
    assert res.data["address"] == "42 Pine St"
    assert res.data["city"] == "Irvine"


def test_client_listings_bulk(api_client: DemoApiClient) -> None:
    out = api_client.listings_bulk(["L-1", "L-2", "L-missing"])
    assert "L-1" in out and "L-2" in out
    # Missing-id entries still come back as found=False payloads, not omitted.
    assert "L-missing" in out
    assert out["L-missing"].get("found") is False


def test_client_feedback_round_trip(
    api_client: DemoApiClient, feedback_log,  # noqa: ANN001
) -> None:
    res = api_client.feedback(
        listing_id="L-1",
        query="pool home",
        rating=1,
        mode="hybrid",
        latency_ms=12.0,
        note="great match",
    )
    assert res.ok()
    assert res.data["status"] == "recorded"
    metrics_res = api_client.metrics()
    assert metrics_res.ok()
    assert metrics_res.data["thumbs_up"] >= 1


def test_client_raises_api_error_on_bad_mode(api_client: DemoApiClient) -> None:
    with pytest.raises(ApiError):
        api_client.search("pool", mode="bogus")


# ---------------------------------------------------------------------------
# Metadata store
# ---------------------------------------------------------------------------


def test_metadata_store_loads_records(stub_metadata_store: ListingMetadataStore) -> None:
    record = stub_metadata_store.get("L-1")
    assert record is not None
    assert record["price"] == 725000
    assert record["beds"] == 3
    assert "L-1" in stub_metadata_store
    assert len(stub_metadata_store) >= 2


def test_metadata_store_missing_file_returns_empty(tmp_path) -> None:  # noqa: ANN001
    store = ListingMetadataStore(tmp_path / "absent.csv")
    assert store.get("anything") is None
    assert len(store) == 0


def test_metadata_store_get_many(stub_metadata_store: ListingMetadataStore) -> None:
    records = stub_metadata_store.get_many(["L-1", "L-missing"])
    assert "L-1" in records
    assert "L-missing" not in records


# ---------------------------------------------------------------------------
# UI pure helpers
# ---------------------------------------------------------------------------


def test_format_price() -> None:
    assert format_price(725000) == "$725,000"
    assert format_price(None) == "Price n/a"
    assert format_price(0) == "$0"


def test_format_beds_baths() -> None:
    assert format_beds_baths(3, 2) == "3 beds · 2 baths"
    assert format_beds_baths(1, 1) == "1 bed · 1 bath"
    assert format_beds_baths(None, None) == "Beds/baths n/a"


def test_filter_chips_renders_each_filter() -> None:
    filters = {
        "city": "Irvine",
        "bedrooms_min": 3,
        "price_max": 700_000,
        "amenities_in": ["pool"],
        "amenities_out": ["hoa"],
    }
    chips = filter_chips(filters)
    assert "city = Irvine" in chips
    assert "beds >= 3" in chips
    assert "price <= $700,000" in chips
    assert "+pool" in chips
    assert "-hoa" in chips


def test_filter_chips_empty_filters() -> None:
    assert filter_chips({}) == []


def test_enrich_results_joins_hits_with_metadata() -> None:
    hits = [
        {"listing_id": "L-1", "remark": "x", "score": 0.9},
        {"listing_id": "L-99", "remark": "y", "score": 0.5},
    ]
    details = {
        "L-1": {
            "address": "42 Pine St",
            "city": "Irvine",
            "beds": 3,
            "baths": 2,
            "price": 725000,
            "summary": "great place",
            "compliance_ok": False,
            "compliance_error_count": 1,
            "compliance_warning_count": 2,
            "found": True,
        },
    }
    rows = enrich_results(hits, details)
    assert rows[0]["address"] == "42 Pine St"
    assert rows[0]["price"] == 725000
    assert rows[0]["compliance_ok"] is False
    assert rows[0]["compliance_error_count"] == 1
    assert rows[0]["found"] is True
    assert rows[1]["address"] is None
    assert rows[1]["found"] is False


class _StubStreamlit:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def warning(self, text: str) -> None:
        self.calls.append(("warning", text))

    def info(self, text: str) -> None:
        self.calls.append(("info", text))

    def caption(self, text: str) -> None:
        self.calls.append(("caption", text))


def test_render_compliance_flag_variants() -> None:
    st = _StubStreamlit()
    render_compliance_flag(
        st,
        {
            "compliance_ok": False,
            "compliance_error_count": 1,
            "compliance_warning_count": 2,
        },
    )
    render_compliance_flag(
        st,
        {
            "compliance_ok": True,
            "compliance_error_count": 0,
            "compliance_warning_count": 1,
        },
    )
    render_compliance_flag(
        st,
        {
            "compliance_ok": True,
            "compliance_error_count": 0,
            "compliance_warning_count": 0,
        },
    )
    assert st.calls[0][0] == "warning"
    assert "blocking issue" in st.calls[0][1]
    assert st.calls[1][0] == "info"
    assert "review recommended" in st.calls[1][1]
    assert st.calls[2][0] == "caption"
    assert "no issues flagged" in st.calls[2][1]


def test_overlap_summary_text() -> None:
    overlap = {
        "semantic_vs_keyword": 1,
        "semantic_vs_hybrid": 3,
        "keyword_vs_hybrid": 2,
        "all_three": 1,
    }
    text = overlap_summary(overlap, top_k=5)
    assert "semantic ∩ keyword = 1/5" in text
    assert "all three = 1/5" in text


def test_overlap_summary_empty() -> None:
    assert overlap_summary({}, top_k=5) == ""


def test_comparison_mode_label_includes_raw_bm25() -> None:
    assert comparison_mode_label("bm25_raw") == "RAW BM25"
    assert comparison_mode_label("keyword") == "KEYWORD + FILTERS"


def test_compare_listing_ids_deduplicates_across_modes() -> None:
    result = CompareResult(
        query="pool",
        filters={},
        top_k=3,
        modes={
            "semantic": {
                "results": [
                    {"listing_id": "L-1", "remark": "a", "score": 0.9},
                    {"listing_id": "L-2", "remark": "b", "score": 0.8},
                ]
            },
            "keyword": {
                "results": [
                    {"listing_id": "L-2", "remark": "b", "score": 4.0},
                    {"listing_id": "L-3", "remark": "c", "score": 3.0},
                ]
            },
            "hybrid": {"results": [{"listing_id": "L-1", "remark": "a", "score": 1.0}]},
        },
        overlap={},
        client_latency_ms=1.2,
        raw={},
    )
    assert compare_listing_ids(result) == ["L-1", "L-2", "L-3"]


def test_aggregate_mode_distribution_sorted() -> None:
    rows = aggregate_mode_distribution({"hybrid": 5, "keyword": 1, "semantic": 3})
    assert [r["mode"] for r in rows] == ["hybrid", "semantic", "keyword"]


# ---------------------------------------------------------------------------
# Smoke: DemoApiClient against module-level FastAPI app for /metrics
# ---------------------------------------------------------------------------


def test_metrics_dashboard_increments_counts(
    api_client: DemoApiClient, feedback_log,  # noqa: ANN001
) -> None:
    api_client.search("pool", top_k=2, mode="hybrid")
    api_client.search("pool", top_k=2, mode="semantic")
    api_client.feedback(
        listing_id="L-1", query="pool", rating=1, mode="hybrid"
    )
    metrics_data = api_client.metrics().data
    assert metrics_data["search_requests_total"] >= 2
    assert metrics_data["thumbs_up"] >= 1
    assert metrics_data["mode_distribution"]["hybrid"] >= 1
