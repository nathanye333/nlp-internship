"""Streamlit demo UI for the Real Estate NLP pipeline (Week 11).

Run locally::

    streamlit run scripts/product_demo.py

Set ``API_BASE_URL`` to point the UI at a deployed FastAPI service; defaults
to ``http://localhost:8000``.

The app is organised as three tabs:

1. **Search** -- natural-language query -> parsed filters -> mode-aware
   search results enriched with listing metadata + extractive summaries,
   with thumbs-up/down feedback per result.
2. **Compare** -- side-by-side comparison of NLP semantic search vs
   filtered keyword (BM25) search vs the hybrid fused ranker, plus a raw
   BM25 baseline that bypasses the NLP pipeline.
3. **Metrics** -- query volume, latency percentiles, satisfaction proxy
   (from feedback events), and the API's response cache statistics.

The view-rendering helpers (``render_*``) and the data-shaping helpers are
deliberately pure functions so they can be exercised from unit tests.
"""

from __future__ import annotations

import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.demo_api_client import (  # noqa: E402
    ApiError,
    CompareResult,
    DemoApiClient,
    SearchResult,
)


DEFAULT_API_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
SEMANTIC_VS_BM25_CSV = (
    _PROJECT_ROOT / "data" / "processed" / "semantic_vs_bm25.csv"
)
SAMPLE_QUERIES = [
    "3 bed 2 bath under 700k in Irvine",
    "modern home with pool and garage in Los Angeles",
    "cozy condo near downtown with hardwood floors",
    "4 bedroom waterfront property under $1M",
    "new construction with stainless steel appliances",
]


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested)
# ---------------------------------------------------------------------------


def format_price(price: Optional[float]) -> str:
    if price is None:
        return "Price n/a"
    try:
        return f"${int(price):,}"
    except (TypeError, ValueError):
        return f"${price}"


def format_beds_baths(
    beds: Optional[float],
    baths: Optional[float],
) -> str:
    parts: list[str] = []
    if beds is not None:
        parts.append(f"{beds:g} bed" + ("s" if float(beds) != 1 else ""))
    if baths is not None:
        parts.append(f"{baths:g} bath" + ("s" if float(baths) != 1 else ""))
    return " · ".join(parts) if parts else "Beds/baths n/a"


def filter_chips(filters: Mapping[str, Any]) -> list[str]:
    """Render parsed filter dict as a list of human-friendly chip strings."""
    chips: list[str] = []
    if not filters:
        return chips
    if (city := filters.get("city")) is not None:
        chips.append(f"city = {city}")
    if (bmin := filters.get("bedrooms_min")) is not None:
        chips.append(f"beds >= {bmin}")
    if (bmax := filters.get("bedrooms_max")) is not None:
        chips.append(f"beds <= {bmax}")
    if (btmin := filters.get("bathrooms_min")) is not None:
        chips.append(f"baths >= {btmin}")
    if (pmin := filters.get("price_min")) is not None:
        chips.append(f"price >= ${int(pmin):,}")
    if (pmax := filters.get("price_max")) is not None:
        chips.append(f"price <= ${int(pmax):,}")
    for amenity in filters.get("amenities_in") or []:
        chips.append(f"+{amenity}")
    for amenity in filters.get("amenities_out") or []:
        chips.append(f"-{amenity}")
    return chips


def enrich_results(
    hits: Iterable[Any],
    detail_lookup: Mapping[str, Mapping[str, Any]],
) -> list[dict]:
    """Merge search hits with listing detail records by ``listing_id``.

    ``hits`` items may be :class:`SearchHit` dataclasses or dicts; the
    output is always a list of plain dicts ready for rendering.
    """
    rows: list[dict] = []
    for hit in hits:
        if hasattr(hit, "__dict__") and not isinstance(hit, dict):
            hit_dict = asdict(hit) if hasattr(hit, "listing_id") else dict(hit.__dict__)
        elif isinstance(hit, Mapping):
            hit_dict = dict(hit)
        else:
            hit_dict = {}
        listing_id = str(hit_dict.get("listing_id", ""))
        detail = detail_lookup.get(listing_id, {}) or {}
        rows.append(
            {
                "listing_id": listing_id,
                "score": float(hit_dict.get("score", 0.0)),
                "remark": hit_dict.get("remark", ""),
                "address": detail.get("address"),
                "city": detail.get("city"),
                "beds": detail.get("beds"),
                "baths": detail.get("baths"),
                "price": detail.get("price"),
                "summary": detail.get("summary"),
                "compliance_ok": detail.get("compliance_ok"),
                "compliance_error_count": detail.get("compliance_error_count", 0),
                "compliance_warning_count": detail.get("compliance_warning_count", 0),
                "found": detail.get("found", False),
            }
        )
    return rows


def overlap_summary(overlap: Mapping[str, int], top_k: int) -> str:
    """Human-friendly description of the comparison overlap counts."""
    if not overlap:
        return ""
    parts = [
        f"semantic ∩ keyword = {overlap.get('semantic_vs_keyword', 0)}/{top_k}",
        f"semantic ∩ hybrid = {overlap.get('semantic_vs_hybrid', 0)}/{top_k}",
        f"keyword ∩ hybrid = {overlap.get('keyword_vs_hybrid', 0)}/{top_k}",
        f"all three = {overlap.get('all_three', 0)}/{top_k}",
    ]
    return " · ".join(parts)


def comparison_mode_label(mode_name: str) -> str:
    labels = {
        "semantic": "SEMANTIC",
        "keyword": "KEYWORD + FILTERS",
        "hybrid": "HYBRID",
        "bm25_raw": "RAW BM25",
    }
    return labels.get(mode_name, mode_name.upper())


def compare_listing_ids(compare_result: CompareResult) -> list[str]:
    """Return unique listing ids across all compare-mode result lists."""
    ids: list[str] = []
    seen: set[str] = set()
    for mode_data in compare_result.modes.values():
        for hit in mode_data.get("results", []):
            listing_id = str(hit.get("listing_id", ""))
            if listing_id and listing_id not in seen:
                ids.append(listing_id)
                seen.add(listing_id)
    return ids


def aggregate_mode_distribution(modes: Mapping[str, int]) -> list[dict]:
    rows = [{"mode": k, "count": int(v)} for k, v in modes.items()]
    rows.sort(key=lambda r: r["count"], reverse=True)
    return rows


def render_compliance_flag(st_module: Any, row: Mapping[str, Any]) -> None:
    """Display a soft compliance flag for demo cards without filtering results."""
    compliance_ok = row.get("compliance_ok")
    if compliance_ok is None:
        return
    errors = int(row.get("compliance_error_count", 0) or 0)
    warnings = int(row.get("compliance_warning_count", 0) or 0)
    if errors > 0:
        st_module.warning(
            f"Compliance flag: {errors} blocking issue(s), {warnings} warning(s)."
        )
    elif warnings > 0:
        st_module.info(
            f"Compliance review recommended: {warnings} warning(s), no blocking issues."
        )
    else:
        st_module.caption("Compliance check: no issues flagged")


# ---------------------------------------------------------------------------
# Streamlit views
# ---------------------------------------------------------------------------


def _client_from_state(st_module: Any) -> DemoApiClient:
    """Return (and cache) a :class:`DemoApiClient` keyed by base URL."""
    base_url = st_module.session_state.get("api_base_url", DEFAULT_API_URL)
    cached: Optional[DemoApiClient] = st_module.session_state.get("api_client")
    cached_url: Optional[str] = st_module.session_state.get("api_client_url")
    if cached is None or cached_url != base_url:
        if cached is not None:
            try:
                cached.close()
            except Exception:
                pass
        new_client = DemoApiClient(base_url=base_url)
        st_module.session_state["api_client"] = new_client
        st_module.session_state["api_client_url"] = base_url
        return new_client
    return cached


def render_search_tab(st_module: Any, client: DemoApiClient) -> None:
    st_module.subheader("Natural-language search")
    st_module.caption(
        "Type a query in plain English. The pipeline parses it into "
        "structured filters, retrieves listings, then summarises each one."
    )

    cols = st_module.columns([4, 1, 1])
    query = cols[0].text_input(
        "Search query",
        value=st_module.session_state.get(
            "last_query", "3 bed 2 bath under 700k in Irvine"
        ),
        key="search_query_input",
    )
    mode = cols[1].selectbox(
        "Mode",
        ["hybrid", "semantic", "keyword"],
        index=["hybrid", "semantic", "keyword"].index(
            st_module.session_state.get("default_mode", "hybrid")
        ),
        key="search_mode_input",
        help=(
            "hybrid = semantic + BM25 fused (recommended); "
            "semantic = sentence-transformer only; "
            "keyword = BM25 only."
        ),
    )
    top_k = cols[2].slider(
        "top_k",
        min_value=1,
        max_value=20,
        value=int(st_module.session_state.get("default_top_k", 5)),
        key="search_top_k_input",
    )

    st_module.caption("Try an example: " + " · ".join(SAMPLE_QUERIES[:3]))

    if st_module.button("Search", type="primary", key="search_button"):
        st_module.session_state["last_query"] = query
        try:
            search_result = client.search(query, top_k=top_k, mode=mode)
        except ApiError as exc:
            st_module.error(f"API error: {exc}")
            return
        details = client.listings_bulk([h.listing_id for h in search_result.hits])
        st_module.session_state["last_search_result"] = search_result
        st_module.session_state["last_search_details"] = details

    search_result: Optional[SearchResult] = st_module.session_state.get(
        "last_search_result"
    )
    details: dict = st_module.session_state.get("last_search_details", {})
    if search_result is None:
        st_module.info("Run a search to see results here.")
        return

    badge_cols = st_module.columns(4)
    badge_cols[0].metric("Mode", search_result.mode)
    badge_cols[1].metric("Hits", len(search_result.hits))
    badge_cols[2].metric(
        "Server latency",
        f"{search_result.server_latency_ms:.0f} ms",
    )
    badge_cols[3].metric(
        "Round-trip",
        f"{search_result.client_latency_ms:.0f} ms",
        delta="cached" if search_result.cached else None,
    )

    chips = filter_chips(search_result.filters)
    with st_module.expander("Parsed filters", expanded=bool(chips)):
        if chips:
            st_module.write(" · ".join(f"`{c}`" for c in chips))
        else:
            st_module.write(
                "_No structured filters extracted -- relying on text retrieval._"
            )
        st_module.json(search_result.filters, expanded=False)

    rows = enrich_results(search_result.hits, details)
    if not rows:
        st_module.warning("No listings matched -- try relaxing the query.")
        return

    for idx, row in enumerate(rows):
        with st_module.container(border=True):
            header_cols = st_module.columns([3, 1])
            header_cols[0].markdown(
                f"**{row['address'] or 'Address unavailable'}**  "
                + (f"_{row['city']}_" if row.get("city") else "")
            )
            header_cols[1].markdown(
                f"<div style='text-align:right'>"
                f"<b>{format_price(row['price'])}</b><br/>"
                f"<small>{format_beds_baths(row['beds'], row['baths'])}</small>"
                f"</div>",
                unsafe_allow_html=True,
            )
            if row.get("summary"):
                st_module.markdown(f"_{row['summary']}_")
            elif row.get("remark"):
                st_module.write(row["remark"][:280] + ("…" if len(row["remark"]) > 280 else ""))
            remark = row.get("remark") or ""
            if remark:
                with st_module.expander("Full listing text"):
                    st_module.write(remark)
            render_compliance_flag(st_module, row)
            meta_cols = st_module.columns([1, 1, 2])
            meta_cols[0].caption(f"id: `{row['listing_id']}`")
            meta_cols[1].caption(f"score: {row['score']:.3f}")
            if not row["found"]:
                meta_cols[2].caption("metadata not available")

            fb_cols = st_module.columns([1, 1, 6])
            up_key = f"thumb_up_{idx}_{row['listing_id']}"
            down_key = f"thumb_down_{idx}_{row['listing_id']}"
            if fb_cols[0].button("Helpful", key=up_key):
                _submit_feedback(st_module, client, search_result, row, rating=1)
            if fb_cols[1].button("Not helpful", key=down_key):
                _submit_feedback(st_module, client, search_result, row, rating=-1)


def _submit_feedback(
    st_module: Any,
    client: DemoApiClient,
    search_result: SearchResult,
    row: Mapping[str, Any],
    *,
    rating: int,
) -> None:
    try:
        result = client.feedback(
            listing_id=str(row.get("listing_id", "")),
            query=search_result.query,
            mode=search_result.mode,
            rating=rating,
            latency_ms=search_result.client_latency_ms,
        )
    except Exception as exc:  # pragma: no cover - UI feedback path
        st_module.warning(f"Could not record feedback: {exc}")
        return
    if result.ok():
        st_module.toast(
            ("Thanks! +1 recorded" if rating > 0 else "Thanks -- we'll improve this"),
            icon="✅",
        )
    else:
        st_module.warning(f"Feedback rejected: {result.data}")


def render_compare_tab(st_module: Any, client: DemoApiClient) -> None:
    st_module.subheader("NLP pipeline vs raw keyword search")
    st_module.caption(
        "Same query, four retrieval strategies. Semantic, filtered keyword, "
        "and hybrid run through parsed filters; raw BM25 is plain keyword "
        "matching over listing remarks with no query parsing or post-filtering."
    )
    query = st_module.text_input(
        "Comparison query",
        value=st_module.session_state.get(
            "compare_query",
            "3 bedroom home with pool in Los Angeles",
        ),
        key="compare_query_input",
    )
    cols = st_module.columns([1, 1, 4])
    top_k = cols[0].slider(
        "top_k",
        1,
        15,
        int(st_module.session_state.get("compare_top_k", 5)),
        key="compare_top_k_input",
    )
    use_offline = cols[1].toggle(
        "Offline sample",
        value=False,
        help=(
            "Read pre-computed semantic vs BM25 results from "
            "data/processed/semantic_vs_bm25.csv when the API is unavailable."
        ),
    )

    if st_module.button("Compare modes", key="compare_button", type="primary"):
        st_module.session_state["compare_query"] = query
        st_module.session_state["compare_top_k"] = top_k
        if use_offline:
            offline = _load_offline_comparison(query)
            st_module.session_state["compare_result"] = None
            st_module.session_state["compare_offline"] = offline
        else:
            try:
                compare_result = client.compare(query, top_k=top_k)
            except ApiError as exc:
                st_module.error(f"API error: {exc}")
                return
            st_module.session_state["compare_result"] = compare_result
            st_module.session_state["compare_offline"] = None

    compare_result: Optional[CompareResult] = st_module.session_state.get(
        "compare_result"
    )
    offline_rows: Optional[list[dict]] = st_module.session_state.get(
        "compare_offline"
    )

    if compare_result is None and not offline_rows:
        st_module.info("Run a comparison to see side-by-side results.")
        return

    if compare_result is not None:
        chips = filter_chips(compare_result.filters)
        if chips:
            st_module.write("Parsed filters: " + " · ".join(f"`{c}`" for c in chips))
        st_module.caption(
            "Overlap @ top-k: " + overlap_summary(compare_result.overlap, compare_result.top_k)
        )
        st_module.caption(
            "RAW BM25 is intentionally unfiltered, so it may return listings "
            "that the NLP-filtered modes remove."
        )
        details = client.listings_bulk(compare_listing_ids(compare_result))
        cols = st_module.columns(4)
        for col, mode_name in zip(cols, ("semantic", "keyword", "hybrid", "bm25_raw")):
            mode_data = compare_result.modes.get(mode_name, {})
            with col:
                st_module.markdown(f"**{comparison_mode_label(mode_name)}**")
                if not mode_data.get("available", True):
                    st_module.warning(
                        f"Unavailable: {mode_data.get('error', 'unknown error')}"
                    )
                    continue
                st_module.caption(
                    f"{mode_data.get('count', 0)} hits · "
                    f"{mode_data.get('latency_ms', 0):.0f} ms"
                )
                rows = enrich_results(
                    mode_data.get("results", [])[:compare_result.top_k],
                    details,
                )
                if not rows:
                    st_module.write("_no rows_")
                    continue
                for row in rows:
                    with st_module.container(border=True):
                        st_module.markdown(
                            f"**{row['address'] or 'Address unavailable'}**"
                        )
                        if row.get("city"):
                            st_module.caption(row["city"])
                        st_module.write(
                            f"{format_price(row['price'])} · "
                            f"{format_beds_baths(row['beds'], row['baths'])}"
                        )
                        st_module.caption(
                            f"id: `{row['listing_id']}` · score: {row['score']:.3f}"
                        )
                        if row.get("summary"):
                            st_module.markdown(f"_{row['summary']}_")
                        render_compliance_flag(st_module, row)
                        remark = row.get("remark") or ""
                        if remark:
                            with st_module.expander("Full listing text"):
                                st_module.write(remark)
    elif offline_rows:
        st_module.info(
            f"Offline mode -- showing canned data for the closest matching "
            f"query in semantic_vs_bm25.csv."
        )
        cols = st_module.columns(2)
        for col, method in zip(cols, ("semantic", "bm25")):
            with col:
                st_module.markdown(f"**{method.upper()}**")
                rows_for_method = [r for r in offline_rows if r.get("method") == method]
                if not rows_for_method:
                    st_module.write("_no rows_")
                    continue
                for row in rows_for_method[:8]:
                    with st_module.container(border=True):
                        st_module.markdown(f"**Listing `{row.get('listing_id', '?')}`**")
                        st_module.caption(f"score: {float(row.get('score', 0)):.3f}")
                        st_module.write(str(row.get("remark", "")))


def _load_offline_comparison(query: str) -> list[dict]:
    if not SEMANTIC_VS_BM25_CSV.exists():
        return []
    import csv

    matches: list[dict] = []
    with SEMANTIC_VS_BM25_CSV.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    if not rows:
        return []
    queries = sorted({r.get("query", "") for r in rows})
    chosen = next(
        (q for q in queries if q.lower() == query.lower()),
        None,
    ) or next(
        (q for q in queries if query.lower() in q.lower()),
        None,
    ) or queries[0]
    matches = [r for r in rows if r.get("query") == chosen]
    return matches


def render_metrics_tab(st_module: Any, client: DemoApiClient) -> None:
    st_module.subheader("Demo metrics")
    st_module.caption(
        "Live metrics scraped from the API. Query volume and latency are "
        "in-process counters; the satisfaction proxy is computed from "
        "thumbs-up/down feedback events."
    )

    if st_module.button("Refresh metrics", key="metrics_refresh"):
        st_module.session_state["_metrics_refresh_token"] = (
            st_module.session_state.get("_metrics_refresh_token", 0) + 1
        )

    try:
        metrics_result = client.metrics()
    except Exception as exc:  # pragma: no cover - network path
        st_module.error(f"Could not load metrics: {exc}")
        return
    if not metrics_result.ok():
        st_module.error(f"Metrics endpoint returned {metrics_result.status_code}")
        return
    data = metrics_result.data

    top_cols = st_module.columns(4)
    top_cols[0].metric("Requests (total)", data.get("query_volume_total", 0))
    top_cols[1].metric("Last hour", data.get("query_volume_last_hour", 0))
    top_cols[2].metric("Search calls", data.get("search_requests_total", 0))
    proxy = data.get("satisfaction_proxy")
    top_cols[3].metric(
        "Satisfaction",
        f"{proxy * 100:.0f}%" if proxy is not None else "—",
        delta=(
            f"{data.get('thumbs_up', 0)} up · {data.get('thumbs_down', 0)} down"
        ),
    )

    latency = data.get("latency_ms", {})
    lat_cols = st_module.columns(4)
    lat_cols[0].metric("Latency p50", f"{latency.get('p50', 0):.0f} ms")
    lat_cols[1].metric("Latency p95", f"{latency.get('p95', 0):.0f} ms")
    lat_cols[2].metric("Latency p99", f"{latency.get('p99', 0):.0f} ms")
    lat_cols[3].metric("Latency max", f"{latency.get('max', 0):.0f} ms")

    st_module.divider()

    chart_cols = st_module.columns(2)
    with chart_cols[0]:
        st_module.markdown("**Search modes**")
        mode_rows = aggregate_mode_distribution(data.get("mode_distribution", {}))
        if mode_rows:
            st_module.bar_chart(
                {row["mode"]: row["count"] for row in mode_rows},
                use_container_width=True,
            )
        else:
            st_module.caption("_no search calls yet_")
    with chart_cols[1]:
        st_module.markdown("**Cache**")
        cache = data.get("cache", {})
        st_module.write(
            {
                "size": cache.get("size", 0),
                "maxsize": cache.get("maxsize", 0),
                "hits": cache.get("hits", 0),
                "misses": cache.get("misses", 0),
                "hit_rate": (
                    cache.get("hits", 0)
                    / max(cache.get("hits", 0) + cache.get("misses", 0), 1)
                ),
            }
        )

    st_module.caption(
        f"Feedback events: {data.get('feedback_events_total', 0)} "
        f"(+{data.get('thumbs_up', 0)} / -{data.get('thumbs_down', 0)} / "
        f"~{data.get('thumbs_neutral', 0)})"
    )


def render_sidebar(st_module: Any) -> None:
    st_module.sidebar.title("Real Estate NLP demo")
    st_module.sidebar.caption("Week 11 product integration")
    st_module.sidebar.text_input(
        "API base URL",
        value=st_module.session_state.get("api_base_url", DEFAULT_API_URL),
        key="api_base_url",
        help="Where the FastAPI service lives.",
    )
    st_module.sidebar.selectbox(
        "Default search mode",
        ["hybrid", "semantic", "keyword"],
        index=["hybrid", "semantic", "keyword"].index(
            st_module.session_state.get("default_mode", "hybrid")
        ),
        key="default_mode",
    )
    st_module.sidebar.slider(
        "Default top_k",
        1,
        20,
        int(st_module.session_state.get("default_top_k", 5)),
        key="default_top_k",
    )
    st_module.sidebar.markdown("---")
    st_module.sidebar.markdown(
        "**Demo flow**: enter a natural-language query → see parsed "
        "filters → semantic + BM25 retrieval → enriched listings + "
        "summaries. Compare retrieval strategies on the *Compare* tab; "
        "track usage on the *Metrics* tab."
    )


def main() -> None:  # pragma: no cover - exercised via Streamlit, not pytest
    import streamlit as st  # pyright: ignore[reportMissingImports]

    st.set_page_config(
        page_title="Real Estate NLP demo",
        page_icon="🏡",
        layout="wide",
    )
    render_sidebar(st)
    client = _client_from_state(st)

    st.title("Real Estate Intelligent Search")
    tabs = st.tabs(["Search", "Compare", "Metrics"])
    with tabs[0]:
        render_search_tab(st, client)
    with tabs[1]:
        render_compare_tab(st, client)
    with tabs[2]:
        render_metrics_tab(st, client)


if __name__ == "__main__":  # pragma: no cover
    main()
