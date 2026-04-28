import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from scripts.semantic_searcher import (
    BM25Searcher,
    SemanticSearcher,
    benchmark_latency,
    build_comparison_table,
    build_relevance_pairs,
    summarize_relevance_scores,
)


class FakeEncoder:
    """
    Deterministic, fast encoder for tests.
    Produces semantically aligned vectors for synonyms and stable hashed vectors for others.
    """

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim
        self._synonyms = {
            "pool": np.array([1.0, 0.0, 0.0], dtype=np.float32),
            "swimming": np.array([1.0, 0.0, 0.0], dtype=np.float32),
            "garage": np.array([0.0, 1.0, 0.0], dtype=np.float32),
            "carport": np.array([0.0, 1.0, 0.0], dtype=np.float32),
            "waterfront": np.array([0.0, 0.0, 1.0], dtype=np.float32),
            "ocean": np.array([0.0, 0.0, 1.0], dtype=np.float32),
        }

    def _embed_text(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        lowered = text.lower()
        for token, basis in self._synonyms.items():
            if token in lowered:
                vec[:3] += basis

        if not np.any(vec):
            digest = hashlib.sha256(lowered.encode("utf-8")).digest()
            raw = np.frombuffer(digest, dtype=np.uint8).astype(np.float32)
            padded = np.resize(raw, self.dim)
            vec = padded / 255.0
        return vec

    def encode(self, texts, *, batch_size: int = 64, show_progress_bar: bool = False) -> np.ndarray:
        _ = (batch_size, show_progress_bar)
        return np.vstack([self._embed_text(t) for t in texts]).astype(np.float32)


def test_semantic_search_returns_expected_neighbors():
    remarks = [
        "Beautiful home with swimming pool and patio",
        "Condo with attached garage and storage",
        "Waterfront property with ocean views",
    ]
    ids = ["A", "B", "C"]
    searcher = SemanticSearcher(encoder=FakeEncoder())
    searcher.build_index(remarks, listing_ids=ids)

    results = searcher.search("house with pool", top_k=2)
    assert len(results) == 2
    assert results[0].listing_id == "A"
    assert searcher.embedding_dim == 384


def test_semantic_search_supports_allowed_listing_ids():
    remarks = [
        "Beautiful home with swimming pool and patio",
        "Condo with attached garage and storage",
        "Waterfront property with ocean views",
    ]
    ids = ["A", "B", "C"]
    searcher = SemanticSearcher(encoder=FakeEncoder())
    searcher.build_index(remarks, listing_ids=ids)

    results = searcher.search("house with pool", top_k=3, allowed_listing_ids={"B", "C"})
    assert {r.listing_id for r in results} == {"B", "C"}
    assert len(results) == 2


def test_bm25_searcher_prefers_exact_keyword_match():
    remarks = [
        "Loft with skyline view",
        "Family house with garage",
        "Garage plus workshop and parking",
    ]
    bm25 = BM25Searcher()
    bm25.build_index(remarks)

    results = bm25.search("workshop garage", top_k=2)
    assert results
    assert "workshop" in results[0].remark.lower()


def test_bm25_search_supports_allowed_listing_ids():
    remarks = [
        "Loft with skyline view",
        "Family house with garage",
        "Garage plus workshop and parking",
    ]
    bm25 = BM25Searcher()
    bm25.build_index(remarks, listing_ids=["A", "B", "C"])

    results = bm25.search("workshop garage", top_k=3, allowed_listing_ids={"B"})
    assert [r.listing_id for r in results] == ["B"]


def test_comparison_table_and_relevance_pair_generation():
    remarks = [
        "Townhome with pool and gym",
        "Single family home with garage",
        "Beachfront condo with ocean view",
    ]
    searcher = SemanticSearcher(encoder=FakeEncoder())
    searcher.build_index(remarks, listing_ids=["1", "2", "3"])

    bm25 = BM25Searcher()
    bm25.build_index(remarks, listing_ids=["1", "2", "3"])

    queries = ["pool home", "ocean condo", "garage house"]
    comparison_df = build_comparison_table(searcher, bm25, queries, top_k=2)
    assert {"query", "method", "rank", "listing_id", "score", "remark"}.issubset(comparison_df.columns)

    pairs = build_relevance_pairs(comparison_df, per_method_per_query=1, target_pairs=6)
    assert len(pairs) == 6
    assert set(pairs["method"]) == {"semantic", "bm25"}
    assert "relevant" in pairs.columns


def test_relevance_summary_supports_fifty_pairs():
    queries = [f"query-{i}" for i in range(25)]
    rows = []
    for q in queries:
        rows.append({"query": q, "method": "semantic", "relevant": 1})
        rows.append({"query": q, "method": "bm25", "relevant": 0})
    df = pd.DataFrame(rows)

    assert len(df) == 50
    summary = summarize_relevance_scores(df)
    assert set(summary["method"]) == {"semantic", "bm25"}
    semantic_mean = float(np.asarray(summary.loc[summary["method"] == "semantic", "mean_relevance"])[0])
    bm25_mean = float(np.asarray(summary.loc[summary["method"] == "bm25", "mean_relevance"])[0])
    assert semantic_mean > bm25_mean


@pytest.mark.performance
def test_semantic_search_latency_under_100ms_for_10k():
    n = 10_000
    remarks = [f"listing {i} with pool and garage features" for i in range(n)]
    searcher = SemanticSearcher(encoder=FakeEncoder())
    searcher.build_index(remarks)

    queries = [f"query pool garage {i}" for i in range(100)]
    stats = benchmark_latency(searcher, queries, top_k=10, warmup=5)

    # Deliverable: latency < 100ms for 10k listings.
    assert stats.p95_ms < 100.0, f"p95 latency too high: {stats.p95_ms:.2f} ms"
