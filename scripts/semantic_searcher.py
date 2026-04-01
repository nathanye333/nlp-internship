from __future__ import annotations

import argparse
import json
import math
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import faiss
import numpy as np
import pandas as pd


_TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class SearchResult:
    listing_id: str
    remark: str
    score: float


@dataclass(frozen=True)
class LatencyStats:
    mean_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float


class SemanticSearcher:
    """
    FAISS-backed semantic search over listing remarks.

    Uses cosine similarity implemented as normalized vectors + inner product.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        *,
        encoder: Any | None = None,
        batch_size: int = 128,
    ) -> None:
        self.model_name = model_name
        self._encoder = encoder
        self.batch_size = batch_size
        self.index: faiss.IndexFlatIP | None = None
        self.listing_ids: list[str] = []
        self.remarks: list[str] = []
        self.embedding_dim: int | None = None

    @property
    def is_ready(self) -> bool:
        return self.index is not None and bool(self.remarks)

    def _ensure_encoder(self) -> Any:
        if self._encoder is not None:
            return self._encoder
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for SemanticSearcher when no custom "
                "encoder is provided. Install with `pip install sentence-transformers`."
            ) from exc
        self._encoder = SentenceTransformer(self.model_name)
        return self._encoder

    @staticmethod
    def _normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
        arr = np.asarray(embeddings, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D embedding matrix, got shape={arr.shape}")
        faiss.normalize_L2(arr)  # pyright: ignore[reportCallIssue]
        return arr

    def build_index(self, remarks: Sequence[str], listing_ids: Sequence[str] | None = None) -> None:
        if not remarks:
            raise ValueError("Cannot build index with empty remarks.")

        self.remarks = [str(r) for r in remarks]
        if listing_ids is None:
            self.listing_ids = [str(i) for i in range(len(self.remarks))]
        else:
            if len(listing_ids) != len(self.remarks):
                raise ValueError("listing_ids length must match remarks length.")
            self.listing_ids = [str(v) for v in listing_ids]

        encoder = self._ensure_encoder()
        embeddings = encoder.encode(
            self.remarks,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        vectors = self._normalize_embeddings(embeddings)
        self.embedding_dim = int(vectors.shape[1])
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(vectors)  # pyright: ignore[reportCallIssue]

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        if not self.is_ready or self.index is None:
            raise RuntimeError("Index is not built. Call build_index first.")
        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        encoder = self._ensure_encoder()
        query_vectors = encoder.encode([query], show_progress_bar=False)
        query_vectors = self._normalize_embeddings(query_vectors)
        scores, indices = self.index.search(  # pyright: ignore[reportCallIssue]
            query_vectors,
            min(top_k, len(self.remarks)),
        )

        results: list[SearchResult] = []
        for rank, idx in enumerate(indices[0]):
            if idx < 0:
                continue
            results.append(
                SearchResult(
                    listing_id=self.listing_ids[idx],
                    remark=self.remarks[idx],
                    score=float(scores[0][rank]),
                )
            )
        return results

    def save(self, index_path: str | Path, metadata_path: str | Path) -> None:
        if not self.is_ready or self.index is None:
            raise RuntimeError("Index is not built. Nothing to save.")

        index_path = Path(index_path)
        metadata_path = Path(metadata_path)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(index_path))
        metadata = {
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
            "listing_ids": self.listing_ids,
            "remarks": self.remarks,
        }
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    def load(self, index_path: str | Path, metadata_path: str | Path) -> None:
        index_path = Path(index_path)
        metadata_path = Path(metadata_path)

        self.index = faiss.read_index(str(index_path))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.model_name = metadata["model_name"]
        self.embedding_dim = int(metadata["embedding_dim"])
        self.listing_ids = [str(v) for v in metadata["listing_ids"]]
        self.remarks = [str(v) for v in metadata["remarks"]]


class BM25Searcher:
    """Simple BM25 baseline for keyword retrieval quality comparison."""

    def __init__(self, *, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.listing_ids: list[str] = []
        self.remarks: list[str] = []
        self._doc_tokens: list[list[str]] = []
        self._doc_term_freqs: list[dict[str, int]] = []
        self._doc_lengths: list[int] = []
        self._idf: dict[str, float] = {}
        self._avgdl: float = 0.0

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return _TOKEN_RE.findall(text.lower())

    def build_index(self, remarks: Sequence[str], listing_ids: Sequence[str] | None = None) -> None:
        if not remarks:
            raise ValueError("Cannot build BM25 index with empty remarks.")
        self.remarks = [str(r) for r in remarks]
        if listing_ids is None:
            self.listing_ids = [str(i) for i in range(len(self.remarks))]
        else:
            if len(listing_ids) != len(self.remarks):
                raise ValueError("listing_ids length must match remarks length.")
            self.listing_ids = [str(v) for v in listing_ids]

        self._doc_tokens = [self.tokenize(text) for text in self.remarks]
        self._doc_lengths = [len(tokens) for tokens in self._doc_tokens]
        self._avgdl = float(np.mean(self._doc_lengths)) if self._doc_lengths else 0.0

        self._doc_term_freqs = []
        doc_freq: dict[str, int] = {}
        for tokens in self._doc_tokens:
            tf: dict[str, int] = {}
            for term in tokens:
                tf[term] = tf.get(term, 0) + 1
            self._doc_term_freqs.append(tf)
            for term in tf:
                doc_freq[term] = doc_freq.get(term, 0) + 1

        n_docs = len(self._doc_tokens)
        self._idf = {}
        for term, df in doc_freq.items():
            self._idf[term] = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        if not self.remarks:
            raise RuntimeError("BM25 index not built. Call build_index first.")
        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        q_terms = self.tokenize(query)
        if not q_terms:
            return []

        scores = np.zeros(len(self.remarks), dtype=np.float32)
        for i, term_freqs in enumerate(self._doc_term_freqs):
            score = 0.0
            dl = self._doc_lengths[i]
            denom_norm = self.k1 * (1.0 - self.b + self.b * dl / max(self._avgdl, 1e-9))
            for term in q_terms:
                f = term_freqs.get(term, 0)
                if f == 0:
                    continue
                idf = self._idf.get(term, 0.0)
                score += idf * (f * (self.k1 + 1.0)) / (f + denom_norm)
            scores[i] = score

        top_indices = np.argsort(scores)[::-1][: min(top_k, len(scores))]
        return [
            SearchResult(
                listing_id=self.listing_ids[idx],
                remark=self.remarks[idx],
                score=float(scores[idx]),
            )
            for idx in top_indices
            if scores[idx] > 0
        ]


def benchmark_latency(
    searcher: SemanticSearcher,
    queries: Sequence[str],
    *,
    top_k: int = 10,
    warmup: int = 5,
) -> LatencyStats:
    if not queries:
        raise ValueError("queries cannot be empty.")
    for i in range(min(warmup, len(queries))):
        searcher.search(queries[i], top_k=top_k)

    durations_ms: list[float] = []
    for query in queries:
        start = time.perf_counter()
        searcher.search(query, top_k=top_k)
        durations_ms.append((time.perf_counter() - start) * 1000.0)

    arr = np.array(durations_ms, dtype=np.float64)
    return LatencyStats(
        mean_ms=float(arr.mean()),
        p95_ms=float(np.percentile(arr, 95)),
        p99_ms=float(np.percentile(arr, 99)),
        max_ms=float(arr.max()),
    )


def build_comparison_table(
    semantic_searcher: SemanticSearcher,
    bm25_searcher: BM25Searcher,
    queries: Sequence[str],
    *,
    top_k: int = 5,
) -> pd.DataFrame:
    rows: list[dict] = []
    for query in queries:
        sem_results = semantic_searcher.search(query, top_k=top_k)
        bm_results = bm25_searcher.search(query, top_k=top_k)
        for rank, result in enumerate(sem_results, start=1):
            rows.append(
                {
                    "query": query,
                    "method": "semantic",
                    "rank": rank,
                    "listing_id": result.listing_id,
                    "score": result.score,
                    "remark": result.remark,
                }
            )
        for rank, result in enumerate(bm_results, start=1):
            rows.append(
                {
                    "query": query,
                    "method": "bm25",
                    "rank": rank,
                    "listing_id": result.listing_id,
                    "score": result.score,
                    "remark": result.remark,
                }
            )
    return pd.DataFrame(rows)


def build_relevance_pairs(
    comparison_df: pd.DataFrame,
    *,
    per_method_per_query: int = 1,
    target_pairs: int = 50,
) -> pd.DataFrame:
    if comparison_df.empty:
        return pd.DataFrame(columns=["query", "method", "rank", "listing_id", "score", "remark", "relevant"])

    query_series = comparison_df.get("query")
    if query_series is None:
        return pd.DataFrame(columns=["query", "method", "rank", "listing_id", "score", "remark", "relevant"])

    rows: list[dict] = []
    for query in query_series.dropna().astype(str).drop_duplicates():
        qdf = comparison_df[comparison_df["query"] == query]
        for method in ("semantic", "bm25"):
            subset = pd.DataFrame(qdf[qdf["method"] == method]).sort_values(by="rank").head(per_method_per_query)
            for _, rec in subset.iterrows():
                rows.append(
                    {
                        "query": rec["query"],
                        "method": rec["method"],
                        "rank": int(rec["rank"]),
                        "listing_id": str(rec["listing_id"]),
                        "score": float(rec["score"]),
                        "remark": rec["remark"],
                        "relevant": "",
                    }
                )
    pairs_df = pd.DataFrame(rows).head(target_pairs).copy()
    return pairs_df


def summarize_relevance_scores(relevance_df: pd.DataFrame) -> pd.DataFrame:
    required = {"query", "method", "relevant"}
    missing = required - set(relevance_df.columns)
    if missing:
        raise ValueError(f"Missing required columns for evaluation: {sorted(missing)}")

    scored = relevance_df.copy()
    scored["relevant"] = pd.to_numeric(scored["relevant"], errors="coerce")
    scored = scored.dropna(subset=["relevant"])
    if scored.empty:
        return pd.DataFrame(
            columns=[
                "method",
                "num_pairs",
                "mean_relevance",
                "relevant_rate",
            ]
        )
    grouped = scored.groupby("method")["relevant"]
    summary = grouped.agg(
        num_pairs="count",
        mean_relevance="mean",
    ).reset_index()
    summary["relevant_rate"] = (summary["mean_relevance"] >= 0.5).astype(float)
    return summary.sort_values("method").reset_index(drop=True)


def _load_remarks_dataset(dataset_path: Path) -> tuple[list[str], list[str]]:
    df = pd.read_csv(dataset_path)
    if "remarks" not in df.columns:
        raise ValueError("Dataset must contain a `remarks` column.")
    remarks = df["remarks"].fillna("").astype(str).tolist()
    listing_ids = (
        df["L_ListingID"].astype(str).tolist()
        if "L_ListingID" in df.columns
        else [str(i) for i in range(len(df))]
    )
    return remarks, listing_ids


def _load_queries(queries_path: Path, limit: int = 25) -> list[str]:
    qdf = pd.read_csv(queries_path)
    if "query" not in qdf.columns:
        raise ValueError("Query file must contain a `query` column.")
    queries = qdf["query"].dropna().astype(str).tolist()
    return queries[:limit]


def _repeat_to_size(items: Sequence[str], target_size: int) -> list[str]:
    if not items:
        return []
    if len(items) >= target_size:
        return list(items[:target_size])
    out = list(items)
    i = 0
    while len(out) < target_size:
        out.append(items[i % len(items)])
        i += 1
    return out


def run_week5_pipeline(
    *,
    dataset_path: Path,
    queries_path: Path,
    index_path: Path,
    metadata_path: Path,
    comparison_out: Path,
    relevance_pairs_out: Path,
    labeled_relevance_path: Path | None = None,
    latency_target_ms: float = 100.0,
) -> dict:
    remarks, listing_ids = _load_remarks_dataset(dataset_path)

    semantic_searcher = SemanticSearcher()
    semantic_searcher.build_index(remarks, listing_ids=listing_ids)
    semantic_searcher.save(index_path, metadata_path)

    bm25 = BM25Searcher()
    bm25.build_index(remarks, listing_ids=listing_ids)

    queries = _load_queries(queries_path, limit=25)
    latency_queries = _repeat_to_size(queries, target_size=100)
    latency_stats = benchmark_latency(semantic_searcher, latency_queries, top_k=10, warmup=10)
    comparison_df = build_comparison_table(semantic_searcher, bm25, queries, top_k=5)
    comparison_out.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(comparison_out, index=False)

    relevance_pairs = build_relevance_pairs(
        comparison_df,
        per_method_per_query=1,
        target_pairs=50,
    )
    relevance_pairs_out.parent.mkdir(parents=True, exist_ok=True)
    relevance_pairs.to_csv(relevance_pairs_out, index=False)

    relevance_summary = None
    if labeled_relevance_path is not None and labeled_relevance_path.exists():
        labeled_df = pd.read_csv(labeled_relevance_path)
        relevance_summary = summarize_relevance_scores(labeled_df)

    # Benchmark against a 10k listing index (replicated if source sample is smaller).
    scaled_remarks = _repeat_to_size(remarks, target_size=10_000)
    scaled_ids = [str(i) for i in range(len(scaled_remarks))]
    scaled_searcher = SemanticSearcher(
        model_name=semantic_searcher.model_name,
        encoder=semantic_searcher._ensure_encoder(),
        batch_size=semantic_searcher.batch_size,
    )
    scaled_searcher.build_index(scaled_remarks, listing_ids=scaled_ids)
    latency_10k_stats = benchmark_latency(scaled_searcher, latency_queries, top_k=10, warmup=10)

    return {
        "num_listings": len(remarks),
        "embedding_dim": semantic_searcher.embedding_dim,
        "latency_ms": asdict(latency_stats),
        "latency_10k_ms": asdict(latency_10k_stats),
        "latency_target_ms": latency_target_ms,
        "latency_target_met": latency_10k_stats.p95_ms < latency_target_ms,
        "comparison_rows": len(comparison_df),
        "relevance_pairs_rows": len(relevance_pairs),
        "relevance_summary": None if relevance_summary is None else relevance_summary.to_dict("records"),
    }


def _default_path(relative_path: str) -> Path:
    return Path(__file__).resolve().parents[1] / relative_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Week 5 semantic search pipeline.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=_default_path("data/processed/listing_sample_cleaned.csv"),
        help="CSV with listing remarks.",
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=_default_path("data/processed/user_queries.csv"),
        help="CSV with query column.",
    )
    parser.add_argument(
        "--index-out",
        type=Path,
        default=_default_path("data/models/listings_semantic.faiss"),
    )
    parser.add_argument(
        "--meta-out",
        type=Path,
        default=_default_path("data/models/listings_semantic_meta.json"),
    )
    parser.add_argument(
        "--comparison-out",
        type=Path,
        default=_default_path("data/processed/semantic_vs_bm25.csv"),
    )
    parser.add_argument(
        "--relevance-out",
        type=Path,
        default=_default_path("data/processed/relevance_pairs_template.csv"),
    )
    parser.add_argument(
        "--labeled-relevance",
        type=Path,
        default=None,
        help="Optional labeled CSV (same format as relevance template with relevant column filled 0/1).",
    )
    parser.add_argument(
        "--latency-target-ms",
        type=float,
        default=100.0,
    )
    args = parser.parse_args()

    report = run_week5_pipeline(
        dataset_path=args.dataset,
        queries_path=args.queries,
        index_path=args.index_out,
        metadata_path=args.meta_out,
        comparison_out=args.comparison_out,
        relevance_pairs_out=args.relevance_out,
        labeled_relevance_path=args.labeled_relevance,
        latency_target_ms=args.latency_target_ms,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()