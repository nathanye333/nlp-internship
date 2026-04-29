"""Production REST API exposing the project's NLP capabilities.

This module wires the subroutines implemented across weeks 1-9 -- query
parsing, entity extraction, semantic search, listing summarization, and Fair
Housing compliance checking -- into a single FastAPI application.

Features
--------
* 9 HTTP endpoints (8 POST endpoints plus supporting ``/health`` and cache
  management). See the auto-generated OpenAPI docs at ``/docs`` for full
  schemas.
* Request / response validation via Pydantic models.
* An in-memory TTL + LRU response cache (keyed by endpoint + payload) with
  introspection via ``/cache/stats``.
* Per-IP rate limiting (10 requests / second) backed by ``slowapi``.
* Structured JSON logging of every request (method, path, client, latency).

Run locally::

    uvicorn scripts.production_api:app --host 0.0.0.0 --port 8000

Then visit ``http://localhost:8000/docs`` for interactive API docs.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import time
from collections import OrderedDict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock, RLock
from typing import Any, Callable, Deque, Literal, Optional

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.compliance_checker import ComplianceChecker  # noqa: E402
from scripts.entity_extractor import EntityExtractor  # noqa: E402
from scripts.listing_metadata import (  # noqa: E402
    ListingMetadataStore,
    default_store as default_metadata_store,
)
from scripts.listing_submission_example import (  # noqa: E402
    Listing,
    ListingSubmissionError,
    ListingSubmissionService,
)
from scripts.listing_summarizer import ListingSummarizer  # noqa: E402
from scripts.query_parser import QueryParser  # noqa: E402

# ``semantic_searcher`` pulls in faiss + sentence-transformers which are heavy
# native dependencies. We defer that import until the searcher is actually
# requested so that the rest of the API (and the test suite) remain usable in
# stripped-down environments.
if False:  # pragma: no cover - typing only
    from scripts.semantic_searcher import BM25Searcher, SearchResult, SemanticSearcher


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger("production_api")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(os.environ.get("API_LOG_LEVEL", "INFO"))


# ---------------------------------------------------------------------------
# In-memory TTL + LRU cache
# ---------------------------------------------------------------------------


class TTLCache:
    """Small thread-safe TTL + LRU cache used for response memoization.

    The cache is deliberately dependency-free so the API can ship without
    Redis; swap this out for a redis-backed implementation if you need to
    share state across processes.
    """

    def __init__(self, maxsize: int = 512, ttl_seconds: float = 60.0) -> None:
        self.maxsize = maxsize
        self.ttl_seconds = ttl_seconds
        self._data: "OrderedDict[str, tuple[Any, float]]" = OrderedDict()
        self._lock = Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Any:
        with self._lock:
            item = self._data.get(key)
            if item is None:
                self.misses += 1
                return None
            value, expires_at = item
            if expires_at <= time.time():
                del self._data[key]
                self.misses += 1
                return None
            self._data.move_to_end(key)
            self.hits += 1
            return value

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._data[key] = (value, time.time() + self.ttl_seconds)
            self._data.move_to_end(key)
            while len(self._data) > self.maxsize:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self.hits = 0
            self.misses = 0

    def stats(self) -> dict:
        with self._lock:
            return {
                "size": len(self._data),
                "maxsize": self.maxsize,
                "ttl_seconds": self.ttl_seconds,
                "hits": self.hits,
                "misses": self.misses,
            }


_cache = TTLCache(maxsize=512, ttl_seconds=60.0)


def _cache_key(prefix: str, payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return f"{prefix}:{hashlib.sha1(data).hexdigest()}"


# ---------------------------------------------------------------------------
# Lazy-loaded NLP components
# ---------------------------------------------------------------------------


class _Components:
    """Container that instantiates heavy NLP components on first access.

    The semantic searcher in particular requires downloading a
    sentence-transformer model and loading a FAISS index, so we avoid doing
    that eagerly at import time. Each accessor is idempotent and thread-safe.
    """

    def __init__(self) -> None:
        # ``RLock`` so methods that call into other component accessors
        # (e.g. ``submission_service`` -> ``compliance_checker``) do not
        # deadlock on the same thread.
        self._lock = RLock()
        self._query_parser: Optional[QueryParser] = None
        self._entity_extractor: Optional[EntityExtractor] = None
        self._summarizer: Optional[ListingSummarizer] = None
        self._compliance_checker: Optional[ComplianceChecker] = None
        self._submission_service: Optional[ListingSubmissionService] = None
        self._searcher: Any = None
        self._bm25_searcher: Any = None
        self._searcher_error: Optional[str] = None
        self._metadata_store: Optional[ListingMetadataStore] = None

    def query_parser(self) -> QueryParser:
        with self._lock:
            if self._query_parser is None:
                self._query_parser = QueryParser()
            return self._query_parser

    def entity_extractor(self) -> EntityExtractor:
        with self._lock:
            if self._entity_extractor is None:
                self._entity_extractor = EntityExtractor()
            return self._entity_extractor

    def summarizer(self) -> ListingSummarizer:
        with self._lock:
            if self._summarizer is None:
                self._summarizer = ListingSummarizer()
            return self._summarizer

    def compliance_checker(self) -> ComplianceChecker:
        with self._lock:
            if self._compliance_checker is None:
                self._compliance_checker = ComplianceChecker()
            return self._compliance_checker

    def submission_service(self) -> ListingSubmissionService:
        with self._lock:
            if self._submission_service is None:
                self._submission_service = ListingSubmissionService(
                    checker=self.compliance_checker()
                )
            return self._submission_service

    def semantic_searcher(self) -> Any:
        """Return a ready semantic searcher or raise HTTPException(503).

        The searcher is loaded from the Week 5 artefacts
        (``data/models/listings_semantic.faiss`` + JSON metadata).  If those
        artefacts are missing or the sentence-transformer backend fails to
        initialise we raise a 503 so clients can fall back to keyword search.
        """
        with self._lock:
            if self._searcher is not None:
                return self._searcher
            if self._searcher_error is not None:
                raise HTTPException(
                    status_code=503,
                    detail=f"Semantic searcher unavailable: {self._searcher_error}",
                )
            index_path = _PROJECT_ROOT / "data" / "models" / "listings_semantic.faiss"
            meta_path = _PROJECT_ROOT / "data" / "models" / "listings_semantic_meta.json"
            if not index_path.exists() or not meta_path.exists():
                self._searcher_error = (
                    f"index artefacts not found at {index_path} / {meta_path}"
                )
                raise HTTPException(status_code=503, detail=self._searcher_error)
            try:
                from scripts.semantic_searcher import BM25Searcher, SemanticSearcher
                searcher = SemanticSearcher()
                searcher.load(index_path, meta_path)
                self._searcher = searcher
                bm25 = BM25Searcher()
                bm25.build_index(searcher.remarks, listing_ids=searcher.listing_ids)
                self._bm25_searcher = bm25
                return searcher
            except Exception as exc:  # pragma: no cover - depends on env
                self._searcher_error = str(exc)
                logger.exception("Failed to load semantic searcher")
                raise HTTPException(
                    status_code=503,
                    detail=f"Semantic searcher unavailable: {exc}",
                ) from exc

    def bm25_searcher(self) -> Any:
        """Return BM25 searcher built from saved remark metadata."""
        with self._lock:
            if self._bm25_searcher is not None:
                return self._bm25_searcher
            meta_path = _PROJECT_ROOT / "data" / "models" / "listings_semantic_meta.json"
            if meta_path.exists():
                from scripts.semantic_searcher import BM25Searcher

                metadata = json.loads(meta_path.read_text(encoding="utf-8"))
                bm25 = BM25Searcher()
                bm25.build_index(
                    metadata.get("remarks", []),
                    listing_ids=metadata.get("listing_ids"),
                )
                self._bm25_searcher = bm25
                return bm25
            # Fall back to the semantic loader in older/dev setups where the
            # BM25 corpus has not been materialized independently.
            self.semantic_searcher()
            return self._bm25_searcher

    def warmup_semantic(self) -> None:
        """Load semantic/BM25 components and run one tiny query embedding."""
        searcher = self.semantic_searcher()
        # ``load`` reads the FAISS index but leaves the sentence-transformer
        # encoder lazy. A one-result search pays that model-load cost during
        # startup instead of the first user-facing hybrid/semantic query.
        searcher.search("__warmup__", top_k=1)
        self.bm25_searcher()

    def metadata_store(self) -> ListingMetadataStore:
        """Return the listing metadata store (CSV-backed, lazy-loaded)."""
        with self._lock:
            if self._metadata_store is None:
                self._metadata_store = default_metadata_store()
            return self._metadata_store

    def reset(self) -> None:
        """Clear cached component instances. Used mainly by tests."""
        with self._lock:
            self._query_parser = None
            self._entity_extractor = None
            self._summarizer = None
            self._compliance_checker = None
            self._submission_service = None
            self._searcher = None
            self._bm25_searcher = None
            self._searcher_error = None
            self._metadata_store = None


components = _Components()


# ---------------------------------------------------------------------------
# Dependency injection shims (so tests can override without monkeypatching)
# ---------------------------------------------------------------------------


def get_query_parser() -> QueryParser:
    return components.query_parser()


def get_entity_extractor() -> EntityExtractor:
    return components.entity_extractor()


def get_summarizer() -> ListingSummarizer:
    return components.summarizer()


def get_compliance_checker() -> ComplianceChecker:
    return components.compliance_checker()


def get_submission_service() -> ListingSubmissionService:
    return components.submission_service()


def get_semantic_searcher() -> Any:
    return components.semantic_searcher()


def get_bm25_searcher() -> Any:
    try:
        return components.bm25_searcher()
    except HTTPException:
        # Keep search endpoint usable with semantic-only retrieval in test/dev
        # environments where persisted artefacts are unavailable.
        return None


def get_metadata_store() -> ListingMetadataStore:
    return components.metadata_store()


# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------


SearchMode = Literal["hybrid", "semantic", "keyword"]
CompareMode = Literal["semantic", "keyword", "hybrid", "bm25_raw"]


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="Free-text user query")
    top_k: int = Field(10, ge=1, le=100, description="Maximum results to return")
    mode: SearchMode = Field(
        "hybrid",
        description=(
            "Retrieval mode: 'hybrid' fuses semantic + BM25 scores, 'semantic' "
            "returns only sentence-transformer results, 'keyword' returns only "
            "BM25 (lexical) results."
        ),
    )

    @field_validator("query")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("query must not be blank")
        return v


class SearchHit(BaseModel):
    listing_id: str
    remark: str
    score: float


class SearchResponse(BaseModel):
    query: str
    filters: dict
    results: list[SearchHit]
    count: int
    mode: SearchMode = "hybrid"
    latency_ms: float = 0.0
    cached: bool = False


class RawBM25SearchResponse(BaseModel):
    query: str
    filters: dict = Field(default_factory=dict)
    results: list[SearchHit]
    count: int
    mode: Literal["bm25_raw"] = "bm25_raw"
    latency_ms: float = 0.0
    cached: bool = False


class CompareRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    top_k: int = Field(10, ge=1, le=50)

    @field_validator("query")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("query must not be blank")
        return v


class CompareModeResult(BaseModel):
    mode: CompareMode
    results: list[SearchHit]
    count: int
    latency_ms: float
    available: bool = True
    error: Optional[str] = None


class CompareOverlap(BaseModel):
    semantic_vs_keyword: int
    semantic_vs_hybrid: int
    keyword_vs_hybrid: int
    all_three: int


class CompareResponse(BaseModel):
    query: str
    filters: dict
    top_k: int
    modes: dict[str, CompareModeResult]
    overlap: CompareOverlap


class ListingDetail(BaseModel):
    listing_id: str
    address: Optional[str] = None
    city: Optional[str] = None
    beds: Optional[float] = None
    baths: Optional[float] = None
    price: Optional[int] = None
    sqft: Optional[int] = None
    remarks: Optional[str] = None
    summary: Optional[str] = None
    compliance_ok: Optional[bool] = None
    compliance_error_count: int = 0
    compliance_warning_count: int = 0
    found: bool = True


class FeedbackRequest(BaseModel):
    listing_id: str = Field(..., min_length=1, max_length=64)
    query: str = Field(..., min_length=1, max_length=500)
    mode: SearchMode = "hybrid"
    rating: int = Field(..., ge=-1, le=1, description="-1 thumbs down, 0 neutral, +1 thumbs up")
    latency_ms: float = Field(0.0, ge=0.0)
    note: Optional[str] = Field(None, max_length=500)


class FeedbackResponse(BaseModel):
    status: str
    recorded_at: float
    total_events: int


class LatencyPercentiles(BaseModel):
    p50: float
    p95: float
    p99: float
    max: float
    count: int


class MetricsDashboardResponse(BaseModel):
    query_volume_total: int
    query_volume_last_hour: int
    search_requests_total: int
    feedback_events_total: int
    satisfaction_proxy: Optional[float] = None
    thumbs_up: int
    thumbs_down: int
    thumbs_neutral: int
    latency_ms: LatencyPercentiles
    cache: dict
    mode_distribution: dict


class ParseQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)


class ParseQueryResponse(BaseModel):
    query: str
    filters: dict
    where_sql: str
    params: list


class ExtractEntitiesRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10_000)


class ExtractEntitiesResponse(BaseModel):
    text: str
    bedrooms: Optional[int] = None
    bathrooms: Optional[float] = None
    price: Optional[int] = None
    sqft: list[int] = Field(default_factory=list)
    amenities: list[str] = Field(default_factory=list)


class SummarizeRequest(BaseModel):
    remarks: str = Field(..., min_length=1, max_length=20_000)
    bedrooms: Optional[float] = None
    bathrooms: Optional[float] = None
    price: Optional[float] = None
    city: Optional[str] = None
    features: Optional[list[str]] = None
    num_sentences: int = Field(2, ge=1, le=10)
    mode: str = Field("extractive", pattern="^(extractive|abstractive)$")


class SummarizeResponse(BaseModel):
    summary: str
    mode: str


class ComplianceRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=20_000)


class ComplianceViolation(BaseModel):
    category: str
    severity: str
    matched_text: str
    start: int
    end: int
    message: str
    suggestion: str


class ComplianceResponse(BaseModel):
    compliant: bool
    error_count: int
    warning_count: int
    info_count: int
    violations: list[ComplianceViolation]


class SubmitListingRequest(BaseModel):
    listing_id: str = Field(..., min_length=1, max_length=64)
    address: str = Field(..., min_length=1, max_length=256)
    price_usd: int = Field(..., ge=0)
    description: str = Field(..., min_length=1, max_length=20_000)


class SubmitListingResponse(BaseModel):
    listing_id: str
    status: str
    compliance: ComplianceResponse
    audit_log: list[str]


class CacheStatsResponse(BaseModel):
    size: int
    maxsize: int
    ttl_seconds: float
    hits: int
    misses: int


class HealthResponse(BaseModel):
    status: str
    version: str
    semantic_search_ready: bool


# ---------------------------------------------------------------------------
# In-process metrics + feedback log
# ---------------------------------------------------------------------------


_FEEDBACK_LOG_PATH = (
    _PROJECT_ROOT / "data" / "processed" / "demo_event_log.jsonl"
)


class MetricsRegistry:
    """Tiny thread-safe registry tracking request volume / latency / feedback.

    The state lives in-process which is fine for the demo; for a real
    deployment swap this for a Prometheus pull or push to Datadog.
    """

    LATENCY_WINDOW = 1024

    def __init__(self) -> None:
        self._lock = Lock()
        self._latencies: Deque[float] = deque(maxlen=self.LATENCY_WINDOW)
        self._request_timestamps: Deque[float] = deque(maxlen=4096)
        self._search_requests = 0
        self._mode_counts: dict[str, int] = {
            "hybrid": 0,
            "semantic": 0,
            "keyword": 0,
            "bm25_raw": 0,
        }

    def record_request(self, latency_ms: float, *, path: str = "") -> None:
        with self._lock:
            self._latencies.append(float(latency_ms))
            self._request_timestamps.append(time.time())
            if path.startswith("/search"):
                self._search_requests += 1

    def record_search_mode(self, mode: str) -> None:
        with self._lock:
            self._mode_counts[mode] = self._mode_counts.get(mode, 0) + 1

    @staticmethod
    def _percentile(values: list[float], pct: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        if len(ordered) == 1:
            return ordered[0]
        k = (len(ordered) - 1) * pct
        lo = int(k)
        hi = min(lo + 1, len(ordered) - 1)
        frac = k - lo
        return ordered[lo] + (ordered[hi] - ordered[lo]) * frac

    def snapshot(self) -> dict:
        now = time.time()
        with self._lock:
            latencies = list(self._latencies)
            timestamps = list(self._request_timestamps)
            search_total = self._search_requests
            mode_counts = dict(self._mode_counts)
        last_hour = sum(1 for ts in timestamps if now - ts <= 3600)
        return {
            "query_volume_total": len(timestamps),
            "query_volume_last_hour": last_hour,
            "search_requests_total": search_total,
            "mode_distribution": mode_counts,
            "latency_ms": {
                "p50": self._percentile(latencies, 0.5),
                "p95": self._percentile(latencies, 0.95),
                "p99": self._percentile(latencies, 0.99),
                "max": max(latencies) if latencies else 0.0,
                "count": len(latencies),
            },
        }

    def reset(self) -> None:
        with self._lock:
            self._latencies.clear()
            self._request_timestamps.clear()
            self._search_requests = 0
            self._mode_counts = {
                "hybrid": 0,
                "semantic": 0,
                "keyword": 0,
                "bm25_raw": 0,
            }


metrics = MetricsRegistry()


_feedback_lock = Lock()


def feedback_log_path() -> Path:
    """Return the JSONL path used to persist feedback events."""
    return _FEEDBACK_LOG_PATH


def set_feedback_log_path(path: Path | str) -> None:
    """Override the feedback log path (used by tests)."""
    global _FEEDBACK_LOG_PATH
    _FEEDBACK_LOG_PATH = Path(path)


def append_feedback_event(event: dict) -> int:
    """Append ``event`` (already JSON-serialisable) to the feedback log.

    Returns the new total number of events on disk.
    """
    path = feedback_log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(event, default=str)
    with _feedback_lock:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        return _count_feedback_events_unlocked(path)


def _count_feedback_events_unlocked(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


def aggregate_feedback() -> dict:
    """Aggregate the feedback JSONL into satisfaction counts."""
    path = feedback_log_path()
    up = down = neutral = total = 0
    if path.exists():
        with _feedback_lock:
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    total += 1
                    rating = event.get("rating")
                    if rating == 1:
                        up += 1
                    elif rating == -1:
                        down += 1
                    else:
                        neutral += 1
    decided = up + down
    proxy = (up / decided) if decided > 0 else None
    return {
        "feedback_events_total": total,
        "thumbs_up": up,
        "thumbs_down": down,
        "thumbs_neutral": neutral,
        "satisfaction_proxy": proxy,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def apply_filters(results: list, filters: dict) -> list:
    """Post-filter semantic search results using structured query filters.

    The semantic searcher only surfaces listing ids and remark text, so we
    apply lightweight text-level filters here (amenity include/exclude).
    Numeric filters (price / beds / baths / city) are returned in the
    ``filters`` field of the response for downstream SQL execution; we do not
    have the metadata loaded in-process to filter on them.

    Accepts either attribute-style hits (e.g. ``SearchResult`` dataclasses)
    or dict-style hits emitted by :func:`hybrid_retrieve` / :func:`run_search`.
    """

    if not results:
        return results
    includes = [a.lower() for a in filters.get("amenities_in", []) or []]
    excludes = [a.lower() for a in filters.get("amenities_out", []) or []]
    if not includes and not excludes:
        return results

    filtered: list = []
    for r in results:
        remark = _hit_field(r, "remark", "")
        remark_lower = (remark or "").lower()
        if includes and not all(token in remark_lower for token in includes):
            continue
        if excludes and any(token in remark_lower for token in excludes):
            continue
        filtered.append(r)
    return filtered


def _amenity_filters_present(filters: dict) -> bool:
    return bool(filters.get("amenities_in") or filters.get("amenities_out"))


def _metadata_filters_present(filters: dict) -> bool:
    """Return True when filters require structured listing metadata."""
    keys = {
        "price_min",
        "price_max",
        "sqft_min",
        "sqft_max",
        "bedrooms_min",
        "bedrooms_max",
        "bathrooms_min",
        "bathrooms_max",
        "city",
    }
    return any(key in filters for key in keys)


def _matches_range(
    value: Any,
    *,
    minimum: Any = None,
    maximum: Any = None,
) -> bool:
    if minimum is None and maximum is None:
        return True
    if value is None:
        return False
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    if minimum is not None and numeric < float(minimum):
        return False
    if maximum is not None and numeric > float(maximum):
        return False
    return True


def listing_matches_metadata_filters(record: dict | None, filters: dict) -> bool:
    """Evaluate parsed query filters against a listing metadata record.

    This mirrors the SQL generated by :class:`QueryParser` for the fields we
    have in ``listing_sample_cleaned.csv``. Missing metadata is treated as
    non-matching when a structured constraint is present.
    """
    if not _metadata_filters_present(filters):
        return True
    if record is None:
        return False

    city = filters.get("city")
    if city is not None:
        record_city = (record.get("city") or "").strip().lower()
        if record_city != str(city).strip().lower():
            return False

    if not _matches_range(
        record.get("price"),
        minimum=filters.get("price_min"),
        maximum=filters.get("price_max"),
    ):
        return False
    if not _matches_range(
        record.get("beds"),
        minimum=filters.get("bedrooms_min"),
        maximum=filters.get("bedrooms_max"),
    ):
        return False
    if not _matches_range(
        record.get("baths"),
        minimum=filters.get("bathrooms_min"),
        maximum=filters.get("bathrooms_max"),
    ):
        return False
    if not _matches_range(
        record.get("sqft"),
        minimum=filters.get("sqft_min"),
        maximum=filters.get("sqft_max"),
    ):
        return False

    return True


def listing_matches_query_filters(record: dict | None, filters: dict) -> bool:
    """Evaluate the parsed SQL-like constraints against one listing record."""
    if record is None:
        return False
    if not listing_matches_metadata_filters(record, filters):
        return False
    if _amenity_filters_present(filters):
        remark_lower = (
            (record.get("remarks_clean") or record.get("remarks") or "").lower()
        )
        includes = [a.lower() for a in filters.get("amenities_in", []) or []]
        excludes = [a.lower() for a in filters.get("amenities_out", []) or []]
        if includes and not all(token in remark_lower for token in includes):
            return False
        if excludes and any(token in remark_lower for token in excludes):
            return False
    return True


def candidate_listing_ids(
    filters: dict,
    metadata_store: ListingMetadataStore | None,
) -> set[str] | None:
    """Return the SQL-eligible candidate listing IDs before retrieval.

    If no structured/amenity filters are present we return ``None`` so the
    searchers operate over the full corpus. Otherwise, only listings matching
    the parsed query constraints are eligible for semantic/BM25 ranking.
    """
    if metadata_store is None:
        return set()
    if not (_metadata_filters_present(filters) or _amenity_filters_present(filters)):
        return None
    return {
        listing_id
        for listing_id, record in metadata_store.all_records().items()
        if listing_matches_query_filters(record, filters)
    }


def apply_metadata_filters(
    results: list,
    filters: dict,
    metadata_store: ListingMetadataStore | None,
) -> list:
    """Filter search hits by parsed city / price / beds / baths constraints."""
    if not results or not _metadata_filters_present(filters):
        return results
    if metadata_store is None:
        return []

    filtered: list = []
    for hit in results:
        listing_id = str(_hit_field(hit, "listing_id", ""))
        record = metadata_store.get(listing_id)
        if listing_matches_metadata_filters(record, filters):
            filtered.append(hit)
    return filtered


def retrieval_depth(top_k: int, filters: dict, metadata_store: ListingMetadataStore | None) -> int:
    """Fetch a wider candidate pool when structured filters are present.

    The persisted FAISS/BM25 searchers do not support a true SQL pre-filter,
    so the next-best behaviour is to over-retrieve, apply metadata filters
    before final truncation, then return top-k from the filtered candidates.
    """
    if not _metadata_filters_present(filters):
        return top_k
    corpus_size = len(metadata_store) if metadata_store is not None else 0
    if corpus_size <= 0:
        return max(top_k * 10, 100)
    return min(corpus_size, max(top_k * 25, 250))


def _hit_field(item: Any, field: str, default: Any = "") -> Any:
    """Read ``field`` from either an attribute-style hit or a dict-style hit."""
    if isinstance(item, dict):
        return item.get(field, default)
    return getattr(item, field, default)


def _normalize_hits(items: list) -> list[dict]:
    """Normalise a heterogeneous list of search hits to dicts."""
    return [
        {
            "listing_id": str(_hit_field(item, "listing_id", "")),
            "remark": str(_hit_field(item, "remark", "")),
            "score": float(_hit_field(item, "score", 0.0)),
        }
        for item in items
    ]


HYBRID_SEMANTIC_WEIGHT = 0.93
HYBRID_BM25_WEIGHT = 0.07


def _search_with_candidates(
    searcher: Any,
    query: str,
    *,
    top_k: int,
    allowed_listing_ids: set[str] | None = None,
) -> Any:
    """Call a searcher with candidate IDs when supported, else fall back."""
    if allowed_listing_ids is None:
        return searcher.search(query, top_k=top_k)
    try:
        return searcher.search(
            query,
            top_k=top_k,
            allowed_listing_ids=allowed_listing_ids,
        )
    except TypeError:
        # Test stubs and older searcher implementations may not yet accept the
        # allowlist kwarg. Fall back to the legacy signature.
        return searcher.search(query, top_k=top_k)


def run_search(
    query: str,
    *,
    mode: SearchMode,
    top_k: int,
    semantic_searcher: Any,
    bm25_searcher: Any | None,
    allowed_listing_ids: set[str] | None = None,
) -> list[dict]:
    """Dispatch to the requested retrieval mode and return normalised hits."""
    if mode == "semantic":
        if semantic_searcher is None:
            raise HTTPException(
                status_code=503,
                detail="Semantic searcher unavailable for semantic mode",
            )
        return _normalize_hits(
            _search_with_candidates(
                semantic_searcher,
                query,
                top_k=top_k,
                allowed_listing_ids=allowed_listing_ids,
            )
        )
    if mode == "keyword":
        if bm25_searcher is None:
            raise HTTPException(
                status_code=503,
                detail="BM25 searcher unavailable for keyword mode",
            )
        return _normalize_hits(
            _search_with_candidates(
                bm25_searcher,
                query,
                top_k=top_k,
                allowed_listing_ids=allowed_listing_ids,
            )
        )
    if mode == "hybrid":
        if semantic_searcher is None:
            raise HTTPException(
                status_code=503,
                detail="Semantic searcher unavailable for hybrid mode",
            )
        return _normalize_hits(
            hybrid_retrieve(
                query,
                top_k=top_k,
                semantic_searcher=semantic_searcher,
                bm25_searcher=bm25_searcher,
                allowed_listing_ids=allowed_listing_ids,
            )
        )
    raise HTTPException(status_code=400, detail=f"unknown mode: {mode}")


def hybrid_retrieve(
    query: str,
    *,
    top_k: int,
    semantic_searcher: Any,
    bm25_searcher: Any | None = None,
    allowed_listing_ids: set[str] | None = None,
    semantic_weight: float = HYBRID_SEMANTIC_WEIGHT,
    bm25_weight: float = HYBRID_BM25_WEIGHT,
) -> list:
    """Combine semantic + BM25 retrieval with weighted score fusion.

    The default weights were tuned offline on
    ``data/processed/relevance_pairs_labeled_auto.csv`` via a simple grid
    search over mean NDCG@5 / MAP@5. On the current labeled set, the best
    setting was heavily semantic-weighted.
    """
    semantic_hits = _search_with_candidates(
        semantic_searcher,
        query,
        top_k=top_k,
        allowed_listing_ids=allowed_listing_ids,
    )
    if bm25_searcher is None:
        return semantic_hits

    bm25_hits = _search_with_candidates(
        bm25_searcher,
        query,
        top_k=top_k,
        allowed_listing_ids=allowed_listing_ids,
    )

    sem_max = max((float(getattr(h, "score", 0.0)) for h in semantic_hits), default=0.0)
    bm_max = max((float(getattr(h, "score", 0.0)) for h in bm25_hits), default=0.0)
    sem_denom = sem_max if sem_max > 0 else 1.0
    bm_denom = bm_max if bm_max > 0 else 1.0

    merged: dict[str, dict[str, Any]] = {}

    for hit in semantic_hits:
        listing_id = str(getattr(hit, "listing_id", ""))
        merged[listing_id] = {
            "listing_id": listing_id,
            "remark": str(getattr(hit, "remark", "")),
            "score": semantic_weight * (float(getattr(hit, "score", 0.0)) / sem_denom),
        }

    for hit in bm25_hits:
        listing_id = str(getattr(hit, "listing_id", ""))
        contribution = bm25_weight * (float(getattr(hit, "score", 0.0)) / bm_denom)
        if listing_id in merged:
            merged[listing_id]["score"] += contribution
            if not merged[listing_id]["remark"]:
                merged[listing_id]["remark"] = str(getattr(hit, "remark", ""))
        else:
            merged[listing_id] = {
                "listing_id": listing_id,
                "remark": str(getattr(hit, "remark", "")),
                "score": contribution,
            }

    ranked = sorted(merged.values(), key=lambda row: row["score"], reverse=True)[:top_k]
    return ranked


# ---------------------------------------------------------------------------
# FastAPI app + middleware
# ---------------------------------------------------------------------------


API_VERSION = "1.0.0"

limiter = Limiter(key_func=get_remote_address, default_limits=["10/second"])


def _should_warmup_semantic() -> bool:
    if os.environ.get("API_WARMUP_SEMANTIC", "1").lower() in {"0", "false", "no"}:
        return False
    # Keep tests fast and independent from heavyweight FAISS/transformer setup.
    return "PYTEST_CURRENT_TEST" not in os.environ


def _warmup_components_for_startup() -> None:
    """Move semantic model cold-start cost from first search to API startup."""
    if not _should_warmup_semantic():
        return
    try:
        started = time.perf_counter()
        components.warmup_semantic()
        # Load the CSV metadata too so the first filtered query avoids disk IO.
        len(components.metadata_store())
        logger.info(
            "semantic_warmup_complete elapsed_ms=%.2f",
            (time.perf_counter() - started) * 1000.0,
        )
    except HTTPException as exc:
        logger.warning("semantic_warmup_skipped detail=%s", exc.detail)
    except Exception:
        logger.exception("semantic_warmup_failed")


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    _warmup_components_for_startup()
    yield


app = FastAPI(
    title="Real Estate NLP API",
    description=(
        "Production REST API for the NLP pipeline: semantic search, query "
        "parsing, entity extraction, summarization, and Fair Housing "
        "compliance checking."
    ),
    version=API_VERSION,
    lifespan=lifespan,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, lambda request, exc: JSONResponse(
    status_code=429,
    content={"detail": "Rate limit exceeded"},
))

# Permissive CORS so the Streamlit demo (or a static React build) can talk to
# the API when both are deployed on different Render services.
_ALLOWED_ORIGINS = os.environ.get("API_CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _ALLOWED_ORIGINS if o.strip()] or ["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next: Callable) -> Any:
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        metrics.record_request(elapsed_ms, path=request.url.path)
        logger.exception(
            "request_error method=%s path=%s client=%s elapsed_ms=%.2f",
            request.method,
            request.url.path,
            request.client.host if request.client else "unknown",
            elapsed_ms,
        )
        raise
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    metrics.record_request(elapsed_ms, path=request.url.path)
    logger.info(
        "request method=%s path=%s client=%s status=%s elapsed_ms=%.2f",
        request.method,
        request.url.path,
        request.client.host if request.client else "unknown",
        response.status_code,
        elapsed_ms,
    )
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
    return response


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse, tags=["meta"])
@limiter.limit("10/second")
async def health(request: Request) -> HealthResponse:
    """Liveness + readiness probe."""
    semantic_ready = components._searcher is not None
    return HealthResponse(
        status="ok",
        version=API_VERSION,
        semantic_search_ready=semantic_ready,
    )


@app.post("/search", response_model=SearchResponse, tags=["search"])
@limiter.limit("10/second")
async def search_listings(
    request: Request,
    payload: SearchRequest,
    parser: QueryParser = Depends(get_query_parser),
    searcher: Any = Depends(get_semantic_searcher),
    bm25_searcher: Any = Depends(get_bm25_searcher),
    metadata_store: ListingMetadataStore = Depends(get_metadata_store),
) -> SearchResponse:
    """Mode-aware search: parse the query for structured filters, then retrieve
    listings via hybrid / semantic / keyword retrieval and post-filter by
    amenity constraints. ``payload.mode`` selects the retrieval strategy."""
    metrics.record_search_mode(payload.mode)
    key = _cache_key("search", payload.model_dump())
    cached = _cache.get(key)
    if cached is not None:
        return SearchResponse(**cached, cached=True)

    filters = parser.parse(payload.query)
    allowed_listing_ids = candidate_listing_ids(filters, metadata_store)
    if allowed_listing_ids == set():
        body = {
            "query": payload.query,
            "filters": filters,
            "results": [],
            "count": 0,
            "mode": payload.mode,
            "latency_ms": 0.0,
        }
        _cache.set(key, body)
        return SearchResponse(**body, cached=False)
    started = time.perf_counter()
    try:
        raw_hits = run_search(
            payload.query,
            mode=payload.mode,
            top_k=payload.top_k,
            semantic_searcher=searcher,
            bm25_searcher=bm25_searcher,
            allowed_listing_ids=allowed_listing_ids,
        )
    except HTTPException:
        raise
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    filtered = apply_filters(raw_hits, filters)[: payload.top_k]
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    hits = [
        SearchHit(
            listing_id=str(_hit_field(r, "listing_id", "")),
            remark=str(_hit_field(r, "remark", "")),
            score=float(_hit_field(r, "score", 0.0)),
        )
        for r in filtered
    ]

    body = {
        "query": payload.query,
        "filters": filters,
        "results": [h.model_dump() for h in hits],
        "count": len(hits),
        "mode": payload.mode,
        "latency_ms": round(elapsed_ms, 3),
    }
    _cache.set(key, body)
    return SearchResponse(**body, cached=False)


@app.post("/search/bm25", response_model=RawBM25SearchResponse, tags=["search"])
@limiter.limit("10/second")
async def search_bm25_raw(
    request: Request,
    payload: CompareRequest,
    bm25_searcher: Any = Depends(get_bm25_searcher),
) -> RawBM25SearchResponse:
    """Pure BM25 keyword search for demo comparison.

    This intentionally skips the query parser, structured metadata filters,
    amenity filters, semantic search, hybrid fusion, and summarization.
    """
    metrics.record_search_mode("bm25_raw")
    key = _cache_key("search_bm25_raw", payload.model_dump())
    cached = _cache.get(key)
    if cached is not None:
        return RawBM25SearchResponse(**cached, cached=True)
    if bm25_searcher is None:
        raise HTTPException(status_code=503, detail="BM25 searcher unavailable")

    started = time.perf_counter()
    try:
        raw_hits = bm25_searcher.search(payload.query, top_k=payload.top_k)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    hits = [
        SearchHit(
            listing_id=str(_hit_field(r, "listing_id", "")),
            remark=str(_hit_field(r, "remark", "")),
            score=float(_hit_field(r, "score", 0.0)),
        )
        for r in raw_hits
    ]
    body = {
        "query": payload.query,
        "filters": {},
        "results": [h.model_dump() for h in hits],
        "count": len(hits),
        "mode": "bm25_raw",
        "latency_ms": round(elapsed_ms, 3),
    }
    _cache.set(key, body)
    return RawBM25SearchResponse(**body, cached=False)


@app.post("/search/compare", response_model=CompareResponse, tags=["search"])
@limiter.limit("5/second")
async def compare_search(
    request: Request,
    payload: CompareRequest,
    parser: QueryParser = Depends(get_query_parser),
    searcher: Any = Depends(get_semantic_searcher),
    bm25_searcher: Any = Depends(get_bm25_searcher),
    metadata_store: ListingMetadataStore = Depends(get_metadata_store),
) -> CompareResponse:
    """Run all three retrieval modes in one call and report per-mode latency
    plus pairwise overlap between top-k result sets. This powers the side-by-
    side comparison tab in the demo UI."""
    key = _cache_key("search_compare", payload.model_dump())
    cached = _cache.get(key)
    if cached is not None:
        return CompareResponse(**cached)

    filters = parser.parse(payload.query)
    allowed_listing_ids = candidate_listing_ids(filters, metadata_store)
    modes: list[CompareMode] = ["semantic", "keyword", "hybrid", "bm25_raw"]
    results: dict[str, CompareModeResult] = {}
    listing_sets: dict[str, set[str]] = {}

    for mode in modes:
        metrics.record_search_mode(mode)
        started = time.perf_counter()
        try:
            if mode == "bm25_raw":
                if bm25_searcher is None:
                    raise HTTPException(
                        status_code=503,
                        detail="BM25 searcher unavailable for raw BM25 mode",
                    )
                raw = _normalize_hits(
                    bm25_searcher.search(payload.query, top_k=payload.top_k)
                )
            else:
                raw = run_search(
                    payload.query,
                    mode=mode,
                    top_k=payload.top_k,
                    semantic_searcher=searcher,
                    bm25_searcher=bm25_searcher,
                    allowed_listing_ids=allowed_listing_ids,
                )
        except HTTPException as exc:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            results[mode] = CompareModeResult(
                mode=mode,
                results=[],
                count=0,
                latency_ms=round(elapsed_ms, 3),
                available=False,
                error=str(exc.detail),
            )
            listing_sets[mode] = set()
            continue
        if mode == "bm25_raw":
            filtered = raw[: payload.top_k]
        else:
            filtered = apply_filters(raw, filters)[: payload.top_k]
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        hits = [
            SearchHit(
                listing_id=str(_hit_field(r, "listing_id", "")),
                remark=str(_hit_field(r, "remark", "")),
                score=float(_hit_field(r, "score", 0.0)),
            )
            for r in filtered
        ]
        results[mode] = CompareModeResult(
            mode=mode,
            results=hits,
            count=len(hits),
            latency_ms=round(elapsed_ms, 3),
            available=True,
        )
        listing_sets[mode] = {h.listing_id for h in hits}

    sem = listing_sets.get("semantic", set())
    kw = listing_sets.get("keyword", set())
    hy = listing_sets.get("hybrid", set())
    overlap = CompareOverlap(
        semantic_vs_keyword=len(sem & kw),
        semantic_vs_hybrid=len(sem & hy),
        keyword_vs_hybrid=len(kw & hy),
        all_three=len(sem & kw & hy),
    )

    body = CompareResponse(
        query=payload.query,
        filters=filters,
        top_k=payload.top_k,
        modes=results,
        overlap=overlap,
    ).model_dump()
    _cache.set(key, body)
    return CompareResponse(**body)


@app.get("/listings/{listing_id}", response_model=ListingDetail, tags=["search"])
@limiter.limit("20/second")
async def get_listing_detail(
    request: Request,
    listing_id: str,
    store: ListingMetadataStore = Depends(get_metadata_store),
    summarizer: ListingSummarizer = Depends(get_summarizer),
    checker: ComplianceChecker = Depends(get_compliance_checker),
) -> ListingDetail:
    """Enrich a search hit with structured metadata (address, price, beds,
    baths) and a short extractive summary."""
    key = _cache_key("listing_detail", {"listing_id": listing_id})
    cached = _cache.get(key)
    if cached is not None:
        return ListingDetail(**cached)

    record = store.get(listing_id)
    if record is None:
        # Return a 'not found' shaped response rather than 404 so the UI can
        # gracefully render listings whose metadata was never persisted.
        body = ListingDetail(listing_id=listing_id, found=False).model_dump()
        _cache.set(key, body)
        return ListingDetail(**body)

    summary: Optional[str] = None
    compliance_ok: Optional[bool] = None
    compliance_error_count = 0
    compliance_warning_count = 0
    remarks_text = record.get("remarks_clean") or record.get("remarks") or ""
    if remarks_text:
        try:
            summary = summarizer.extractive_summary(
                {
                    "remarks": remarks_text,
                    "bedrooms": record.get("beds"),
                    "bathrooms": record.get("baths"),
                    "price": record.get("price"),
                    "city": record.get("city"),
                },
                num_sentences=2,
            )
        except Exception:  # pragma: no cover - defensive
            logger.exception("extractive summary failed for %s", listing_id)
            summary = None
        try:
            compliance_report = checker.check_listing(remarks_text)
            compliance_ok = compliance_report.compliant
            compliance_error_count = compliance_report.error_count
            compliance_warning_count = compliance_report.warning_count
        except Exception:  # pragma: no cover - defensive
            logger.exception("compliance check failed for %s", listing_id)

    body = ListingDetail(
        listing_id=str(record.get("listing_id", listing_id)),
        address=record.get("address"),
        city=record.get("city"),
        beds=record.get("beds"),
        baths=record.get("baths"),
        price=record.get("price"),
        sqft=record.get("sqft"),
        remarks=record.get("remarks"),
        summary=summary,
        compliance_ok=compliance_ok,
        compliance_error_count=compliance_error_count,
        compliance_warning_count=compliance_warning_count,
        found=True,
    ).model_dump()
    _cache.set(key, body)
    return ListingDetail(**body)


@app.post("/feedback", response_model=FeedbackResponse, tags=["meta"])
@limiter.limit("20/second")
async def submit_feedback(
    request: Request,
    payload: FeedbackRequest,
) -> FeedbackResponse:
    """Persist a user feedback event (thumbs up / down on a search result).

    Events are appended as JSONL to ``data/processed/demo_event_log.jsonl``
    and surfaced as a satisfaction proxy via ``/metrics/dashboard``.
    """
    event = {
        "ts": time.time(),
        "listing_id": payload.listing_id,
        "query": payload.query,
        "mode": payload.mode,
        "rating": payload.rating,
        "latency_ms": payload.latency_ms,
        "note": payload.note,
        "client": request.client.host if request.client else None,
    }
    total = append_feedback_event(event)
    return FeedbackResponse(
        status="recorded",
        recorded_at=event["ts"],
        total_events=total,
    )


@app.get(
    "/metrics/dashboard",
    response_model=MetricsDashboardResponse,
    tags=["meta"],
)
@limiter.limit("20/second")
async def metrics_dashboard(request: Request) -> MetricsDashboardResponse:
    """Aggregated metrics for the demo dashboard tab.

    Combines the in-process latency / volume registry with the JSONL feedback
    log and the response cache statistics.
    """
    snapshot = metrics.snapshot()
    feedback = aggregate_feedback()
    return MetricsDashboardResponse(
        query_volume_total=snapshot["query_volume_total"],
        query_volume_last_hour=snapshot["query_volume_last_hour"],
        search_requests_total=snapshot["search_requests_total"],
        feedback_events_total=feedback["feedback_events_total"],
        satisfaction_proxy=feedback["satisfaction_proxy"],
        thumbs_up=feedback["thumbs_up"],
        thumbs_down=feedback["thumbs_down"],
        thumbs_neutral=feedback["thumbs_neutral"],
        latency_ms=LatencyPercentiles(**snapshot["latency_ms"]),
        cache=_cache.stats(),
        mode_distribution=snapshot["mode_distribution"],
    )


@app.post("/parse-query", response_model=ParseQueryResponse, tags=["nlp"])
@limiter.limit("10/second")
async def parse_query(
    request: Request,
    payload: ParseQueryRequest,
    parser: QueryParser = Depends(get_query_parser),
) -> ParseQueryResponse:
    """Parse a natural-language query into structured filters and a
    parameterised SQL WHERE clause."""
    key = _cache_key("parse", payload.model_dump())
    cached = _cache.get(key)
    if cached is not None:
        return ParseQueryResponse(**cached)

    parsed = parser.parse_to_sql(payload.query)
    body = {
        "query": payload.query,
        "filters": parsed.filters,
        "where_sql": parsed.where_sql,
        "params": parsed.params,
    }
    _cache.set(key, body)
    return ParseQueryResponse(**body)


@app.post(
    "/extract-entities",
    response_model=ExtractEntitiesResponse,
    tags=["nlp"],
)
@limiter.limit("10/second")
async def extract_entities(
    request: Request,
    payload: ExtractEntitiesRequest,
    extractor: EntityExtractor = Depends(get_entity_extractor),
) -> ExtractEntitiesResponse:
    """Extract structured entities (beds, baths, price, sqft, amenities) from
    listing remark text."""
    key = _cache_key("entities", payload.model_dump())
    cached = _cache.get(key)
    if cached is not None:
        return ExtractEntitiesResponse(**cached)

    result = extractor.extract_all(payload.text)
    sqft_raw = result.get("sqft") or []
    amenities_raw = result.get("amenities") or []
    body = {
        "text": payload.text,
        "bedrooms": result.get("bedrooms"),
        "bathrooms": result.get("bathrooms"),
        "price": result.get("price"),
        "sqft": list(sqft_raw) if isinstance(sqft_raw, list) else [],
        "amenities": list(amenities_raw) if isinstance(amenities_raw, list) else [],
    }
    _cache.set(key, body)
    return ExtractEntitiesResponse(**body)


@app.post("/summarize", response_model=SummarizeResponse, tags=["nlp"])
@limiter.limit("10/second")
async def summarize(
    request: Request,
    payload: SummarizeRequest,
    summarizer: ListingSummarizer = Depends(get_summarizer),
) -> SummarizeResponse:
    """Generate a short summary for a listing. ``mode`` toggles between the
    extractive and (lightweight) abstractive rewrites."""
    key = _cache_key("summary", payload.model_dump())
    cached = _cache.get(key)
    if cached is not None:
        return SummarizeResponse(**cached)

    listing: dict[str, Any] = {
        "remarks": payload.remarks,
        "bedrooms": payload.bedrooms,
        "bathrooms": payload.bathrooms,
        "price": payload.price,
        "city": payload.city,
        "features": payload.features,
    }
    if payload.mode == "abstractive":
        summary = summarizer.abstractive_summary(listing, max_sentences=payload.num_sentences)
    else:
        summary = summarizer.extractive_summary(listing, num_sentences=payload.num_sentences)
    body = {"summary": summary, "mode": payload.mode}
    _cache.set(key, body)
    return SummarizeResponse(**body)


@app.post(
    "/check-compliance",
    response_model=ComplianceResponse,
    tags=["compliance"],
)
@limiter.limit("10/second")
async def check_compliance(
    request: Request,
    payload: ComplianceRequest,
    checker: ComplianceChecker = Depends(get_compliance_checker),
) -> ComplianceResponse:
    """Scan a listing description for Fair Housing Act compliance issues."""
    key = _cache_key("compliance", payload.model_dump())
    cached = _cache.get(key)
    if cached is not None:
        return ComplianceResponse(**cached)

    report = checker.check_listing(payload.text)
    violations = [
        ComplianceViolation(
            category=v.category,
            severity=v.severity,
            matched_text=v.matched_text,
            start=v.start,
            end=v.end,
            message=v.message,
            suggestion=v.suggestion,
        )
        for v in report.violations
    ]
    body = ComplianceResponse(
        compliant=report.compliant,
        error_count=report.error_count,
        warning_count=report.warning_count,
        info_count=report.info_count,
        violations=violations,
    ).model_dump()
    _cache.set(key, body)
    return ComplianceResponse(**body)


@app.post(
    "/submit-listing",
    response_model=SubmitListingResponse,
    tags=["compliance"],
)
@limiter.limit("10/second")
async def submit_listing(
    request: Request,
    payload: SubmitListingRequest,
    service: ListingSubmissionService = Depends(get_submission_service),
) -> SubmitListingResponse:
    """Submit a draft listing through the Fair Housing compliance workflow."""
    listing = Listing(
        listing_id=payload.listing_id,
        address=payload.address,
        price_usd=payload.price_usd,
        description=payload.description,
    )
    blocked = False
    try:
        service.submit(listing)
    except ListingSubmissionError:
        blocked = True

    report = listing.compliance_report
    if report is None:
        raise HTTPException(status_code=500, detail="compliance report missing")

    compliance = ComplianceResponse(
        compliant=report.compliant,
        error_count=report.error_count,
        warning_count=report.warning_count,
        info_count=report.info_count,
        violations=[
            ComplianceViolation(
                category=v.category,
                severity=v.severity,
                matched_text=v.matched_text,
                start=v.start,
                end=v.end,
                message=v.message,
                suggestion=v.suggestion,
            )
            for v in report.violations
        ],
    )
    status = "blocked" if blocked else listing.status
    return SubmitListingResponse(
        listing_id=listing.listing_id,
        status=status,
        compliance=compliance,
        audit_log=list(listing.audit_log),
    )


@app.get("/cache/stats", response_model=CacheStatsResponse, tags=["meta"])
@limiter.limit("10/second")
async def cache_stats(request: Request) -> CacheStatsResponse:
    """Inspect the response cache."""
    return CacheStatsResponse(**_cache.stats())


@app.delete("/cache", tags=["meta"])
@limiter.limit("10/second")
async def clear_cache(request: Request) -> JSONResponse:
    """Flush the response cache."""
    _cache.clear()
    return JSONResponse({"status": "cleared"})


__all__ = [
    "app",
    "apply_filters",
    "TTLCache",
    "components",
    "hybrid_retrieve",
    "run_search",
    "get_query_parser",
    "get_entity_extractor",
    "get_summarizer",
    "get_compliance_checker",
    "get_submission_service",
    "get_semantic_searcher",
    "get_bm25_searcher",
    "get_metadata_store",
    "limiter",
    "metrics",
    "MetricsRegistry",
    "feedback_log_path",
    "set_feedback_log_path",
    "append_feedback_event",
    "aggregate_feedback",
]
