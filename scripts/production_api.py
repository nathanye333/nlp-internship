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
from collections import OrderedDict
from pathlib import Path
from threading import Lock, RLock
from typing import Any, Callable, Optional

from fastapi import Depends, FastAPI, HTTPException, Request
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
    from scripts.semantic_searcher import SearchResult, SemanticSearcher


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
        self._searcher_error: Optional[str] = None

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
                from scripts.semantic_searcher import SemanticSearcher
                searcher = SemanticSearcher()
                searcher.load(index_path, meta_path)
                self._searcher = searcher
                return searcher
            except Exception as exc:  # pragma: no cover - depends on env
                self._searcher_error = str(exc)
                logger.exception("Failed to load semantic searcher")
                raise HTTPException(
                    status_code=503,
                    detail=f"Semantic searcher unavailable: {exc}",
                ) from exc

    def reset(self) -> None:
        """Clear cached component instances. Used mainly by tests."""
        with self._lock:
            self._query_parser = None
            self._entity_extractor = None
            self._summarizer = None
            self._compliance_checker = None
            self._submission_service = None
            self._searcher = None
            self._searcher_error = None


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


# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="Free-text user query")
    top_k: int = Field(10, ge=1, le=100, description="Maximum results to return")

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
    cached: bool = False


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
# Helpers
# ---------------------------------------------------------------------------


def apply_filters(results: list, filters: dict) -> list:
    """Post-filter semantic search results using structured query filters.

    The semantic searcher only surfaces listing ids and remark text, so we
    apply lightweight text-level filters here (amenity include/exclude).
    Numeric filters (price / beds / baths / city) are returned in the
    ``filters`` field of the response for downstream SQL execution; we do not
    have the metadata loaded in-process to filter on them.
    """

    if not results:
        return results
    includes = [a.lower() for a in filters.get("amenities_in", []) or []]
    excludes = [a.lower() for a in filters.get("amenities_out", []) or []]
    if not includes and not excludes:
        return results

    filtered: list = []
    for r in results:
        remark_lower = (getattr(r, "remark", "") or "").lower()
        if includes and not all(token in remark_lower for token in includes):
            continue
        if excludes and any(token in remark_lower for token in excludes):
            continue
        filtered.append(r)
    return filtered


# ---------------------------------------------------------------------------
# FastAPI app + middleware
# ---------------------------------------------------------------------------


API_VERSION = "1.0.0"

limiter = Limiter(key_func=get_remote_address, default_limits=["10/second"])

app = FastAPI(
    title="Real Estate NLP API",
    description=(
        "Production REST API for the NLP pipeline: semantic search, query "
        "parsing, entity extraction, summarization, and Fair Housing "
        "compliance checking."
    ),
    version=API_VERSION,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, lambda request, exc: JSONResponse(
    status_code=429,
    content={"detail": "Rate limit exceeded"},
))


@app.middleware("http")
async def log_requests(request: Request, call_next: Callable) -> Any:
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        logger.exception(
            "request_error method=%s path=%s client=%s elapsed_ms=%.2f",
            request.method,
            request.url.path,
            request.client.host if request.client else "unknown",
            elapsed_ms,
        )
        raise
    elapsed_ms = (time.perf_counter() - start) * 1000.0
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
) -> SearchResponse:
    """Hybrid search: parse the query for structured filters, then retrieve
    semantically similar listings and post-filter by amenity constraints."""
    key = _cache_key("search", payload.model_dump())
    cached = _cache.get(key)
    if cached is not None:
        return SearchResponse(**cached, cached=True)

    filters = parser.parse(payload.query)
    try:
        raw_hits = searcher.search(payload.query, top_k=payload.top_k)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    filtered = apply_filters(raw_hits, filters)
    hits = [
        SearchHit(
            listing_id=str(getattr(r, "listing_id", "")),
            remark=str(getattr(r, "remark", "")),
            score=float(getattr(r, "score", 0.0)),
        )
        for r in filtered
    ]

    body = {
        "query": payload.query,
        "filters": filters,
        "results": [h.model_dump() for h in hits],
        "count": len(hits),
    }
    _cache.set(key, body)
    return SearchResponse(**body, cached=False)


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
    "get_query_parser",
    "get_entity_extractor",
    "get_summarizer",
    "get_compliance_checker",
    "get_submission_service",
    "get_semantic_searcher",
    "limiter",
]
