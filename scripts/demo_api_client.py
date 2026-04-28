"""HTTP client for the Week 10 NLP API used by the Week 11 demo UI.

The Streamlit app (and its tests) shouldn't sprinkle :mod:`httpx` calls
everywhere, so this module wraps each endpoint in a typed dataclass return
and consistently records the round-trip latency that the user actually
experiences (which can differ from the server-side ``latency_ms`` if the
network adds non-trivial overhead).

The client supports two transports:

* The default constructor uses :class:`httpx.Client` against a network base
  URL. Use this from the deployed Streamlit app.
* Pass any :class:`httpx.BaseTransport` (e.g. ``httpx.ASGITransport`` or the
  one wrapped by :class:`fastapi.testclient.TestClient`) to talk to an
  in-process FastAPI app from tests.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional

import httpx


DEFAULT_TIMEOUT = float(os.environ.get("DEMO_API_TIMEOUT", "30"))


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ApiResult:
    """Wraps a parsed JSON payload and round-trip latency."""

    data: dict
    status_code: int
    client_latency_ms: float

    def ok(self) -> bool:
        return 200 <= self.status_code < 300

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)


@dataclass(frozen=True)
class SearchHit:
    listing_id: str
    remark: str
    score: float


@dataclass(frozen=True)
class SearchResult:
    query: str
    mode: str
    filters: dict
    hits: list[SearchHit]
    server_latency_ms: float
    client_latency_ms: float
    cached: bool
    raw: dict = field(repr=False)


@dataclass(frozen=True)
class CompareResult:
    query: str
    filters: dict
    top_k: int
    modes: dict[str, dict]
    overlap: dict
    client_latency_ms: float
    raw: dict = field(repr=False)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class DemoApiClient:
    """Thin, well-typed wrapper around the demo's FastAPI service."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        *,
        timeout: float = DEFAULT_TIMEOUT,
        transport: Optional[httpx.BaseTransport] = None,
        client: Optional[httpx.Client] = None,
    ) -> None:
        self.base_url = (
            base_url
            or os.environ.get("API_BASE_URL")
            or "http://localhost:8000"
        ).rstrip("/")
        self.timeout = timeout
        if client is not None:
            self._client = client
            self._owns_client = False
        else:
            self._client = httpx.Client(
                base_url=self.base_url,
                timeout=timeout,
                transport=transport,
            )
            self._owns_client = True
        self._listing_cache: dict[str, dict] = {}

    # ------------------------------------------------------------------ utils

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: Optional[Mapping[str, Any]] = None,
        params: Optional[Mapping[str, Any]] = None,
    ) -> ApiResult:
        started = time.perf_counter()
        response = self._client.request(method, path, json=json, params=params)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        try:
            data = response.json()
        except ValueError:
            data = {"raw_text": response.text}
        return ApiResult(
            data=data,
            status_code=response.status_code,
            client_latency_ms=round(elapsed_ms, 3),
        )

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "DemoApiClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        self.close()

    # ------------------------------------------------------------- endpoints

    def health(self) -> ApiResult:
        return self._request("GET", "/health")

    def parse_query(self, query: str) -> ApiResult:
        return self._request("POST", "/parse-query", json={"query": query})

    def extract_entities(self, text: str) -> ApiResult:
        return self._request("POST", "/extract-entities", json={"text": text})

    def summarize(
        self,
        remarks: str,
        *,
        bedrooms: Optional[float] = None,
        bathrooms: Optional[float] = None,
        price: Optional[float] = None,
        city: Optional[str] = None,
        features: Optional[list[str]] = None,
        num_sentences: int = 2,
        mode: str = "extractive",
    ) -> ApiResult:
        payload: dict[str, Any] = {
            "remarks": remarks,
            "num_sentences": num_sentences,
            "mode": mode,
        }
        for k, v in (
            ("bedrooms", bedrooms),
            ("bathrooms", bathrooms),
            ("price", price),
            ("city", city),
            ("features", features),
        ):
            if v is not None:
                payload[k] = v
        return self._request("POST", "/summarize", json=payload)

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        mode: str = "hybrid",
    ) -> SearchResult:
        result = self._request(
            "POST",
            "/search",
            json={"query": query, "top_k": top_k, "mode": mode},
        )
        if not result.ok():
            raise ApiError(result)
        data = result.data
        hits = [
            SearchHit(
                listing_id=str(h.get("listing_id", "")),
                remark=str(h.get("remark", "")),
                score=float(h.get("score", 0.0)),
            )
            for h in data.get("results", [])
        ]
        return SearchResult(
            query=data.get("query", query),
            mode=data.get("mode", mode),
            filters=data.get("filters", {}),
            hits=hits,
            server_latency_ms=float(data.get("latency_ms", 0.0)),
            client_latency_ms=result.client_latency_ms,
            cached=bool(data.get("cached", False)),
            raw=data,
        )

    def search_bm25_raw(self, query: str, *, top_k: int = 10) -> SearchResult:
        result = self._request(
            "POST",
            "/search/bm25",
            json={"query": query, "top_k": top_k},
        )
        if not result.ok():
            raise ApiError(result)
        data = result.data
        hits = [
            SearchHit(
                listing_id=str(h.get("listing_id", "")),
                remark=str(h.get("remark", "")),
                score=float(h.get("score", 0.0)),
            )
            for h in data.get("results", [])
        ]
        return SearchResult(
            query=data.get("query", query),
            mode=data.get("mode", "bm25_raw"),
            filters=data.get("filters", {}),
            hits=hits,
            server_latency_ms=float(data.get("latency_ms", 0.0)),
            client_latency_ms=result.client_latency_ms,
            cached=bool(data.get("cached", False)),
            raw=data,
        )

    def compare(self, query: str, *, top_k: int = 10) -> CompareResult:
        result = self._request(
            "POST",
            "/search/compare",
            json={"query": query, "top_k": top_k},
        )
        if not result.ok():
            raise ApiError(result)
        data = result.data
        return CompareResult(
            query=data.get("query", query),
            filters=data.get("filters", {}),
            top_k=int(data.get("top_k", top_k)),
            modes=data.get("modes", {}),
            overlap=data.get("overlap", {}),
            client_latency_ms=result.client_latency_ms,
            raw=data,
        )

    def listing(self, listing_id: str) -> ApiResult:
        key = str(listing_id)
        cached = self._listing_cache.get(key)
        if cached is not None:
            return ApiResult(data=cached, status_code=200, client_latency_ms=0.0)
        result = self._request("GET", f"/listings/{listing_id}")
        if result.ok():
            self._listing_cache[key] = result.data
        return result

    def listings_bulk(self, listing_ids: Iterable[str]) -> dict[str, dict]:
        """Convenience helper used by the UI to enrich a result list.

        Calls :meth:`listing` sequentially. The list is small (<= 50) so
        we don't bother with concurrency here; the endpoint is cheap and
        keeping it sequential avoids tripping the per-IP rate limiter.
        """
        out: dict[str, dict] = {}
        for lid in listing_ids:
            res = self.listing(lid)
            if res.ok():
                out[str(lid)] = res.data
        return out

    def feedback(
        self,
        *,
        listing_id: str,
        query: str,
        rating: int,
        mode: str = "hybrid",
        latency_ms: float = 0.0,
        note: Optional[str] = None,
    ) -> ApiResult:
        payload: dict[str, Any] = {
            "listing_id": listing_id,
            "query": query,
            "rating": rating,
            "mode": mode,
            "latency_ms": latency_ms,
        }
        if note:
            payload["note"] = note
        return self._request("POST", "/feedback", json=payload)

    def metrics(self) -> ApiResult:
        return self._request("GET", "/metrics/dashboard")

    def cache_stats(self) -> ApiResult:
        return self._request("GET", "/cache/stats")


class ApiError(RuntimeError):
    """Raised when the API returns a non-2xx response for a typed endpoint."""

    def __init__(self, result: ApiResult) -> None:
        detail = result.data.get("detail") if isinstance(result.data, dict) else None
        super().__init__(
            f"API error {result.status_code}: {detail or result.data!r}"
        )
        self.result = result
        self.status_code = result.status_code
        self.detail = detail


__all__ = [
    "ApiError",
    "ApiResult",
    "CompareResult",
    "DemoApiClient",
    "SearchHit",
    "SearchResult",
]
