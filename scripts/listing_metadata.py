"""Lightweight in-memory store for listing metadata used by the demo.

The Week 5+ semantic index only retains ``listing_id`` and ``remark`` text, so
the demo UI needs a separate source of truth for structured fields like
address, city, beds, baths, and price. This module loads those fields from
:file:`data/processed/listing_sample_cleaned.csv` once on first access.

Usage::

    store = ListingMetadataStore()
    record = store.get("1151896186")
    # -> {"listing_id": "1151896186", "address": ..., "price": 449000, ...}

A module-level :func:`default_store` singleton is provided for convenience so
that repeated calls (across the API and tests) reuse the same in-memory dict.
"""

from __future__ import annotations

import csv
from pathlib import Path
from threading import Lock
from typing import Iterable, Optional


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CSV = _PROJECT_ROOT / "data" / "processed" / "listing_sample_cleaned.csv"


def _coerce_int(value: str | None) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _coerce_float(value: str | None) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class ListingMetadataStore:
    """In-memory listing metadata indexed by ``L_ListingID`` (string)."""

    def __init__(self, csv_path: Path | str | None = None) -> None:
        self.csv_path = Path(csv_path) if csv_path is not None else _DEFAULT_CSV
        self._records: dict[str, dict] | None = None
        self._lock = Lock()

    def _load(self) -> dict[str, dict]:
        if self._records is not None:
            return self._records
        with self._lock:
            if self._records is not None:
                return self._records
            records: dict[str, dict] = {}
            if not self.csv_path.exists():
                # Allow degraded mode: tests / deployments without the CSV
                # still get an empty store rather than a hard failure.
                self._records = records
                return records
            with self.csv_path.open(newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    listing_id = (row.get("L_ListingID") or "").strip()
                    if not listing_id:
                        continue
                    records[listing_id] = {
                        "listing_id": listing_id,
                        "address": (row.get("L_Address") or "").strip() or None,
                        "city": (row.get("L_City") or "").strip() or None,
                        "beds": _coerce_float(row.get("beds")),
                        "baths": _coerce_float(row.get("baths")),
                        "price": _coerce_int(row.get("price")),
                        "remarks": (row.get("remarks") or "").strip() or None,
                        "remarks_clean": (row.get("remarks_clean") or "").strip()
                        or None,
                    }
            self._records = records
            return records

    def get(self, listing_id: str) -> Optional[dict]:
        """Return the record for ``listing_id`` or ``None`` if unknown."""
        records = self._load()
        return records.get(str(listing_id))

    def get_many(self, listing_ids: Iterable[str]) -> dict[str, dict]:
        """Bulk lookup; missing ids are simply omitted from the result."""
        records = self._load()
        out: dict[str, dict] = {}
        for lid in listing_ids:
            key = str(lid)
            if key in records:
                out[key] = records[key]
        return out

    def all_records(self) -> dict[str, dict]:
        """Return the full listing metadata mapping."""
        return self._load()

    def __len__(self) -> int:
        return len(self._load())

    def __contains__(self, listing_id: object) -> bool:
        return str(listing_id) in self._load()

    def reset(self) -> None:
        """Drop the cached records (mainly for tests)."""
        with self._lock:
            self._records = None


_default_store: ListingMetadataStore | None = None
_default_store_lock = Lock()


def default_store() -> ListingMetadataStore:
    """Return the process-wide default :class:`ListingMetadataStore`."""
    global _default_store
    if _default_store is None:
        with _default_store_lock:
            if _default_store is None:
                _default_store = ListingMetadataStore()
    return _default_store


def reset_default_store() -> None:
    """Force re-initialisation of the module-level singleton (for tests)."""
    global _default_store
    with _default_store_lock:
        _default_store = None


__all__ = [
    "ListingMetadataStore",
    "default_store",
    "reset_default_store",
]
