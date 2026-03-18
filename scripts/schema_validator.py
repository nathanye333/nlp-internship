from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    errors: list[str]


class SchemaValidator:
    """
    Validate parsed query filters against basic schema/domain constraints.

    - Cities: must exist in known city set (loaded from `data/processed/listing_sample.csv` when available)
    - Ranges: enforce sensible numeric bounds and min<=max relationships
    """

    def __init__(
        self,
        valid_cities: Iterable[str] | None = None,
        *,
        sample_csv_path: str | Path = "data/processed/listing_sample.csv",
        price_min_allowed: int = 50_000,
        price_max_allowed: int = 50_000_000,
        beds_min_allowed: int = 0,
        beds_max_allowed: int = 20,
        baths_min_allowed: float = 0.0,
        baths_max_allowed: float = 20.0,
    ):
        self.price_min_allowed = int(price_min_allowed)
        self.price_max_allowed = int(price_max_allowed)
        self.beds_min_allowed = int(beds_min_allowed)
        self.beds_max_allowed = int(beds_max_allowed)
        self.baths_min_allowed = float(baths_min_allowed)
        self.baths_max_allowed = float(baths_max_allowed)

        if valid_cities is not None:
            self.valid_cities = {str(c) for c in valid_cities if str(c).strip()}
        else:
            self.valid_cities = self._load_valid_cities_from_sample(sample_csv_path)

    def validate_filters(self, filters: dict) -> ValidationResult:
        errors: list[str] = []

        # City
        city = filters.get("city")
        if city is not None:
            if not isinstance(city, str) or not city.strip():
                errors.append("City must be a non-empty string.")
            elif self.valid_cities and city not in self.valid_cities:
                errors.append(f"City '{city}' not found in known cities.")

        # Price
        self._validate_numeric_bounds(errors, filters, "price_min", self.price_min_allowed, self.price_max_allowed, int)
        self._validate_numeric_bounds(errors, filters, "price_max", self.price_min_allowed, self.price_max_allowed, int)
        self._validate_min_le_max(errors, filters, "price_min", "price_max")

        # Beds
        self._validate_numeric_bounds(errors, filters, "bedrooms_min", self.beds_min_allowed, self.beds_max_allowed, int)
        self._validate_numeric_bounds(errors, filters, "bedrooms_max", self.beds_min_allowed, self.beds_max_allowed, int)
        self._validate_min_le_max(errors, filters, "bedrooms_min", "bedrooms_max")

        # Baths
        self._validate_numeric_bounds(
            errors, filters, "bathrooms_min", self.baths_min_allowed, self.baths_max_allowed, float
        )
        self._validate_numeric_bounds(
            errors, filters, "bathrooms_max", self.baths_min_allowed, self.baths_max_allowed, float
        )
        self._validate_min_le_max(errors, filters, "bathrooms_min", "bathrooms_max")

        # Amenities lists must be lists of strings
        for key in ("amenities_in", "amenities_out"):
            if key in filters:
                v = filters[key]
                if not isinstance(v, list) or not all(isinstance(x, str) and x.strip() for x in v):
                    errors.append(f"{key} must be a list of non-empty strings.")

        return ValidationResult(ok=(len(errors) == 0), errors=errors)

    # -----------------------
    # internals
    # -----------------------

    def _load_valid_cities_from_sample(self, sample_csv_path: str | Path) -> set[str]:
        path = Path(sample_csv_path)
        if not path.exists():
            return set()
        try:
            with path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames is None:
                    return set()
                # match `scripts/data_loading.py` extract
                if "L_City" in reader.fieldnames:
                    col = "L_City"
                elif "city" in reader.fieldnames:
                    col = "city"
                else:
                    return set()
                out: set[str] = set()
                for row in reader:
                    v = (row.get(col) or "").strip()
                    if v:
                        out.add(v)
                return out
        except Exception:
            return set()

    def _validate_numeric_bounds(self, errors: list[str], filters: dict, key: str, lo, hi, caster):
        if key not in filters:
            return
        try:
            v = caster(filters[key])
        except Exception:
            errors.append(f"{key} must be a number.")
            return
        if v < lo or v > hi:
            errors.append(f"{key}={v} outside allowed range [{lo}, {hi}].")

    def _validate_min_le_max(self, errors: list[str], filters: dict, key_min: str, key_max: str):
        if key_min in filters and key_max in filters:
            try:
                if float(filters[key_min]) > float(filters[key_max]):
                    errors.append(f"{key_min} cannot be greater than {key_max}.")
            except Exception:
                # bounds validator will already surface type issues
                return