from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


_RE_KEYWORDS = (
    "house",
    "home",
    "bed",
    "bath",
    "property",
    "listing",
    "price",
    "sqft",
    "pool",
    "garage",
    "hoa",
    "rent",
    "buy",
    "sale",
    "condo",
)


@dataclass(frozen=True)
class AnswerabilityResult:
    answerable: bool
    reason: str
    details: list[str]


class AnswerabilityChecker:
    def __init__(self, taxonomy: dict[str, Any], schema_validator: Any, parser: Any):
        self.taxonomy = taxonomy
        self.validator = schema_validator
        self.parser = parser

    def check_pre_query(self, query: str) -> AnswerabilityResult:
        """Check answerability before executing SQL."""
        clean_query = (query or "").strip()
        if not clean_query:
            return AnswerabilityResult(
                answerable=False,
                reason="Empty query",
                details=["Please provide a real estate question with location or property criteria."],
            )

        query_lower = clean_query.lower()
        if not any(kw in query_lower for kw in _RE_KEYWORDS):
            return AnswerabilityResult(
                answerable=False,
                reason="Out of domain query",
                details=[
                    "This appears unrelated to real estate listings.",
                    "Try including terms like bed/bath, price, city, or listing features.",
                ],
            )

        filters = self.parser.parse(clean_query)
        if not filters:
            return AnswerabilityResult(
                answerable=False,
                reason="Insufficient constraints",
                details=[
                    "I could not identify searchable listing criteria from your query.",
                    "Add at least one concrete constraint such as city, budget, beds, baths, or amenities.",
                ],
            )

        if self.validator is not None:
            valid, errors = self.validator.validate_query(filters)
            if not valid:
                return AnswerabilityResult(
                    answerable=False,
                    reason="Invalid query constraints",
                    details=[str(error) for error in errors],
                )

        taxonomy_errors = self._validate_taxonomy_values(filters)
        if taxonomy_errors:
            return AnswerabilityResult(
                answerable=False,
                reason="Unsupported filter values",
                details=taxonomy_errors,
            )

        return AnswerabilityResult(answerable=True, reason="Query is answerable", details=[])

    def check_post_query(self, query: str, results_df: Any) -> AnswerabilityResult:
        """Check answerability after query execution."""
        if results_df is None:
            return AnswerabilityResult(
                answerable=False,
                reason="No results object returned",
                details=["The query did not return a valid result set."],
            )

        if len(results_df) == 0:
            return AnswerabilityResult(
                answerable=False,
                reason="No listings match criteria",
                details=["Try widening price range, location radius, or feature constraints."],
            )

        if hasattr(results_df, "isnull") and results_df.isnull().all().all():
            return AnswerabilityResult(
                answerable=False,
                reason="Results contain no meaningful values",
                details=["The matched rows are empty; try changing requested attributes."],
            )

        return AnswerabilityResult(answerable=True, reason="Results found", details=[])

    def _validate_taxonomy_values(self, filters: dict[str, Any]) -> list[str]:
        errors: list[str] = []
        taxonomy_cities = self._extract_city_set(self.taxonomy)
        if "city" in filters and taxonomy_cities:
            city = str(filters["city"]).strip().lower()
            if city and city not in taxonomy_cities:
                errors.append(f"City '{filters['city']}' is not present in the supported taxonomy.")

        # Guard common obviously invalid bedroom/bathroom ranges.
        if "bedrooms_min" in filters and "bedrooms_max" in filters:
            if float(filters["bedrooms_min"]) > float(filters["bedrooms_max"]):
                errors.append("Bedroom minimum is greater than bedroom maximum.")
        if "bathrooms_min" in filters and "bathrooms_max" in filters:
            if float(filters["bathrooms_min"]) > float(filters["bathrooms_max"]):
                errors.append("Bathroom minimum is greater than bathroom maximum.")

        return errors

    def _extract_city_set(self, taxonomy: Any) -> set[str]:
        cities: set[str] = set()
        if isinstance(taxonomy, dict):
            for key, value in taxonomy.items():
                if re.search(r"city|cities|location", str(key), flags=re.I):
                    if isinstance(value, list):
                        for item in value:
                            cities.add(str(item).strip().lower())
                    elif isinstance(value, dict):
                        cities.update(str(k).strip().lower() for k in value.keys())
                if isinstance(value, (dict, list)):
                    cities.update(self._extract_city_set(value))
        elif isinstance(taxonomy, list):
            for item in taxonomy:
                cities.update(self._extract_city_set(item))
        return {city for city in cities if city}