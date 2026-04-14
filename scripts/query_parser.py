from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


_WS_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class ParsedQuery:
    """
    Result of parsing a natural language query.

    - `filters` is a structured, JSON-serializable dict.
    - `where_sql` is a parameterized SQL WHERE clause (no leading "WHERE" if empty).
    - `params` are positional parameters in the same order as placeholders.
    """

    filters: dict
    where_sql: str
    params: list


class QueryParser:
    """
    Parse natural-language real estate queries into structured filters and
    parameterized SQL WHERE clauses.

    This module intentionally avoids string concatenation with user input; all
    user values flow through SQL parameters.
    """

    # Column mapping used by this repository's MySQL extract (see `scripts/data_loading.py`)
    COL_PRICE = "L_SystemPrice"
    COL_CITY = "L_City"
    COL_BEDS = "L_Keyword2"
    COL_BATHS = "LM_Dec_3"
    COL_REMARKS = "L_Remarks"

    # Common amenity keywords -> (SQL snippet, param builder)
    # We keep it simple: search in remarks with LIKE; safe via parameters.
    AMENITY_ALIASES: dict[str, list[str]] = {
        "pool": ["pool", "swimming pool"],
        "garage": ["garage"],
        "fireplace": ["fireplace"],
        "air conditioning": ["air conditioning", "a/c", "ac"],
        "hardwood": ["hardwood"],
        "basement": ["basement"],
        "waterfront": ["waterfront"],
        "washer dryer": ["washer dryer", "washer", "dryer"],
        "stainless steel": ["stainless steel"],
        "granite": ["granite"],
        "hoa": ["homeowners association", "hoa"],
        "new construction": ["new construction", "newly built"],
    }

    def parse(self, query: str) -> dict:
        """
        Parse `query` into a structured filter dict.

        Keys used:
        - price_min, price_max
        - bedrooms_min, bedrooms_max
        - bathrooms_min, bathrooms_max
        - city
        - amenities_in (list[str]), amenities_out (list[str])
        """
        if query is None:
            return {}

        q = self._normalize_query(query)
        filters: dict = {}

        self._parse_price(q, filters)
        self._parse_beds(q, filters)
        self._parse_baths(q, filters)
        self._parse_city(q, filters)
        self._parse_amenities(q, filters)

        return filters

    def parse_to_sql(self, query: str) -> ParsedQuery:
        filters = self.parse(query)
        where_sql, params = self.to_where_sql(filters)
        return ParsedQuery(filters=filters, where_sql=where_sql, params=params)

    def parse_with_intent(self, query: str, intent_classifier) -> dict:
        """
        Parse query and enrich output with buyer intent signals.
        """
        parsed = self.parse_to_sql(query)
        prediction = intent_classifier.predict(query)
        return {
            "filters": parsed.filters,
            "where_sql": parsed.where_sql,
            "params": parsed.params,
            "intent": prediction.intent,
            "intent_confidence": prediction.confidence,
            "intent_uncertain": prediction.is_uncertain,
            "intent_probabilities": prediction.probabilities,
        }

    def to_where_sql(self, filters: dict) -> tuple[str, list]:
        """
        Convert filters into a safe parameterized WHERE clause.

        Returns (where_sql, params) where `where_sql` does NOT include leading "WHERE".
        """
        conditions: list[str] = []
        params: list = []

        # Price
        if "price_min" in filters:
            conditions.append(f"{self.COL_PRICE} >= %s")
            params.append(int(filters["price_min"]))
        if "price_max" in filters:
            conditions.append(f"{self.COL_PRICE} <= %s")
            params.append(int(filters["price_max"]))

        # Beds / baths
        if "bedrooms_min" in filters:
            conditions.append(f"{self.COL_BEDS} >= %s")
            params.append(int(filters["bedrooms_min"]))
        if "bedrooms_max" in filters:
            conditions.append(f"{self.COL_BEDS} <= %s")
            params.append(int(filters["bedrooms_max"]))

        if "bathrooms_min" in filters:
            conditions.append(f"{self.COL_BATHS} >= %s")
            params.append(float(filters["bathrooms_min"]))
        if "bathrooms_max" in filters:
            conditions.append(f"{self.COL_BATHS} <= %s")
            params.append(float(filters["bathrooms_max"]))

        # City
        if "city" in filters:
            conditions.append(f"{self.COL_CITY} = %s")
            params.append(filters["city"])

        # Amenities include/exclude (LIKE on remarks)
        for a in filters.get("amenities_in", []) or []:
            conditions.append(f"LOWER({self.COL_REMARKS}) LIKE %s")
            params.append(f"%{a.lower()}%")
        for a in filters.get("amenities_out", []) or []:
            conditions.append(f"(LOWER({self.COL_REMARKS}) NOT LIKE %s OR {self.COL_REMARKS} IS NULL)")
            params.append(f"%{a.lower()}%")

        return " AND ".join(conditions), params

    # -----------------------
    # Parsing helpers
    # -----------------------

    def _normalize_query(self, query: str) -> str:
        q = query.strip()
        q = q.replace(",", "")
        q = _WS_RE.sub(" ", q)
        return q

    def _parse_compact_number(self, s: str) -> int:
        """
        Parse numeric tokens like: 700000, 700k, 1.2m, $700k.
        """
        raw = s.strip().lower().replace("$", "")
        m = re.fullmatch(r"(\d+(?:\.\d+)?)([km])?", raw)
        if not m:
            # best effort: take digits only
            digits = re.sub(r"[^\d]", "", raw)
            return int(digits) if digits else 0
        val = float(m.group(1))
        suf = m.group(2)
        if suf == "k":
            val *= 1_000
        elif suf == "m":
            val *= 1_000_000
        return int(val)

    def _set_min_max(self, filters: dict, key_min: str, key_max: str, lo, hi) -> None:
        if lo is not None:
            filters[key_min] = lo
        if hi is not None:
            filters[key_max] = hi

    def _parse_price(self, q: str, filters: dict) -> None:
        # between / from-to
        m = re.search(
            r"(?:between|from)\s+\$?(\d+(?:\.\d+)?[km]?)\s+(?:and|to|-)\s+\$?(\d+(?:\.\d+)?[km]?)",
            q,
            flags=re.I,
        )
        if m:
            lo = self._parse_compact_number(m.group(1))
            hi = self._parse_compact_number(m.group(2))
            if lo and hi:
                self._set_min_max(filters, "price_min", "price_max", min(lo, hi), max(lo, hi))
                return

        # under / max
        best_max = None
        for m in re.finditer(
            r"(?:under|below|less than|<|at most|no more than)\s+(\$?\d+(?:\.\d+)?[km]?)",
            q,
            flags=re.I,
        ):
            raw = m.group(1)
            val = self._parse_compact_number(raw)
            if ("$" in raw) or re.search(r"[km]\b", raw, flags=re.I) or val >= 10_000:
                best_max = val
        if best_max is not None:
            filters["price_max"] = best_max

        # over / min
        best_min = None
        for m in re.finditer(
            r"(?:over|above|more than|>|at least|minimum|min)\s+(\$?\d+(?:\.\d+)?[km]?)",
            q,
            flags=re.I,
        ):
            raw = m.group(1)
            val = self._parse_compact_number(raw)
            if ("$" in raw) or re.search(r"[km]\b", raw, flags=re.I) or val >= 10_000:
                best_min = val
        if best_min is not None:
            filters["price_min"] = best_min

        # bare "$700k" / "700k" hints when paired with "budget"/"price"
        m = re.search(r"(?:budget|price)\s*(?:of|:)?\s*\$?(\d+(?:\.\d+)?[km]?)", q, flags=re.I)
        if m and "price_min" not in filters and "price_max" not in filters:
            filters["price_max"] = self._parse_compact_number(m.group(1))

    def _parse_beds(self, q: str, filters: dict) -> None:
        if re.search(r"\bstudio\b", q, flags=re.I):
            # Treat studio as 0 bedrooms.
            filters["bedrooms_max"] = 0
            return

        # "3-4 bed", "3 to 4 bedrooms"
        m = re.search(r"\b(\d+)\s*(?:-|to|–)\s*(\d+)\s*(?:bed|beds|bedroom|bedrooms|br)\b", q, flags=re.I)
        if m:
            lo, hi = int(m.group(1)), int(m.group(2))
            self._set_min_max(filters, "bedrooms_min", "bedrooms_max", min(lo, hi), max(lo, hi))
            return

        # "3+ bed", "at least 3 bedrooms"
        m = re.search(r"(?:at least|min(?:imum)?)\s*(\d+)\s*(?:bed|beds|bedroom|bedrooms|br)\b", q, flags=re.I)
        if m:
            filters["bedrooms_min"] = int(m.group(1))
            return
        m = re.search(r"\b(\d+)\s*\+\s*(?:bed|beds|bedroom|bedrooms|br)\b", q, flags=re.I)
        if m:
            filters["bedrooms_min"] = int(m.group(1))
            return

        # exact "3 bed", "3br"
        m = re.search(r"\b(\d+)\s*(?:bed|beds|bedroom|bedrooms|br)\b", q, flags=re.I)
        if m:
            n = int(m.group(1))
            filters["bedrooms_min"] = n
            filters["bedrooms_max"] = n

    def _parse_baths(self, q: str, filters: dict) -> None:
        # "2-3 bath", "2 to 3 bathrooms"
        m = re.search(r"\b(\d+(?:\.\d+)?)\s*(?:-|to|–)\s*(\d+(?:\.\d+)?)\s*(?:bath|baths|bathroom|bathrooms|ba)\b", q, flags=re.I)
        if m:
            lo, hi = float(m.group(1)), float(m.group(2))
            self._set_min_max(filters, "bathrooms_min", "bathrooms_max", min(lo, hi), max(lo, hi))
            return

        m = re.search(r"(?:at least|min(?:imum)?)\s*(\d+(?:\.\d+)?)\s*(?:bath|baths|bathroom|bathrooms|ba)\b", q, flags=re.I)
        if m:
            filters["bathrooms_min"] = float(m.group(1))
            return

        m = re.search(r"\b(\d+(?:\.\d+)?)\s*(?:bath|baths|bathroom|bathrooms|ba)\b", q, flags=re.I)
        if m:
            n = float(m.group(1))
            filters["bathrooms_min"] = n
            filters["bathrooms_max"] = n

    def _parse_city(self, q: str, filters: dict) -> None:
        # "in Irvine", "near San Diego", "around Los Angeles"
        m = re.search(r"\b(?:in|near|around|within)\s+([a-z][a-z\s\.\-']{1,50})\b", q, flags=re.I)
        if not m:
            return
        raw = m.group(1).strip()
        # stop at known trailing phrases
        raw = re.split(r"\b(?:under|below|over|above|with|without|no|not|between|from)\b", raw, maxsplit=1, flags=re.I)[0].strip()
        city = self._normalize_city(raw)
        if city:
            filters["city"] = city

    def _normalize_city(self, raw: str) -> str | None:
        # Reject strings with obvious SQL / control characters
        if re.search(r"[;`]|--|\b(or|and)\b\s+\d", raw, flags=re.I):
            return None
        cleaned = re.sub(r"[^a-zA-Z\s\.\-']", "", raw).strip()
        cleaned = _WS_RE.sub(" ", cleaned)
        if not cleaned:
            return None
        # Title-case words, but keep short tokens upper (e.g. "LA")
        parts = []
        for w in cleaned.split(" "):
            if len(w) <= 2 and w.isalpha():
                parts.append(w.upper())
            else:
                parts.append(w[:1].upper() + w[1:].lower())
        return " ".join(parts)

    def _parse_amenities(self, q: str, filters: dict) -> None:
        inc: set[str] = set()
        out: set[str] = set()

        lower = q.lower()
        for canonical, aliases in self.AMENITY_ALIASES.items():
            # negation patterns
            neg_re = r"(?:no|not|without|w/o)\s+(?:a\s+|an\s+)?(" + "|".join(re.escape(a) for a in sorted(aliases, key=len, reverse=True)) + r")\b"
            if re.search(neg_re, lower, flags=re.I):
                out.add(canonical)
                continue

            pos_re = r"(?:with|w/)?\s*\b(" + "|".join(re.escape(a) for a in sorted(aliases, key=len, reverse=True)) + r")\b"
            if re.search(pos_re, lower, flags=re.I):
                inc.add(canonical)

        if inc:
            filters["amenities_in"] = sorted(inc)
        if out:
            filters["amenities_out"] = sorted(out)

    # Small utility for tests / advanced callers
    def supported_amenities(self) -> Iterable[str]:
        return sorted(self.AMENITY_ALIASES.keys())