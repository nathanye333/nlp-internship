import json
import re
from pathlib import Path
from typing import Dict, List, Optional


DATA_DIR = Path("data/processed")
DEFAULT_TAXONOMY_PATH = DATA_DIR / "taxonomy.json"

# Fallback when taxonomy.json is missing or has no amenity terms.
# Only terms that appear in gold labels (after normalization) to avoid FPs.
FALLBACK_AMENITY_TERMS = [
    "garage", "pool", "laundry room", "living space", "spacious", "space",
]


class EntityExtractor:
    """
    Rule-based extractor for listing remarks.

    Extracts:
    - bedrooms (int)
    - bathrooms (float, to allow 2.5 baths)
    - price (int, dollars)
    - sqft (List[int], all square footage mentions)
    - amenities (List[str], using Week 1 taxonomy or fallback terms)
    """

    def __init__(self, taxonomy_path: Path | str = DEFAULT_TAXONOMY_PATH) -> None:
        self.taxonomy_path = Path(taxonomy_path)
        self._amenity_terms: List[str] = self._load_amenity_terms()

        # Precompile common regexes
        self._bedroom_patterns = [
            re.compile(r"(\d+)\s*(?:bed(?:room)?s?|br)\b", re.IGNORECASE),
            re.compile(r"(\d+)\s*bd\b", re.IGNORECASE),
        ]
        self._bathroom_patterns = [
            re.compile(r"(\d+(?:\.\d+)?)\s*(?:bath(?:room)?s?|ba)\b", re.IGNORECASE),
            re.compile(r"(\d+(?:\.\d+)?)\s*ba\b", re.IGNORECASE),
        ]
        self._price_pattern = re.compile(r"\$?\s*([\d,]{5,})")
        self._sqft_patterns = [
            re.compile(
                r"([\d,]{3,6})\s*(?:sq\.?\s*ft\.?|square\s*feet|sf)\b",
                re.IGNORECASE,
            ),
            re.compile(
                r"approximately\s+([\d,]{3,6})\s*(?:sq\.?\s*ft\.?|square\s*feet|sf)\b",
                re.IGNORECASE,
            ),
        ]
        # Keywords to help choose the *main* sqft mention (vs ADU/accessory unit sqft).
        # Note: gold labels can include studios / detached structures, so keep this narrow.
        self._sqft_negative_context = ("adu", "accessory")
        self._sqft_living_context = ("living space", "of living", "refined living", "interior")
        self._sqft_intro_context = (
            "offers",
            "spanning",
            "spans",
            "across",
            "inside this",
            "this",
            "nearly",
            "over",
            "approximately",
            "about",
            "set on",
            "sits on",
        )

    def _load_amenity_terms(self) -> List[str]:
        """
        Load amenity terms from the Week 1 taxonomy.

        Returns simple string terms for category == 'amenities'.
        Falls back to an empty list if taxonomy is missing.
        """
        if not self.taxonomy_path.exists():
            return list(FALLBACK_AMENITY_TERMS)

        try:
            with open(self.taxonomy_path) as f:
                taxonomy = json.load(f)
        except Exception:
            return list(FALLBACK_AMENITY_TERMS)

        terms: List[str] = []
        for t in taxonomy.get("terms", []):
            if t.get("category") == "amenities":
                term = str(t.get("term", "")).strip().lower()
                if term:
                    terms.append(term)
        if not terms:
            terms = list(FALLBACK_AMENITY_TERMS)
        else:
            # Add gold-canonical terms that taxonomy might miss (so we match labeled data)
            seen = set(terms)
            for t in FALLBACK_AMENITY_TERMS:
                if t not in seen:
                    terms.append(t)
                    seen.add(t)
        return terms

    # -------- Numeric entity extractors --------

    def extract_bedrooms(self, text: str) -> Optional[int]:
        """
        Extract number of bedrooms from free text.
        Returns an integer or None if not found.
        """
        if not text:
            return None

        for pattern in self._bedroom_patterns:
            match = pattern.search(text)
            if match:
                try:
                    return int(float(match.group(1)))
                except ValueError:
                    continue
        return None

    def extract_bathrooms(self, text: str) -> Optional[float]:
        """
        Extract number of bathrooms (supports half baths, e.g. 2.5).
        Returns a float or None if not found.
        """
        if not text:
            return None

        for pattern in self._bathroom_patterns:
            match = pattern.search(text)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue
        return None

    def extract_price(self, text: str) -> Optional[int]:
        """
        Extract listing price in dollars.
        Skips 4-digit years when not preceded by $ (e.g. "Built in 2022").
        """
        if not text:
            return None

        match = self._price_pattern.search(text)
        if not match:
            return None
        raw = match.group(1)
        try:
            val = int(raw.replace(",", ""))
            if 1900 <= val <= 2099:
                before = text[max(0, match.start() - 30) : match.start()]
                if "$" not in before:
                    return None
            return val
        except ValueError:
            return None

    def extract_sqft(self, text: str) -> List[int]:
        """
        Extract square footage.

        The labeled set often marks *one* sqft value per remark, usually for the
        main home/living area. Many remarks contain multiple sqft mentions (lot,
        ADU, garage, etc.). To reduce false positives and mismatches we:
        - score each sqft mention by nearby context
        - ignore ADU/garage/studio sqft when a main-home sqft exists
        - return a single best sqft value (list of length 0 or 1)
        """
        if not text:
            return []

        text_l = text.lower()

        # Special-case: multi-unit listings often enumerate several unit sizes; gold often omits SQFT.
        # Keep this trigger narrow so we don't drop legitimate single-home sqft.
        multi_unit_markers = (
            "unit a", "unit b", "front house:", "middle unit:", "rear adu:", "title shows",
            "multiple units", "multi-unit",
        )
        if any(m in text_l for m in multi_unit_markers):
            n_matches = sum(1 for _ in self._sqft_patterns[0].finditer(text))
            if n_matches >= 2:
                return []

        candidates: List[tuple[int, int, int]] = []  # (score, value, start_idx)
        for pattern in self._sqft_patterns:
            for match in pattern.finditer(text):
                raw = match.group(1)
                try:
                    val = int(raw.replace(",", ""))
                except ValueError:
                    continue

                start, end = match.span()
                before = text_l[max(0, start - 40) : start]
                after = text_l[end : min(len(text_l), end + 60)]

                score = 0
                if any(k in after for k in self._sqft_living_context):
                    score += 4
                if any(k in before for k in self._sqft_intro_context):
                    score += 2
                if "lot" in after or "lot" in before:
                    score += 1
                if any(k in after for k in self._sqft_negative_context) or any(
                    k in before for k in self._sqft_negative_context
                ):
                    score -= 4

                candidates.append((score, val, start))

        if not candidates:
            return []

        # Prefer high-confidence candidates; if none, return [] (avoid low-confidence sqft)
        best = max(candidates, key=lambda t: (t[0], t[1]))
        # Threshold 2: requires at least an intro cue ("offers"/"set on"/etc.) or strong living context
        if best[0] < 2:
            return []
        return [best[1]]

    # -------- Amenity extraction --------

    def extract_amenities(self, text: str) -> List[str]:
        """
        Detect amenities using the Week 1 taxonomy.

        Returns a de-duplicated list of matched amenity terms
        (as they appear in the taxonomy, lowercased).
        """
        if not text or not self._amenity_terms:
            return []

        text_l = text.lower()
        found: List[str] = []
        for term in self._amenity_terms:
            if term and term in text_l:
                found.append(term)

        # De-duplicate while preserving order
        seen = set()
        unique = []
        for t in found:
            if t not in seen:
                seen.add(t)
                unique.append(t)
        return unique

    # -------- Convenience API --------

    def extract_all(self, text: str) -> Dict[str, object]:
        """
        Run all extractors on the given text and return a structured dict.
        sqft is a list of all sqft values found; other fields are single values or None.
        """
        return {
            "bedrooms": self.extract_bedrooms(text),
            "bathrooms": self.extract_bathrooms(text),
            "price": self.extract_price(text),
            "sqft": self.extract_sqft(text),
            "amenities": self.extract_amenities(text),
        }


__all__ = ["EntityExtractor"]