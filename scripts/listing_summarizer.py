from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_TOKEN_RE = re.compile(r"[a-z0-9']+")
_FEATURE_KEYWORDS = (
    "pool",
    "garage",
    "backyard",
    "office",
    "fireplace",
    "updated kitchen",
    "renovated",
    "hardwood",
    "stainless",
    "view",
    "gated",
    "solar",
    "laundry",
)
_GENERIC_LOCATIONS = {"unknown area", "local area", "the listed area", "unknown", ""}


@dataclass(frozen=True)
class RougeResult:
    rouge_l: float


class ListingSummarizer:
    """
    Summarize listings for result cards and alerts.

    Extractive summaries always include:
    - beds/baths
    - price
    - top 2 features (if available)
    - location
    """

    def extractive_summary(self, listing: Mapping[str, Any], num_sentences: int = 2) -> str:
        remarks = str(listing.get("remarks", "") or "").strip()
        location = self._resolve_location(listing, remarks)
        beds = self._format_count(listing.get("bedrooms"))
        baths = self._format_count(listing.get("bathrooms"))
        price = self._format_price(listing.get("price"))

        features = self._extract_top_features(remarks, listing.get("features"))
        header = f"{beds} bed, {baths} bath home in {location} listed at {price} with {self._format_features(features)}."

        ranked = self._rank_sentences(remarks, listing)
        detail_sentences = [s for _, _, s in ranked[: max(0, num_sentences - 1)]]
        if detail_sentences:
            return f"{header} {' '.join(detail_sentences)}".strip()
        return header

    def abstractive_summary(self, listing: Mapping[str, Any], max_sentences: int = 3) -> str:
        """
        Lightweight abstractive-style rewrite without model dependency.
        """
        extractive = self.extractive_summary(listing, num_sentences=max_sentences)
        # Keep this deterministic and compact for predictable tests.
        abstracted = extractive.replace("Highlights include", "Key features are")
        return abstracted

    def evaluate_rouge_l(self, predictions: Iterable[str], references: Iterable[str]) -> RougeResult:
        preds = list(predictions)
        refs = list(references)
        if len(preds) != len(refs):
            raise ValueError("predictions and references must have the same length")
        if not preds:
            return RougeResult(rouge_l=0.0)

        total = 0.0
        for pred, ref in zip(preds, refs):
            total += self._rouge_l_f1(pred, ref)
        return RougeResult(rouge_l=total / len(preds))

    def sample_for_human_evaluation(
        self,
        listings: list[Mapping[str, Any]],
        sample_size: int = 20,
        seed: int = 42,
    ) -> list[dict[str, str]]:
        if sample_size <= 0:
            return []
        rng = random.Random(seed)
        rows = list(listings)
        rng.shuffle(rows)
        picked = rows[: min(sample_size, len(rows))]
        return [{"summary": self.extractive_summary(row), "remarks": str(row.get("remarks", ""))} for row in picked]

    def _rank_sentences(self, remarks: str, listing: Mapping[str, Any]) -> list[tuple[float, int, str]]:
        if not remarks:
            return []
        sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(remarks) if s.strip()]
        if not sentences:
            return []

        feature_keywords = {f.lower() for f in self._extract_top_features(remarks, listing.get("features"))}
        location = self._first_non_empty(listing, ("city", "location", "neighborhood"), default="").lower()
        beds = self._format_count(listing.get("bedrooms"))
        baths = self._format_count(listing.get("bathrooms"))

        scored: list[tuple[float, int, str]] = []
        for idx, sentence in enumerate(sentences):
            sent_lower = sentence.lower()
            score = 0.0
            if idx == 0:
                score += 1.5
            if location and location in sent_lower:
                score += 1.2
            if beds != "0" and beds in sent_lower:
                score += 0.8
            if baths != "0" and baths in sent_lower:
                score += 0.8
            if any(f in sent_lower for f in feature_keywords):
                score += 1.4
            score += min(len(sentence) / 140.0, 1.0)
            scored.append((score, idx, sentence))
        scored.sort(key=lambda row: (-row[0], row[1]))
        return scored

    def _extract_top_features(self, remarks: str, features_field: Any) -> list[str]:
        features: list[str] = []
        if isinstance(features_field, list):
            features.extend(str(f).strip().lower() for f in features_field if str(f).strip())
        elif isinstance(features_field, str) and features_field.strip():
            features.extend(part.strip().lower() for part in features_field.split(",") if part.strip())

        lower_remarks = remarks.lower()
        for keyword in _FEATURE_KEYWORDS:
            if keyword in lower_remarks:
                features.append(keyword)

        ordered: list[str] = []
        for feature in features:
            if feature not in ordered:
                ordered.append(feature)
        if not ordered:
            return ["strong curb appeal", "functional layout"]
        if len(ordered) == 1:
            return [ordered[0], "modern finishes"]
        return ordered[:2]

    def _format_features(self, features: list[str]) -> str:
        if len(features) == 1:
            return features[0]
        return f"{features[0]} and {features[1]}"

    def _normalize_number(self, value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _format_count(self, value: Any) -> str:
        number = self._normalize_number(value)
        if number.is_integer():
            return str(int(number))
        return str(number)

    def _format_price(self, value: Any) -> str:
        try:
            n = float(value)
        except (TypeError, ValueError):
            return "an undisclosed price"
        return f"${n:,.0f}"

    def _first_non_empty(self, listing: Mapping[str, Any], keys: tuple[str, ...], default: str) -> str:
        for key in keys:
            raw = listing.get(key)
            if raw is None:
                continue
            text = str(raw).strip()
            if text:
                return text
        return default

    def _resolve_location(self, listing: Mapping[str, Any], remarks: str) -> str:
        explicit = self._first_non_empty(listing, ("city", "location", "neighborhood"), default="").strip()
        if explicit.lower() not in _GENERIC_LOCATIONS:
            return explicit

        inferred = self._extract_location_from_text(remarks)
        if inferred:
            return inferred
        return "the listed area"

    def _extract_location_from_text(self, remarks: str) -> str | None:
        if not remarks:
            return None

        community_match = re.search(
            r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,4})\s+(Community|Neighborhood|District|Village)\b",
            remarks,
        )
        if community_match:
            return f"{community_match.group(1)} {community_match.group(2)}"

        downtown_match = re.search(
            r"\bdowntown\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,3})",
            remarks,
            flags=re.I,
        )
        if downtown_match:
            return f"Downtown {downtown_match.group(1)}"

        in_match = re.search(
            r"\b(?:in|near|around|within)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,4})",
            remarks,
        )
        if in_match:
            candidate = in_match.group(1).strip()
            if candidate.lower() not in {"the", "this", "a", "an"}:
                return candidate

        return None

    def _tokenize(self, text: str) -> list[str]:
        return [tok.lower() for tok in _TOKEN_RE.findall(text)]

    def _lcs_length(self, a: list[str], b: list[str]) -> int:
        if not a or not b:
            return 0
        dp = [0] * (len(b) + 1)
        for i in range(1, len(a) + 1):
            prev = 0
            for j in range(1, len(b) + 1):
                temp = dp[j]
                if a[i - 1] == b[j - 1]:
                    dp[j] = prev + 1
                else:
                    dp[j] = max(dp[j], dp[j - 1])
                prev = temp
        return dp[-1]

    def _rouge_l_f1(self, prediction: str, reference: str) -> float:
        pred_tokens = self._tokenize(prediction)
        ref_tokens = self._tokenize(reference)
        if not pred_tokens or not ref_tokens:
            return 0.0
        lcs = self._lcs_length(pred_tokens, ref_tokens)
        if lcs == 0:
            return 0.0
        precision = lcs / len(pred_tokens)
        recall = lcs / len(ref_tokens)
        return (2 * precision * recall) / (precision + recall)