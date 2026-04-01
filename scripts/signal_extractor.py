from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mysql.connector
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.entity_extractor import DEFAULT_TAXONOMY_PATH, EntityExtractor


@dataclass(frozen=True)
class ExtractionAccuracy:
    metric: str
    accuracy: float
    numerator: int
    denominator: int


def _default_path(relative_path: str) -> Path:
    return Path(__file__).resolve().parents[1] / relative_path


class SignalExtractor:
    """
    Week 6 signal extractor for listing records.

    Combines:
    - Week 3 structured entities via EntityExtractor
    - Taxonomy-aware term matching
    - Rule-based signal matching for amenities, condition, financing, and location
    """

    _PATTERN_GROUPS: dict[str, dict[str, tuple[str, ...]]] = {
        "amenities": {
            "pool": (r"\bpool\b", r"\bswimming pool\b", r"\blap pool\b"),
            "fireplace": (r"\bfireplace\b", r"\bfire place\b", r"\bhearth\b"),
            "garage": (r"\bgarage\b", r"\bcarport\b", r"\bparking\b"),
            "hot_tub": (r"\bhot tub\b", r"\bspa\b", r"\bjacuzzi\b"),
            "solar": (r"\bsolar\b", r"\bpaid[- ]off solar\b"),
            "gym": (r"\bgym\b", r"\bfitness\b"),
            "deck": (r"\bdeck\b", r"\bpatio\b", r"\boutdoor (?:living|dining)\b"),
            "elevator": (r"\belevator\b", r"\bprivate elevator\b"),
            "laundry": (r"\blaundry\b", r"\bwasher(?:\/|\s+)dryer\b"),
        },
        "condition_keywords": {
            "new_construction": (r"\bnew construction\b", r"\bbrand new\b"),
            "updated": (r"\bupdated\b", r"\bupgraded\b"),
            "remodeled": (r"\bremodel(?:ed|ing)?\b", r"\brenovat(?:ed|ion)\b"),
            "turn_key": (r"\bturn[- ]key\b", r"\bmove[- ]in ready\b"),
            "fixer": (r"\bfixer\b", r"\btlc\b", r"\bneeds work\b"),
        },
        "financing_terms": {
            "seller_financing": (r"\bseller financing\b", r"\bowner (?:will )?carry\b"),
            "assumable": (r"\bassumable\b", r"\bassume(?:d|able)\b"),
            "fha": (r"\bfha\b",),
            "va": (r"\bva (?:loan|financing)?\b",),
            "cash": (r"\bcash(?: only)?\b",),
            "conventional": (r"\bconventional\b",),
            "exchange_1031": (r"\b1031\b", r"\bexchange\b"),
        },
        "location_features": {
            "waterfront": (r"\bwaterfront\b", r"\boceanfront\b", r"\blakefront\b"),
            "views": (r"\bview(?:s)?\b", r"\bpanoramic\b", r"\bcity lights\b"),
            "walkable": (r"\bwalk(?:ing)? distance\b", r"\bwalkable\b"),
            "cul_de_sac": (r"\bcul[- ]de[- ]sac\b",),
            "corner_lot": (r"\bcorner lot\b",),
            "gated": (r"\bgated\b", r"\bgated community\b"),
            "near_parks": (r"\bnear\b.*\bpark\b", r"\bsteps from\b.*\bpark\b"),
            "near_shopping": (r"\bnear\b.*\bshopping\b", r"\bclose to\b.*\bstores?\b"),
            "near_transit": (r"\btrolley\b", r"\btransit\b", r"\bcommuter\b"),
            "school_district": (r"\bschool district\b", r"\btop[- ]rated schools?\b"),
        },
    }

    def __init__(
        self,
        taxonomy: dict[str, Any] | Path | str | None = None,
        entity_extractor: EntityExtractor | None = None,
    ) -> None:
        self.extractor = entity_extractor or EntityExtractor()
        self.taxonomy = self._load_taxonomy(taxonomy)
        self._taxonomy_terms = self._build_taxonomy_term_map(self.taxonomy)
        self._compiled_patterns = self._compile_patterns()

    @staticmethod
    def _load_taxonomy(taxonomy: dict[str, Any] | Path | str | None) -> dict[str, Any]:
        if taxonomy is None:
            path = Path(DEFAULT_TAXONOMY_PATH)
            if path.exists():
                return json.loads(path.read_text(encoding="utf-8"))
            return {"terms": []}
        if isinstance(taxonomy, dict):
            return taxonomy
        path = Path(taxonomy)
        if not path.exists():
            return {"terms": []}
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _build_taxonomy_term_map(taxonomy: dict[str, Any]) -> dict[str, list[str]]:
        categories = {
            "amenities": [],
            "condition_keywords": [],
            "location_features": [],
        }
        for term_obj in taxonomy.get("terms", []):
            term = str(term_obj.get("term", "")).strip().lower()
            category = str(term_obj.get("category", "")).strip().lower()
            if not term:
                continue
            if category == "amenities":
                categories["amenities"].append(term)
            elif category == "condition_and_style":
                categories["condition_keywords"].append(term)
            elif category in {"location", "views", "community"}:
                categories["location_features"].append(term)
        return categories

    def _compile_patterns(self) -> dict[str, list[tuple[str, re.Pattern[str]]]]:
        compiled: dict[str, list[tuple[str, re.Pattern[str]]]] = {}
        for group_name, group in self._PATTERN_GROUPS.items():
            compiled[group_name] = []
            for signal_name, patterns in group.items():
                for pattern in patterns:
                    compiled[group_name].append((signal_name, re.compile(pattern, re.IGNORECASE)))
        return compiled

    @staticmethod
    def _pick_remarks(record: Mapping[str, Any]) -> str:
        for key in ("remarks_clean", "remarks", "L_Remarks"):
            value = record.get(key)
            if value is not None and str(value).strip():
                return str(value)
        return ""

    @staticmethod
    def _pick_listing_id(record: Mapping[str, Any]) -> str:
        for key in ("L_ListingID", "listing_id", "id"):
            value = record.get(key)
            if value is not None and str(value).strip():
                return str(value)
        return ""

    @staticmethod
    def _dedupe(items: Iterable[str]) -> list[str]:
        seen: set[str] = set()
        deduped: list[str] = []
        for item in items:
            if item not in seen:
                deduped.append(item)
                seen.add(item)
        return deduped

    @staticmethod
    def _is_signal_term(term: str) -> bool:
        term = term.strip()
        if len(term) < 4:
            return False
        alpha = sum(1 for c in term if c.isalpha())
        return alpha >= 3

    def _match_taxonomy_terms(self, text: str, category: str) -> list[str]:
        text_l = text.lower()
        matches: list[str] = []
        for term in self._taxonomy_terms.get(category, []):
            if not self._is_signal_term(term):
                continue
            if term in text_l:
                matches.append(term.replace(" ", "_"))
        return self._dedupe(matches)

    def _match_pattern_group(self, text: str, group_name: str) -> list[str]:
        matches: list[str] = []
        for signal_name, pattern in self._compiled_patterns.get(group_name, []):
            if pattern.search(text):
                matches.append(signal_name)
        return self._dedupe(matches)

    def _extract_group(self, text: str, group_name: str) -> list[str]:
        pattern_matches = self._match_pattern_group(text, group_name)
        taxonomy_matches = self._match_taxonomy_terms(text, group_name)
        return self._dedupe(pattern_matches + taxonomy_matches)

    @staticmethod
    def _parse_number(value: Any) -> float | None:
        if value is None:
            return None
        if pd.isna(value):
            return None
        if isinstance(value, (int, float)):
            return float(value)
        raw = str(value).strip().replace(",", "")
        if not raw:
            return None
        try:
            return float(raw)
        except ValueError:
            return None

    @classmethod
    def _pick_record_number(cls, record: Mapping[str, Any], keys: tuple[str, ...]) -> float | None:
        for key in keys:
            parsed = cls._parse_number(record.get(key))
            if parsed is not None:
                return parsed
        return None

    def extract_signals(self, listing_record: Mapping[str, Any]) -> dict[str, Any]:
        remarks = self._pick_remarks(listing_record)
        listing_id = self._pick_listing_id(listing_record)
        entities = self.extractor.extract_all(remarks)

        # Prefer structured MLS fields when present; fall back to Week 3 text extraction.
        beds = self._pick_record_number(listing_record, ("beds", "bedrooms", "L_Keyword2"))
        baths = self._pick_record_number(listing_record, ("baths", "bathrooms", "LM_Dec_3"))
        price = self._pick_record_number(listing_record, ("price", "L_SystemPrice"))
        if beds is not None:
            entities["bedrooms"] = int(beds)
        if baths is not None:
            entities["bathrooms"] = float(baths)
        if price is not None:
            entities["price"] = int(price)

        amenities = self._extract_group(remarks, "amenities")
        condition = self._extract_group(remarks, "condition_keywords")
        financing = self._extract_group(remarks, "financing_terms")
        location = self._extract_group(remarks, "location_features")
        entity_amenities = entities.get("amenities")
        if not isinstance(entity_amenities, list):
            entity_amenities = []

        keywords = self._dedupe(
            amenities
            + condition
            + financing
            + location
            + [str(a).replace(" ", "_") for a in entity_amenities]
        )

        return {
            "listing_id": listing_id,
            "entities": entities,
            "amenities": amenities,
            "condition_keywords": condition,
            "financing_terms": financing,
            "location_features": location,
            "keywords": keywords,
        }

    def process_dataframe(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        return [self.extract_signals(rec) for rec in df.to_dict(orient="records")]


def load_rets_property_records(
    *,
    host: str,
    user: str,
    password: str,
    database: str,
    port: int,
) -> pd.DataFrame:
    conn = mysql.connector.connect(
        host=host,
        user=user,
        password=password,
        database=database,
        port=port,
    )
    try:
        query = """
        SELECT
            L_ListingID,
            L_Address,
            L_City,
            L_Keyword2 AS beds,
            LM_Dec_3 AS baths,
            L_SystemPrice AS price,
            L_Remarks
        FROM rets_property
        WHERE L_Remarks IS NOT NULL AND LENGTH(L_Remarks) > 20
        """
        return pd.read_sql(query, conn)
    finally:
        conn.close()


def evaluate_structured_accuracy(
    extracted_records: list[dict[str, Any]],
    source_df: pd.DataFrame,
) -> list[ExtractionAccuracy]:
    if source_df.empty:
        return []
    pred_by_id: dict[str, dict[str, Any]] = {
        str(r["listing_id"]): r["entities"] for r in extracted_records if r.get("listing_id")
    }
    fields: list[tuple[str, str]] = [("beds", "bedrooms"), ("baths", "bathrooms"), ("price", "price")]
    metrics: list[ExtractionAccuracy] = []
    for source_col, entity_col in fields:
        if source_col not in source_df.columns:
            continue
        numerator = 0
        denominator = 0
        for _, row in source_df.iterrows():
            lid = str(row.get("L_ListingID", ""))
            if not lid or lid not in pred_by_id:
                continue
            expected = row.get(source_col)
            expected_num = SignalExtractor._parse_number(expected)
            if expected_num is None:
                continue
            pred = pred_by_id[lid].get(entity_col)
            pred_num = SignalExtractor._parse_number(pred)
            if pred_num is None:
                denominator += 1
                continue
            denominator += 1
            if pred_num == expected_num:
                numerator += 1
        if denominator == 0:
            continue
        metrics.append(
            ExtractionAccuracy(
                metric=f"{entity_col}_exact_match",
                accuracy=numerator / denominator,
                numerator=numerator,
                denominator=denominator,
            )
        )
    return metrics


def evaluate_free_text_accuracy(
    extracted_records: list[dict[str, Any]],
    labeled_jsonl_path: Path | None,
) -> ExtractionAccuracy | None:
    if labeled_jsonl_path is None or not labeled_jsonl_path.exists():
        return None

    pred_by_id: dict[str, set[str]] = {
        str(r["listing_id"]): set(r.get("keywords", []))
        for r in extracted_records
        if r.get("listing_id")
    }
    numerator = 0
    denominator = 0
    with labeled_jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            listing_id = str(row.get("id", ""))
            if listing_id not in pred_by_id:
                continue
            expected_terms = [
                str(ent.get("value", "")).strip().lower().replace(" ", "_")
                for ent in row.get("entities", [])
                if str(ent.get("label", "")).upper() == "AMENITY"
                and str(ent.get("value", "")).strip()
            ]
            for term in expected_terms:
                denominator += 1
                if term in pred_by_id[listing_id]:
                    numerator += 1
    if denominator == 0:
        return None
    return ExtractionAccuracy(
        metric="free_text_amenity_match",
        accuracy=numerator / denominator,
        numerator=numerator,
        denominator=denominator,
    )


def save_signals_jsonl(signals: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in signals:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def run_week6_pipeline(
    *,
    source: str,
    csv_path: Path,
    output_path: Path,
    labeled_jsonl_path: Path | None,
    mysql_host: str,
    mysql_user: str,
    mysql_password: str,
    mysql_database: str,
    mysql_port: int,
) -> dict[str, Any]:
    if source == "mysql":
        df = load_rets_property_records(
            host=mysql_host,
            user=mysql_user,
            password=mysql_password,
            database=mysql_database,
            port=mysql_port,
        )
    elif source == "auto":
        try:
            df = load_rets_property_records(
                host=mysql_host,
                user=mysql_user,
                password=mysql_password,
                database=mysql_database,
                port=mysql_port,
            )
            source = "mysql"
        except Exception:
            df = pd.read_csv(csv_path)
            source = "csv"
    else:
        df = pd.read_csv(csv_path)

    extractor = SignalExtractor()
    signals = extractor.process_dataframe(df)
    save_signals_jsonl(signals, output_path)
    structured_metrics = evaluate_structured_accuracy(signals, df)
    free_text_metric = evaluate_free_text_accuracy(signals, labeled_jsonl_path)

    structured_ok = (
        bool(structured_metrics)
        and min(m.accuracy for m in structured_metrics) >= 0.90
    )
    free_text_ok = None if free_text_metric is None else free_text_metric.accuracy >= 0.75

    return {
        "source": source,
        "num_records": len(df),
        "output_path": str(output_path),
        "structured_accuracy": [asdict(m) for m in structured_metrics],
        "structured_target": 0.90,
        "structured_target_met": structured_ok,
        "free_text_accuracy": None if free_text_metric is None else asdict(free_text_metric),
        "free_text_target": 0.75,
        "free_text_target_met": free_text_ok,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Week 6 listing signal extraction pipeline.")
    parser.add_argument("--source", choices=["auto", "csv", "mysql"], default="auto")
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=_default_path("data/processed/listing_sample_cleaned.csv"),
        help="Fallback or explicit CSV data source for listing records.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=_default_path("data/processed/listing_signals.jsonl"),
        help="JSONL output for extracted listing signals.",
    )
    parser.add_argument(
        "--labeled-jsonl-path",
        type=Path,
        default=_default_path("data/processed/remarks_labeled.jsonl"),
        help="Optional labeled file used to score free-text amenity extraction.",
    )
    parser.add_argument("--mysql-host", default="127.0.0.1")
    parser.add_argument("--mysql-user", default="root")
    parser.add_argument("--mysql-password", default="root")
    parser.add_argument("--mysql-database", default="real_estate")
    parser.add_argument("--mysql-port", type=int, default=3308)
    args = parser.parse_args()

    report = run_week6_pipeline(
        source=args.source,
        csv_path=args.csv_path,
        output_path=args.output_path,
        labeled_jsonl_path=args.labeled_jsonl_path,
        mysql_host=args.mysql_host,
        mysql_user=args.mysql_user,
        mysql_password=args.mysql_password,
        mysql_database=args.mysql_database,
        mysql_port=args.mysql_port,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()