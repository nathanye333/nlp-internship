import json
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

# Allow running as script: ensure project root is on path so "scripts" can be imported
if __name__ == "__main__":
    _root = Path(__file__).resolve().parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from scripts.entity_extractor import EntityExtractor


DATA_DIR = Path("data/processed")
INPUT_CSV = DATA_DIR / "listing_sample.csv"
OUTPUT_JSONL = DATA_DIR / "remarks_labeled.jsonl"


def make_weak_labels(n_samples: int = 250) -> List[Dict]:
    """
    Create a weakly labeled dataset using the rule-based EntityExtractor.

    This is intended as a starting point for manual labeling of 200–300
    remarks with entity spans. It writes a JSONL file where each line is:

    {
      "id": str,
      "text": str,
      "entities": [
        {"label": "BEDROOMS", "start": int, "end": int, "value": 3},
        {"label": "BATHROOMS", "start": int, "end": int, "value": 2.5},
        {"label": "PRICE", "start": int, "end": int, "value": 750000},
        {"label": "SQFT", "start": int, "end": int, "value": 1800},
        {"label": "AMENITY", "start": int, "end": int, "value": "pool"},
        ...
      ]
    }

    You can then refine / correct these labels manually or in a labeling tool.
    """
    df = pd.read_csv(INPUT_CSV)
    df = df.dropna(subset=["remarks"]).copy()

    if len(df) > n_samples:
        df = df.sample(n_samples, random_state=42)

    extractor = EntityExtractor()
    records: List[Dict] = []

    for i, row in df.iterrows():
        text = str(row["remarks"])
        spans: List[Dict] = []

        extracted = extractor.extract_all(text)
        text_l = text.lower()

        # Helper to find first occurrence span for a value rendered as text
        def add_span(label: str, value_str: str, value):
            idx = text_l.find(value_str.lower())
            if idx == -1:
                return
            spans.append(
                {
                    "label": label,
                    "start": idx,
                    "end": idx + len(value_str),
                    "value": value,
                }
            )

        # Bedrooms
        beds = extracted.get("bedrooms")
        if beds is not None:
            # naive rendering, can be corrected during manual labeling
            add_span("BEDROOMS", f"{beds} bed", beds)

        # Bathrooms
        baths = extracted.get("bathrooms")
        if baths is not None:
            baths_str = f"{baths}".rstrip("0").rstrip(".") if isinstance(baths, float) else str(baths)
            add_span("BATHROOMS", f"{baths_str} bath", baths)

        # Price
        price = extracted.get("price")
        if price is not None:
            price_str = f"{price:,}"
            add_span("PRICE", price_str, price)

        # Square footage (may be multiple per listing)
        sqft_vals = extracted.get("sqft") or []
        if not isinstance(sqft_vals, list):
            sqft_vals = [sqft_vals] if sqft_vals is not None else []
        for sqft in sqft_vals:
            if sqft is not None:
                sqft_str = f"{sqft:,}"
                add_span("SQFT", f"{sqft_str} sq ft", sqft)

        # Amenities (may be multiple)
        amenities = extracted.get("amenities")
        if isinstance(amenities, list):
            for amenity in amenities:
                idx = text_l.find(str(amenity).lower())
                if idx != -1:
                    spans.append(
                        {
                            "label": "AMENITY",
                            "start": idx,
                            "end": idx + len(str(amenity)),
                            "value": amenity,
                        }
                    )

        records.append(
            {
                "id": str(row.get("L_ListingID", i)),
                "text": text,
                "entities": spans,
            }
        )

    return records


def save_jsonl(records: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    records = make_weak_labels()
    save_jsonl(records, OUTPUT_JSONL)
    print(f"Wrote {len(records)} weakly labeled remarks to {OUTPUT_JSONL}")
    print("You can now refine entity spans/labels to create a high-quality dataset.")


if __name__ == "__main__":
    main()

