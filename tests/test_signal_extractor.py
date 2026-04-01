import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from scripts.signal_extractor import SignalExtractor, run_week6_pipeline


def test_extract_signals_includes_week6_groups():
    extractor = SignalExtractor(taxonomy={"terms": []})
    record = {
        "L_ListingID": "abc123",
        "L_Remarks": (
            "Brand new construction with pool, fireplace, and a private elevator. "
            "Seller financing available and assumable loan option. "
            "Waterfront lot with panoramic views and close to transit."
        ),
    }

    signals = extractor.extract_signals(record)
    assert signals["listing_id"] == "abc123"
    assert "entities" in signals
    assert "amenities" in signals
    assert "condition_keywords" in signals
    assert "financing_terms" in signals
    assert "location_features" in signals
    assert "keywords" in signals

    assert "pool" in signals["amenities"]
    assert "fireplace" in signals["amenities"]
    assert "elevator" in signals["amenities"]
    assert "new_construction" in signals["condition_keywords"]
    assert "seller_financing" in signals["financing_terms"]
    assert "assumable" in signals["financing_terms"]
    assert "waterfront" in signals["location_features"]
    assert "views" in signals["location_features"]


def test_pipeline_writes_jsonl_output(tmp_path):
    df = pd.DataFrame(
        [
            {
                "L_ListingID": "1",
                "beds": 3,
                "baths": 2,
                "price": 500000,
                "remarks": "Updated kitchen with garage and pool. Cash only sale.",
            },
            {
                "L_ListingID": "2",
                "beds": 2,
                "baths": 1,
                "price": 400000,
                "remarks": "Fixer upper near shopping and parks with fireplace.",
            },
        ]
    )
    csv_path = tmp_path / "sample.csv"
    out_path = tmp_path / "signals.jsonl"
    df.to_csv(csv_path, index=False)

    report = run_week6_pipeline(
        source="csv",
        csv_path=csv_path,
        output_path=out_path,
        labeled_jsonl_path=None,
        mysql_host="127.0.0.1",
        mysql_user="root",
        mysql_password="root",
        mysql_database="real_estate",
        mysql_port=3308,
    )

    assert report["num_records"] == 2
    assert out_path.exists()

    lines = out_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    row = json.loads(lines[0])
    assert {"listing_id", "entities", "amenities", "condition_keywords", "financing_terms", "location_features", "keywords"} <= set(
        row.keys()
    )
