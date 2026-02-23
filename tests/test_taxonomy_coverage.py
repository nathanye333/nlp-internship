import json
from pathlib import Path

import pandas as pd


DATA_DIR = Path("data/processed")


def load_taxonomy():
    with open(DATA_DIR / "taxonomy.json") as f:
        return json.load(f)


def load_remarks():
    df = pd.read_csv(DATA_DIR / "listing_sample.csv")
    return df["remarks"].dropna().astype(str).str.lower().tolist()


def test_taxonomy_coverage_at_least_30_percent():
    taxonomy = load_taxonomy()
    remarks = load_remarks()

    terms = [t["term"].lower() for t in taxonomy["terms"]]

    covered = 0
    for text in remarks:
        if any(term in text for term in terms):
            covered += 1

    coverage = covered / len(remarks) if remarks else 0.0
    assert coverage >= 0.30, f"Expected coverage >= 0.30, got {coverage:.2%}"
