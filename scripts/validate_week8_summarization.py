from __future__ import annotations

import csv
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from scripts.listing_summarizer import ListingSummarizer


def _evaluation_listings() -> list[dict]:
    return [
        {
            "city": "Irvine",
            "bedrooms": 3,
            "bathrooms": 2.5,
            "price": 875000,
            "features": ["pool", "garage"],
            "remarks": (
                "Beautiful 3 bedroom home in Irvine with open-concept living space and great natural light. "
                "The backyard includes a pool and covered patio, perfect for entertaining."
            ),
        },
        {
            "city": "Austin",
            "bedrooms": 4,
            "bathrooms": 3,
            "price": 690000,
            "features": ["office", "fireplace"],
            "remarks": (
                "Spacious 4 bed in Austin near major tech campuses with bright interiors. "
                "Dedicated office and fireplace add comfort for remote work and gatherings."
            ),
        },
        {
            "city": "Seattle",
            "bedrooms": 2,
            "bathrooms": 2,
            "price": 735000,
            "features": ["view", "updated kitchen"],
            "remarks": (
                "Modern Seattle condo with water views and efficient layout. "
                "Updated kitchen and natural light make the home feel open and inviting."
            ),
        },
    ]


def _reference_summaries() -> list[str]:
    return [
        "3 bed, 2.5 bath home in Irvine listed at $875,000 with pool and garage.",
        "4 bed, 3 bath home in Austin listed at $690,000 with office and fireplace.",
        "2 bed, 2 bath home in Seattle listed at $735,000 with view and updated kitchen.",
    ]


def run_validation() -> dict[str, float | int | str]:
    summarizer = ListingSummarizer()
    listings = _evaluation_listings()
    predictions = [summarizer.extractive_summary(listing) for listing in listings]
    rouge = summarizer.evaluate_rouge_l(predictions, _reference_summaries())

    human_eval_rows = summarizer.sample_for_human_evaluation(listings * 10, sample_size=20, seed=42)
    output_path = Path("data/processed/week8_human_eval_sample.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["summary", "remarks"])
        writer.writeheader()
        writer.writerows(human_eval_rows)

    return {
        "rouge_l": float(rouge.rouge_l),
        "rouge_threshold": 0.4,
        "human_eval_rows": len(human_eval_rows),
        "human_eval_output": str(output_path),
    }


if __name__ == "__main__":
    metrics = run_validation()
    print(f"ROUGE-L: {metrics['rouge_l']:.4f}")
    print(f"Threshold: {metrics['rouge_threshold']:.2f}")
    print(f"Human eval rows: {metrics['human_eval_rows']}")
    print(f"Human eval sample: {metrics['human_eval_output']}")
    if metrics["rouge_l"] < metrics["rouge_threshold"]:
        raise SystemExit("Validation failed: ROUGE-L below threshold")
