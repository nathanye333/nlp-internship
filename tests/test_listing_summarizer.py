import re
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from scripts.listing_summarizer import ListingSummarizer


def _sample_listing() -> dict:
    return {
        "city": "Irvine",
        "bedrooms": 3,
        "bathrooms": 2.5,
        "price": 875000,
        "features": ["pool", "garage"],
        "remarks": (
            "Beautiful 3 bedroom home in Irvine with open-concept living space and great natural light. "
            "The backyard includes a pool and covered patio, perfect for entertaining. "
            "Recent upgrades include quartz counters and stainless appliances."
        ),
    }


def test_extractive_summary_includes_required_fields():
    summarizer = ListingSummarizer()
    summary = summarizer.extractive_summary(_sample_listing())
    lowered = summary.lower()

    assert "3 bed" in lowered
    assert "2.5 bath" in lowered
    assert "$875,000" in summary
    assert "irvine" in lowered
    assert "pool" in lowered
    assert "garage" in lowered


def test_extract_summary_returns_compact_two_to_three_sentences():
    summarizer = ListingSummarizer()
    summary = summarizer.extractive_summary(_sample_listing(), num_sentences=3)
    sentences = [part for part in re.split(r"(?<=[.!?])\s+", summary) if part.strip()]
    assert 2 <= len(sentences) <= 3


def test_rouge_l_meets_threshold():
    summarizer = ListingSummarizer()
    listings = [
        _sample_listing(),
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
    ]

    predictions = [summarizer.extractive_summary(listing) for listing in listings]
    references = [
        "3 bed, 2.5 bath home in Irvine listed at $875,000. Highlights include pool and garage.",
        "4 bed, 3 bath home in Austin listed at $690,000. Highlights include office and fireplace.",
    ]
    rouge = summarizer.evaluate_rouge_l(predictions, references)
    assert rouge.rouge_l > 0.4


def test_human_eval_sampler_returns_20_or_less_rows():
    summarizer = ListingSummarizer()
    listings = [_sample_listing() for _ in range(5)]
    sampled = summarizer.sample_for_human_evaluation(listings, sample_size=20)
    assert len(sampled) == 5
    assert all("summary" in row and "remarks" in row for row in sampled)
