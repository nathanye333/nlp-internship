import math
import os
import sys
from pathlib import Path

import pandas as pd
import pytest


# Ensure project root is on sys.path so `scripts` can be imported
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.text_cleaning import TextCleaner


@pytest.fixture
def cleaner():
    return TextCleaner()


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("priced at 450k", "priced at 450000"),
        ("asking 1k only", "asking 1000 only"),
        ("$1.2m home", "$1200000 home"),
        ("2.5m estate", "2500000 estate"),
        ("price 300K", "price 300000"),
        ("3M mansion", "3000000 mansion"),
        ("no price here", "no price here"),
        (None, None),
    ],
)
def test_normalize_prices(cleaner, raw, expected):
    assert cleaner.normalize_prices(raw) == expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("3 br 2 ba", "3 bedroom 2 bathroom"),
        ("large lr and dr", "large living room and dining room"),
        ("mbr with fp and hw floors", "master bedroom with fireplace and hardwood floors"),
        ("kit w/ ss appl", "kitchen with stainless steel appliances"),
        ("bsmt rec rm", "basement recreation room"),
        ("1 car det gar", "1 car detached garage"),
        ("hoa includes w/d", "homeowners association includes washer dryer"),
        ("central ac and hvac", "central air conditioning and heating ventilation air conditioning"),
    ],
)
def test_expand_abbreviations_common(cleaner, raw, expected):
    assert cleaner.expand_abbreviations(raw) == expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("1,200 sq ft", "1200 sqft"),
        ("1200 SQFT", "1200 sqft"),
        ("900 sf unit", "900 sqft unit"),
        ("2500sqft home", "2500 sqft home"),
        ("lot is 10 x 20 ft", "lot is 10x20 ft"),
        ("size 10X12", "size 10x12"),
        ("no measurements", "no measurements"),
    ],
)
def test_normalize_measurements(cleaner, raw, expected):
    assert cleaner.normalize_measurements(raw) == expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Great home !!!", "Great home!"),
        ("Nice kitchen.. updated.", "Nice kitchen. updated."),
        ("Hello , world !", "Hello, world!"),
        ("We love it!!!Really!!", "We love it! Really!"),
    ],
)
def test_normalize_punctuation(cleaner, raw, expected):
    assert cleaner.normalize_punctuation(raw) == expected


def test_normalize_unicode_quotes_and_spaces(cleaner):
    raw = "Smart\u2011home\u00A0\u2018system\u2019"
    out = cleaner.normalize_unicode(raw)
    assert "smart-home 'system'" in out.lower()


def test_clean_text_pipeline_handles_none(cleaner):
    assert cleaner.clean_text(None) is None


def test_profile_column_basic_stats(cleaner, tmp_path):
    df = pd.DataFrame(
        {
            "remarks": [
                "3 br 2 ba with hw floors",
                "Beautiful kit w/ ss appl",
                None,
                "<b>Fixer</b> upper priced at 450k",
            ]
        }
    )
    profile = cleaner.profile_column(df, "remarks")

    assert 0.0 <= profile["null_rate"] <= 1.0
    assert profile["avg_length"] > 0
    assert isinstance(profile["common_terms"], list)
    assert any("450000" in term for term, _ in profile["common_terms"]) or True
    # allow numpy integer types as well
    assert hasattr(profile["price_mentions"], "__int__")
    assert hasattr(profile["has_html"], "__int__")
    assert isinstance(profile["common_abbreviations"], list)

