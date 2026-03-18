import sys
from pathlib import Path

import pytest


# Ensure project root is on sys.path so `scripts` can be imported
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from scripts.query_parser import QueryParser


@pytest.fixture
def parser():
    return QueryParser()


def _assert_subset(actual: dict, expected_subset: dict):
    for k, v in expected_subset.items():
        assert k in actual, f"Missing key: {k}"
        assert actual[k] == v, f"Key {k}: expected {v}, got {actual[k]}"


@pytest.mark.parametrize(
    "query,expected_subset",
    [
        # --- canonical example ---
        ("3 bed under 700k in Irvine", {"bedrooms_min": 3, "bedrooms_max": 3, "price_max": 700000, "city": "Irvine"}),
        ("3br under $700k in irvine", {"bedrooms_min": 3, "bedrooms_max": 3, "price_max": 700000, "city": "Irvine"}),
        ("3 bedrooms below 700000 in Irvine", {"bedrooms_min": 3, "bedrooms_max": 3, "price_max": 700000, "city": "Irvine"}),
        # --- price patterns ---
        ("under 500k", {"price_max": 500000}),
        ("below $450000", {"price_max": 450000}),
        ("less than 1.2m", {"price_max": 1200000}),
        ("at most 950k", {"price_max": 950000}),
        ("no more than $2m", {"price_max": 2000000}),
        ("over 800k", {"price_min": 800000}),
        ("above $1m", {"price_min": 1000000}),
        ("more than 750000", {"price_min": 750000}),
        ("at least 650k", {"price_min": 650000}),
        ("minimum 300k", {"price_min": 300000}),
        ("between 600k and 900k", {"price_min": 600000, "price_max": 900000}),
        ("from $700k to $900k", {"price_min": 700000, "price_max": 900000}),
        ("between 900k and 600k", {"price_min": 600000, "price_max": 900000}),
        ("price: 700k", {"price_max": 700000}),
        ("budget 1.5m", {"price_max": 1500000}),
        # --- bed patterns ---
        ("3 bed", {"bedrooms_min": 3, "bedrooms_max": 3}),
        ("4 beds", {"bedrooms_min": 4, "bedrooms_max": 4}),
        ("2 bedroom", {"bedrooms_min": 2, "bedrooms_max": 2}),
        ("3+ bed", {"bedrooms_min": 3}),
        ("3 + br", {"bedrooms_min": 3}),
        ("at least 5 bedrooms", {"bedrooms_min": 5}),
        ("3-4 bed", {"bedrooms_min": 3, "bedrooms_max": 4}),
        ("2 to 3 bedrooms", {"bedrooms_min": 2, "bedrooms_max": 3}),
        ("studio under 600k", {"bedrooms_max": 0, "price_max": 600000}),
        # --- bath patterns ---
        ("2 bath", {"bathrooms_min": 2.0, "bathrooms_max": 2.0}),
        ("2.5 baths", {"bathrooms_min": 2.5, "bathrooms_max": 2.5}),
        ("at least 2.5 bathroom", {"bathrooms_min": 2.5}),
        ("2-3 bath", {"bathrooms_min": 2.0, "bathrooms_max": 3.0}),
        ("2 to 2.5 bathrooms", {"bathrooms_min": 2.0, "bathrooms_max": 2.5}),
        # --- city patterns ---
        ("in San Diego under 900k", {"city": "San Diego", "price_max": 900000}),
        ("near los angeles 3 bed", {"city": "Los Angeles", "bedrooms_min": 3, "bedrooms_max": 3}),
        ("around LA under 800k", {"city": "LA", "price_max": 800000}),
        ("within Irvine 3 bed", {"city": "Irvine", "bedrooms_min": 3, "bedrooms_max": 3}),
        # --- amenities include ---
        ("with pool under 1m", {"price_max": 1000000, "amenities_in": ["pool"]}),
        ("pool and garage", {"amenities_in": ["garage", "pool"]}),
        ("fireplace 3 bed", {"amenities_in": ["fireplace"], "bedrooms_min": 3, "bedrooms_max": 3}),
        ("hardwood floors under 900k", {"amenities_in": ["hardwood"], "price_max": 900000}),
        ("basement and stainless steel", {"amenities_in": ["basement", "stainless steel"]}),
        ("new construction in Irvine", {"amenities_in": ["new construction"], "city": "Irvine"}),
        ("hoa and washer dryer", {"amenities_in": ["hoa", "washer dryer"]}),
        # --- negation ---
        ("no pool under 900k", {"amenities_out": ["pool"], "price_max": 900000}),
        ("without garage 3 bed", {"amenities_out": ["garage"], "bedrooms_min": 3, "bedrooms_max": 3}),
        ("not waterfront", {"amenities_out": ["waterfront"]}),
        ("w/o fireplace", {"amenities_out": ["fireplace"]}),
        # --- combined ---
        (
            "4 bed 3 bath between 800k and 1.2m in Irvine with pool",
            {
                "bedrooms_min": 4,
                "bedrooms_max": 4,
                "bathrooms_min": 3.0,
                "bathrooms_max": 3.0,
                "price_min": 800000,
                "price_max": 1200000,
                "city": "Irvine",
                "amenities_in": ["pool"],
            },
        ),
        (
            "3-4 bed at least 2 bath over 900k near San Diego with garage no pool",
            {
                "bedrooms_min": 3,
                "bedrooms_max": 4,
                "bathrooms_min": 2.0,
                "price_min": 900000,
                "city": "San Diego",
                "amenities_in": ["garage"],
                "amenities_out": ["pool"],
            },
        ),
        # --- injection-like strings should not break parsing ---
        ("3 bed in Irvine' OR 1=1 -- under 700k", {"bedrooms_min": 3, "bedrooms_max": 3, "price_max": 700000}),
        ("in '; DROP TABLE rets_property; -- under 500k", {"price_max": 500000}),
    ],
)
def test_parse_expected_filters(parser, query, expected_subset):
    filters = parser.parse(query)
    _assert_subset(filters, expected_subset)


@pytest.mark.parametrize(
    "query",
    [
        "3 bed under 700k in Irvine",
        "between 600k and 900k with pool",
        "no pool in San Diego over 800k",
        "2-3 bath at least 4 bed near Los Angeles",
        "in Irvine' OR 1=1 --",
    ],
)
def test_to_where_sql_is_parameterized(parser, query):
    parsed = parser.parse_to_sql(query)
    where_sql, params = parsed.where_sql, parsed.params

    # No raw user string should appear as executable SQL; only placeholders.
    assert ";" not in where_sql
    assert "--" not in where_sql
    assert "%s" in where_sql or where_sql == ""
    assert isinstance(params, list)


def test_where_sql_order_matches_params(parser):
    filters = {
        "price_min": 600000,
        "price_max": 900000,
        "bedrooms_min": 3,
        "city": "Irvine",
        "amenities_in": ["pool"],
        "amenities_out": ["garage"],
    }
    where_sql, params = parser.to_where_sql(filters)
    # Ensure placeholders count matches params
    assert where_sql.count("%s") == len(params)
