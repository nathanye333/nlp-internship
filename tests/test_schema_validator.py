import sys
from pathlib import Path

import pytest


# Ensure project root is on sys.path so `scripts` can be imported
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from scripts.schema_validator import SchemaValidator


@pytest.fixture
def validator():
    # Provide a controlled city set for deterministic tests
    return SchemaValidator(valid_cities={"Irvine", "San Diego", "Los Angeles", "LA"})


@pytest.mark.parametrize(
    "filters,ok",
    [
        ({"city": "Irvine", "price_max": 700000, "bedrooms_min": 3, "bedrooms_max": 3}, True),
        ({"city": "Atlantis", "price_max": 700000}, False),
        ({"price_min": 1000}, False),
        ({"price_max": 100_000_000}, False),
        ({"bedrooms_min": -1}, False),
        ({"bedrooms_max": 999}, False),
        ({"bathrooms_min": -1.0}, False),
        ({"bathrooms_max": 99.0}, False),
        ({"price_min": 900000, "price_max": 600000}, False),
        ({"bedrooms_min": 4, "bedrooms_max": 3}, False),
        ({"bathrooms_min": 3.0, "bathrooms_max": 2.0}, False),
        ({"amenities_in": ["pool", "garage"], "amenities_out": ["waterfront"]}, True),
        ({"amenities_in": "pool"}, False),
        ({"amenities_out": [""]}, False),
    ],
)
def test_validate_filters(validator, filters, ok):
    result = validator.validate_filters(filters)
    assert result.ok == ok
    assert isinstance(result.errors, list)
    if not ok:
        assert len(result.errors) >= 1
