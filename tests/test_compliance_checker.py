"""Tests for :mod:`scripts.compliance_checker`.

The suite does double duty:

1. Unit tests that pin the behaviour of individual patterns and the severity
   API (errors block, warnings / infos do not).
2. A metrics block that replays a labelled corpus through the checker and
   asserts the Week 9 success criteria:

       * recall == 1.0   (no false negatives on known violations)
       * precision > 0.8 (few false positives on clean listings)

The corpus is defined inline so the test is self-contained and easy to extend.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from scripts.compliance_checker import (  # noqa: E402
    ERROR,
    INFO,
    WARNING,
    ComplianceChecker,
    ComplianceReport,
    Violation,
)


@pytest.fixture(scope="module")
def checker() -> ComplianceChecker:
    return ComplianceChecker()


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------
#
# Each entry: (text, expected_has_error) where ``expected_has_error`` is True
# when the listing is known to contain prohibited language that must be
# blocked. This is the ground truth used for recall / precision.
#
# "Violating" examples are drawn from HUD advertising guidance, NFHA training
# materials, and real fair-housing enforcement actions (paraphrased).
#

VIOLATING_LISTINGS: list[str] = [
    # race / color
    "Quiet home in a white neighborhood with mountain views.",
    "Looking for Caucasian tenants only, 1-year lease.",
    "No blacks please, serious buyers only.",
    "Traditionally white suburb, great schools nearby.",
    "Light-skinned renters preferred.",
    # national origin
    "Americans only - no foreigners.",
    "English-speaking family required.",
    "Must speak English to lease this unit.",
    # religion
    "Perfect Christian community for a good Christian family.",
    "Jewish neighborhood, walk to shul.",
    "No Muslims, quiet building.",
    "Church-going household preferred.",
    # sex / orientation
    "Women only, upstairs bedroom available.",
    "No men allowed in this rental.",
    "Straight couples only please.",
    # familial status
    "Adults only, no children, quiet complex.",
    "Child-free building, professional tenants.",
    "No kids under any circumstances.",
    "Not suitable for children - steep stairs and open loft.",
    "No more than 2 children allowed in the unit.",
    # disability
    "No wheelchairs, third floor walk-up.",
    "Tenant must be able-bodied to maintain the yard.",
    "No service animals, strict no-pet policy.",
    "Sane tenants only, no exceptions.",
    # source of income
    "No Section 8, no vouchers.",
    "No housing assistance of any kind.",
    # age
    "Must be over 40 to apply.",
    "No seniors, this is a fast-paced building.",
    # mixed / realistic
    (
        "Charming 2 bed bungalow. Perfect for a young Christian couple, "
        "no children. Close to church and great schools."
    ),
    (
        "Luxury high-rise, English-speaking tenants only. No Section 8. "
        "Adults only community."
    ),
]


CLEAN_LISTINGS: list[str] = [
    "3 bed, 2 bath single-family home with a two-car garage and pool.",
    "Newly renovated condo in downtown Austin. Quartz counters, stainless appliances.",
    "Spacious 4-bedroom colonial on a half-acre lot with mature oak trees.",
    "Open-concept loft with floor-to-ceiling windows, in-unit laundry, and a rooftop deck.",
    "Cozy cottage 0.5 miles from the train station. Hardwood floors throughout.",
    "Modern townhome with smart-home features and an attached two-car garage.",
    "Waterfront property with 120 feet of private dock and panoramic bay views.",
    "Corner-lot ranch with an updated kitchen, new HVAC, and a fenced backyard.",
    "Gated community with 24-hour security, pool, and fitness center.",
    "Walk to farmers market, craft brewery, and the public library.",
    "Quiet cul-de-sac, top-rated school district (buyer to verify).",
    "Studio apartment with Murphy bed, utilities included, pet friendly.",
    "Five-bedroom estate with tennis court, home theater, and wine cellar.",
    "Mid-century ranch awaiting your renovation vision. Sold as-is.",
    "Two-bedroom unit with assigned parking and bike storage.",
    "Bright south-facing unit with Energy Star appliances and LED lighting.",
    "Freshly painted interior, new carpet, and a covered patio.",
    "Walk-up building, third floor unit, no elevator available.",
    # Intentionally ambiguous but legal - describes physical facts.
    "Unit is not accessible by elevator; stairs only.",
    # Tricky: mentions religion only as a landmark; should downgrade to warning, not error.
    "Two blocks from St. Mary's Cathedral and the downtown transit hub.",
    "Rent-controlled apartment near the bus line; income 3x rent required.",
    "Spacious family room, granite kitchen, and gas fireplace.",
    "Recent updates include a new roof (2024) and tankless water heater.",
    "Unit is in a 55+ active-adult community; HOA documents attached.",
    "Four blocks from Temple Emanuel and the light rail station.",
]


# ---------------------------------------------------------------------------
# Behavioural tests
# ---------------------------------------------------------------------------


def test_empty_input_is_compliant(checker: ComplianceChecker) -> None:
    report = checker.check_listing("")
    assert isinstance(report, ComplianceReport)
    assert report.compliant is True
    assert report.violations == []


def test_none_input_is_compliant(checker: ComplianceChecker) -> None:
    report = checker.check_listing(None)
    assert report.compliant is True


def test_clean_listing_has_no_violations(checker: ComplianceChecker) -> None:
    report = checker.check_listing(
        "3 bed, 2 bath home with attached garage and newly landscaped yard."
    )
    assert report.compliant is True
    assert report.error_count == 0
    assert report.warning_count == 0


def test_error_violation_blocks_publication(checker: ComplianceChecker) -> None:
    report = checker.check_listing("Adults only, no children.")
    assert report.compliant is False
    assert report.error_count >= 1
    assert any(v.severity == ERROR for v in report.violations)
    assert report.blocking()


def test_warning_does_not_block(checker: ComplianceChecker) -> None:
    report = checker.check_listing("Perfect for young professionals.")
    # "young ... professionals" is a warning, not an error.
    assert any(v.severity == WARNING for v in report.violations)
    assert report.error_count == 0
    assert report.compliant is True


def test_info_does_not_block(checker: ComplianceChecker) -> None:
    report = checker.check_listing(
        "Located in a desirable neighborhood with great schools."
    )
    assert report.error_count == 0
    assert any(v.severity == INFO for v in report.violations)
    assert report.compliant is True


def test_violation_includes_suggestion_and_offsets(checker: ComplianceChecker) -> None:
    text = "No children allowed in this unit."
    report = checker.check_listing(text)
    violation = next(v for v in report.violations if v.severity == ERROR)
    assert isinstance(violation, Violation)
    assert violation.category == "familial_status"
    assert text[violation.start : violation.end].lower() == violation.matched_text.lower()
    assert violation.suggestion  # non-empty actionable fix


def test_word_boundary_prevents_substring_false_positive(checker: ComplianceChecker) -> None:
    # "mann" inside "Hoffmann" must not trigger the "man only" rule.
    report = checker.check_listing("Listed by the Hoffmann family team.")
    assert report.error_count == 0


def test_religious_landmark_is_warning_not_error(checker: ComplianceChecker) -> None:
    report = checker.check_listing("Walking distance to the church and park.")
    assert report.error_count == 0
    assert report.compliant is True
    assert any(v.category == "religion" and v.severity == WARNING for v in report.violations)


def test_multiple_violations_are_all_detected(checker: ComplianceChecker) -> None:
    report = checker.check_listing(
        "Adults only, no Section 8, English-speaking tenants only."
    )
    categories = {v.category for v in report.violations if v.severity == ERROR}
    assert {"familial_status", "source_of_income", "national_origin"}.issubset(categories)


def test_extra_patterns_are_honored() -> None:
    extra = {
        "state_specific": [
            (r"(?<![A-Za-z0-9])military\s+discouraged(?![A-Za-z0-9])", ERROR,
             "Source-of-income / military-status discrimination",
             "Remove."),
        ]
    }
    custom = ComplianceChecker(extra_patterns=extra)
    report = custom.check_listing("Great unit but military discouraged.")
    assert report.error_count == 1
    assert report.violations[0].category == "state_specific"


def test_format_report_returns_text(checker: ComplianceChecker) -> None:
    out = checker.format_report(checker.check_listing("No children."))
    assert "ERROR" in out
    assert "familial_status" in out


def test_to_dict_is_json_ready(checker: ComplianceChecker) -> None:
    import json

    payload = checker.check_listing("Adults only.").to_dict()
    assert json.loads(json.dumps(payload))["compliant"] is False


# ---------------------------------------------------------------------------
# Metric tests (Week 9 acceptance criteria)
# ---------------------------------------------------------------------------


def _labelled_corpus() -> list[tuple[str, bool]]:
    return [(t, True) for t in VIOLATING_LISTINGS] + [(t, False) for t in CLEAN_LISTINGS]


def test_recall_is_100_percent_on_known_violations(checker: ComplianceChecker) -> None:
    """Every known-bad listing must surface at least one error."""
    missed: list[str] = []
    for text in VIOLATING_LISTINGS:
        report = checker.check_listing(text)
        if report.error_count == 0:
            missed.append(text)
    assert not missed, f"False negatives (missed {len(missed)}): {missed}"


def test_precision_above_80_percent(checker: ComplianceChecker) -> None:
    """On the combined corpus, precision of the error severity must exceed 0.8."""
    true_positive = 0
    false_positive = 0
    false_negative = 0
    for text, is_violation in _labelled_corpus():
        report = checker.check_listing(text)
        flagged_as_error = report.error_count > 0
        if flagged_as_error and is_violation:
            true_positive += 1
        elif flagged_as_error and not is_violation:
            false_positive += 1
        elif not flagged_as_error and is_violation:
            false_negative += 1

    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)

    assert recall == 1.0, f"Recall below 1.0: {recall:.3f}"
    assert precision > 0.8, f"Precision {precision:.3f} did not exceed 0.80"
