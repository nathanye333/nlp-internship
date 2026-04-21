"""Fair Housing Act compliance checking for real estate listings.

The Fair Housing Act (42 U.S.C. 3601 et seq.) prohibits discrimination in
housing advertisements on the basis of seven federally protected classes:
race, color, national origin, religion, sex (including gender identity and
sexual orientation), familial status, and disability. Many states add more
(age, source of income, marital status, military status, etc.).

This module scans listing descriptions for language that HUD and fair-housing
advocacy groups have identified as problematic. Matches are reported with one
of three severities so the publishing workflow can decide whether to block,
warn, or merely inform:

    error    The phrase is almost always non-compliant and must be removed
             or rewritten before the listing can be published.
    warning  The phrase is context dependent. A human reviewer should confirm
             it does not create a discriminatory preference.
    info     Descriptive language that is legal today but frequently
             misunderstood or steered (e.g. "family room"). Logged for
             awareness only.

See ``docs/fair_housing_guidelines.md`` for policy background and
``scripts/listing_submission_example.py`` for a workflow integration.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable


ERROR = "error"
WARNING = "warning"
INFO = "info"

# Ordered so more specific categories surface first in reports.
_CATEGORIES = (
    "race",
    "color",
    "national_origin",
    "religion",
    "sex",
    "familial_status",
    "disability",
    "source_of_income",
    "age",
    "steering",
)


@dataclass(frozen=True)
class Violation:
    category: str
    severity: str
    pattern: str
    matched_text: str
    start: int
    end: int
    message: str
    suggestion: str = ""


@dataclass
class ComplianceReport:
    compliant: bool
    violations: list[Violation] = field(default_factory=list)
    error_count: int = 0
    warning_count: int = 0
    info_count: int = 0

    def to_dict(self) -> dict:
        return {
            "compliant": self.compliant,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "info_count": self.info_count,
            "violations": [v.__dict__ for v in self.violations],
        }

    def blocking(self) -> list[Violation]:
        """Violations that should prevent publication (severity == error)."""
        return [v for v in self.violations if v.severity == ERROR]


# ---------------------------------------------------------------------------
# Pattern library
# ---------------------------------------------------------------------------
#
# Each entry is (regex, severity, human_message, suggested_fix). Patterns are
# compiled with word-boundary wrappers where appropriate so that substrings
# (e.g. "mann" inside "Hoffmann") do not trigger false positives.
#
# Rules of thumb driving the library (see docs/fair_housing_guidelines.md):
#   * Describe the *property*, not the *ideal occupant*.
#   * Avoid references to any protected class or proxy for one.
#   * Religious/ethnic landmark names are allowed as directional references
#     but flagged as WARNING for human review.
#
PatternSpec = tuple[str, str, str, str]


def _word(pattern: str) -> str:
    """Wrap a pattern so it only matches on word boundaries."""
    return rf"(?<![A-Za-z0-9])(?:{pattern})(?![A-Za-z0-9])"


_PATTERN_LIBRARY: dict[str, list[PatternSpec]] = {
    "race": [
        (_word(r"white(?:[\s-]+)(?:only|neighborhood|tenants?|family|families|buyers?|community)"),
         ERROR,
         "Explicit racial preference",
         "Remove references to race; describe the property instead."),
        (_word(r"black(?:[\s-]+)(?:only|tenants?|family|families|buyers?)"),
         ERROR,
         "Explicit racial preference",
         "Remove references to race."),
        (_word(r"(?:asian|hispanic|latino|latina|caucasian|oriental)(?:[\s-]+)(?:only|neighborhood|tenants?|family|families|buyers?|community)"),
         ERROR,
         "Explicit racial or ethnic preference",
         "Remove references to race or ethnicity."),
        (_word(r"no\s+(?:blacks?|whites?|asians?|hispanics?|latinos?|mexicans?)"),
         ERROR,
         "Exclusion of a racial or ethnic group",
         "Remove; exclusions based on race are prohibited."),
        (_word(r"(?:exclusive|restricted)\s+(?:white|black|asian|hispanic|ethnic)\s+(?:community|neighborhood|area)"),
         ERROR,
         "Race-based community restriction",
         "Remove; describe amenities rather than demographics."),
        (_word(r"(?:ethnic|diverse)\s+(?:neighborhood|area|community)"),
         WARNING,
         "'Ethnic' / 'diverse' can steer by race. Review context.",
         "If describing the property, focus on amenities or landmarks instead."),
        (_word(r"traditionally\s+(?:white|black|asian|hispanic)"),
         ERROR,
         "Demographic characterization of neighborhood",
         "Remove demographic descriptors."),
    ],
    "color": [
        (_word(r"(?:light|fair|dark)[-\s]?skinned"),
         ERROR,
         "Preference based on skin color",
         "Remove references to skin color."),
    ],
    "national_origin": [
        (_word(r"(?:americans?|mexicans?|chinese|indians?|filipinos?|koreans?)\s+only"),
         ERROR,
         "National origin preference",
         "Remove references to national origin."),
        (_word(r"no\s+(?:foreigners?|immigrants?)"),
         ERROR,
         "Exclusion based on national origin",
         "Remove; cannot exclude by national origin."),
        (_word(r"english[-\s]?speaking\s+(?:only|tenants?|family|families|household)"),
         ERROR,
         "Language requirement acts as a proxy for national origin",
         "Remove. You may describe languages staff speak but not require them of tenants."),
        (_word(r"must\s+speak\s+english"),
         ERROR,
         "Language requirement acts as a proxy for national origin",
         "Remove language requirements."),
    ],
    "religion": [
        (_word(r"(?:christian|catholic|jewish|muslim|hindu|buddhist|mormon|lds)\s+(?:only|community|neighborhood|family|families|tenants?|buyers?|household)"),
         ERROR,
         "Religious preference",
         "Remove references to religion."),
        (_word(r"(?:no|not)\s+(?:christians?|catholics?|jews?|jewish|muslims?|hindus?)"),
         ERROR,
         "Religious exclusion",
         "Remove; cannot exclude by religion."),
        (_word(r"good\s+christian\s+(?:home|family|values|neighborhood)"),
         ERROR,
         "Implicit religious preference",
         "Remove religious descriptors."),
        (_word(r"(?:near|close\s+to|walking\s+distance\s+to|walk\s+to)(?:\s+(?:the|a|our))?\s+(?:church|synagogue|mosque|temple|cathedral|shul)"),
         WARNING,
         "Religious landmark reference may steer by religion. Review for directional use only.",
         "Prefer neutral landmarks (parks, transit). If retained, ensure you list landmarks of multiple faiths or none."),
        (_word(r"church[-\s]?going"),
         ERROR,
         "Preference for religious observance",
         "Remove religious descriptors of ideal tenant."),
    ],
    "sex": [
        (_word(r"(?:female|male|woman|man|women|men|ladies|gentlemen)\s+only"),
         ERROR,
         "Gender preference",
         "Remove gender preferences. Shared-living exceptions require legal review."),
        (_word(r"no\s+(?:men|women|males|females)"),
         ERROR,
         "Gender exclusion",
         "Remove gender exclusions."),
        (_word(r"(?:perfect|ideal|great)\s+for\s+(?:bachelors?|bachelorettes?)"),
         WARNING,
         "Implied gender or marital preference",
         "Describe the property instead of the ideal occupant."),
        (_word(r"(?:straight|gay|lesbian|lgbt|lgbtq\+?)\s+(?:only|couples?\s+only|friendly\s+only)"),
         ERROR,
         "Sexual-orientation preference",
         "Remove references to sexual orientation."),
    ],
    "familial_status": [
        (_word(r"no\s+(?:children|kids|infants|toddlers|minors)"),
         ERROR,
         "Exclusion of families with children",
         "Remove. HUD-designated 55+ senior housing has a narrow exemption; document eligibility separately."),
        (_word(r"adults?\s+only"),
         ERROR,
         "Exclusion of families with children",
         "Remove unless the property is a certified 55+ or 62+ senior community."),
        (_word(r"child(?:ren)?[-\s]?free"),
         ERROR,
         "Exclusion of families with children",
         "Remove."),
        (_word(r"(?:perfect|ideal|great)\s+for\s+(?:singles?|couples?|young\s+professionals?|empty\s+nesters?)"),
         WARNING,
         "Language describing ideal occupant can exclude families",
         "Describe the home's features rather than the ideal occupant."),
        (_word(r"no\s+more\s+than\s+\d+\s+(?:children|kids|minors)"),
         ERROR,
         "Child-specific occupancy limit is familial-status discrimination",
         "Remove. Apply neutral occupancy limits (per HUD: ~2 persons per bedroom)."),
        (_word(r"no\s+more\s+than\s+\d+\s+(?:people|occupants|adults)"),
         WARNING,
         "Occupancy limits must follow the ~2-per-bedroom HUD guideline and local code",
         "Verify compliance with HUD occupancy standards and state/local law."),
        (_word(r"mature\s+(?:adults?|tenants?|couples?)"),
         WARNING,
         "'Mature' can imply age or familial-status preference",
         "Remove or rephrase around property features."),
        (_word(r"not\s+(?:suitable|appropriate)\s+for\s+(?:children|kids|families)"),
         ERROR,
         "Exclusion of families",
         "Remove. State physical facts (e.g., 'no fence') without excluding families."),
        (_word(r"bachelor\s+pad"),
         WARNING,
         "'Bachelor pad' implies gender and familial status",
         "Describe layout instead (e.g., 'open studio')."),
    ],
    "disability": [
        (_word(r"no\s+(?:wheelchairs?|handicapped|disabled)"),
         ERROR,
         "Exclusion of people with disabilities",
         "Remove. Describe physical access honestly (e.g., 'third-floor walk-up')."),
        (_word(r"must\s+be\s+able[-\s]?bodied"),
         ERROR,
         "Requirement targeting disability status",
         "Remove."),
        (_word(r"(?:no|not)\s+(?:suitable|appropriate)\s+for\s+(?:disabled|handicapped|wheelchair)"),
         ERROR,
         "Exclusion of people with disabilities",
         "Remove; describe access factually."),
        (_word(r"no\s+(?:service\s+animals?|emotional\s+support\s+animals?|esa)"),
         ERROR,
         "Service / assistance animals must be allowed as a reasonable accommodation",
         "Remove. Pet policies can still prohibit ordinary pets."),
        (_word(r"walk[-\s]?up\s+only"),
         WARNING,
         "May discourage buyers with mobility disabilities",
         "Describe the property factually (e.g., 'third-floor unit, stairs only')."),
        (_word(r"sane\s+(?:tenants?|only)"),
         ERROR,
         "Mental-disability exclusion",
         "Remove."),
    ],
    "source_of_income": [
        (_word(r"no\s+section\s*8"),
         ERROR,
         "Source-of-income discrimination (illegal in many jurisdictions)",
         "Remove. Many state and local laws treat housing vouchers as a protected source of income."),
        (_word(r"no\s+(?:vouchers?|housing\s+assistance|hasa|hacla)"),
         ERROR,
         "Source-of-income discrimination",
         "Remove."),
        (_word(r"(?:cash|wage|w2|w-2)\s+only"),
         WARNING,
         "May act as a proxy for source-of-income discrimination",
         "State the financial qualification (e.g., 'income 3x rent') instead."),
    ],
    "age": [
        (_word(r"(?:young|youthful)\s+(?:professionals?|couples?|tenants?|buyers?)"),
         WARNING,
         "Age-based preference",
         "Describe the property rather than the preferred age group."),
        (_word(r"must\s+be\s+(?:over|under)\s+\d+"),
         ERROR,
         "Explicit age restriction",
         "Remove. 55+ / 62+ senior communities have narrow HUD exemptions; document them separately."),
        (_word(r"(?:no|not)\s+seniors?"),
         ERROR,
         "Age-based exclusion",
         "Remove."),
    ],
    "steering": [
        # Language that HUD has historically cited in steering cases but which
        # is not automatically unlawful. Reported as info so editors learn the
        # context without blocking the listing.
        (_word(r"safe\s+neighborhood"),
         INFO,
         "'Safe' has been used to steer by race. Prefer specific, verifiable details.",
         "Cite objective facts (gated entry, on-site security) instead."),
        (_word(r"(?:great|good)\s+schools?"),
         INFO,
         "School-quality language has been used to steer. Keep factual and cite the district.",
         "Name the district and note that school quality is the buyer's responsibility to verify."),
        (_word(r"exclusive\s+(?:community|neighborhood)"),
         INFO,
         "'Exclusive' can imply discriminatory exclusion",
         "Prefer 'private' or describe amenities."),
        (_word(r"desirable\s+(?:neighborhood|area|community)"),
         INFO,
         "'Desirable' is vague and has a history of steering use",
         "Describe concrete amenities instead."),
    ],
}


# Compile once at import for fast ``check_listing`` calls.
_COMPILED: dict[str, list[tuple[re.Pattern[str], str, str, str]]] = {
    category: [
        (re.compile(pattern, flags=re.IGNORECASE), severity, message, suggestion)
        for pattern, severity, message, suggestion in specs
    ]
    for category, specs in _PATTERN_LIBRARY.items()
}


class ComplianceChecker:
    """Scan listing text for Fair Housing Act compliance issues."""

    ERROR = ERROR
    WARNING = WARNING
    INFO = INFO

    def __init__(self, extra_patterns: dict[str, list[PatternSpec]] | None = None):
        """Create a checker. ``extra_patterns`` lets callers add state-specific
        rules without forking the library. Keys are category names; values are
        the same (regex, severity, message, suggestion) tuples as the built-in
        library.
        """
        self._compiled = {k: list(v) for k, v in _COMPILED.items()}
        if extra_patterns:
            for category, specs in extra_patterns.items():
                compiled = [
                    (re.compile(p, flags=re.IGNORECASE), s, m, sug)
                    for p, s, m, sug in specs
                ]
                self._compiled.setdefault(category, []).extend(compiled)

    @property
    def categories(self) -> tuple[str, ...]:
        return tuple(self._compiled.keys())

    def check_listing(self, text: str | None) -> ComplianceReport:
        """Return a :class:`ComplianceReport` for ``text``.

        A listing is considered *compliant* when no ``error``-severity
        violations are detected. ``warning`` and ``info`` findings do not
        block publication but are surfaced for reviewer awareness.
        """
        if not text:
            return ComplianceReport(compliant=True)

        seen: set[tuple[str, int, int]] = set()
        violations: list[Violation] = []
        for category, specs in self._compiled.items():
            for regex, severity, message, suggestion in specs:
                for match in regex.finditer(text):
                    key = (regex.pattern, match.start(), match.end())
                    if key in seen:
                        continue
                    seen.add(key)
                    violations.append(
                        Violation(
                            category=category,
                            severity=severity,
                            pattern=regex.pattern,
                            matched_text=match.group(0),
                            start=match.start(),
                            end=match.end(),
                            message=f"{message} (matched '{match.group(0)}')",
                            suggestion=suggestion,
                        )
                    )

        severity_rank = {ERROR: 0, WARNING: 1, INFO: 2}
        violations.sort(key=lambda v: (severity_rank.get(v.severity, 99), v.start))

        errors = sum(1 for v in violations if v.severity == ERROR)
        warnings = sum(1 for v in violations if v.severity == WARNING)
        infos = sum(1 for v in violations if v.severity == INFO)

        return ComplianceReport(
            compliant=errors == 0,
            violations=violations,
            error_count=errors,
            warning_count=warnings,
            info_count=infos,
        )

    def check_many(self, texts: Iterable[str]) -> list[ComplianceReport]:
        return [self.check_listing(t) for t in texts]

    def format_report(self, report: ComplianceReport) -> str:
        """Human-readable report for logs / UI."""
        if report.compliant and not report.violations:
            return "Compliant: no Fair Housing issues detected."
        lines = [
            f"Compliant: {report.compliant}",
            f"Errors: {report.error_count}, Warnings: {report.warning_count}, Info: {report.info_count}",
        ]
        for v in report.violations:
            lines.append(
                f"  [{v.severity.upper():7}] {v.category}: '{v.matched_text}' -> {v.suggestion}"
            )
        return "\n".join(lines)


__all__ = [
    "ComplianceChecker",
    "ComplianceReport",
    "Violation",
    "ERROR",
    "WARNING",
    "INFO",
]
