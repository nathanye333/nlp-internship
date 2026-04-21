"""End-to-end example: Fair Housing compliance in a listing submission flow.

This script shows how ``ComplianceChecker`` slots into the listing publishing
pipeline. Run it directly for a live demonstration:

    python scripts/listing_submission_example.py

The simulated flow is:

    agent drafts listing
        -> pre-submit compliance check
        -> if errors: block and return to agent with suggestions
        -> if warnings: route to human reviewer queue
        -> otherwise: publish and log any info-level findings

Real-world systems should persist the ``ComplianceReport`` alongside the
listing (for audit) and re-run the checker after any edit.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.compliance_checker import (  # noqa: E402
    ComplianceChecker,
    ComplianceReport,
    ERROR,
    WARNING,
)


Status = Literal["draft", "blocked", "pending_review", "published"]


@dataclass
class Listing:
    listing_id: str
    address: str
    price_usd: int
    description: str
    status: Status = "draft"
    compliance_report: ComplianceReport | None = None
    audit_log: list[str] = field(default_factory=list)

    def log(self, message: str) -> None:
        self.audit_log.append(f"{datetime.utcnow().isoformat(timespec='seconds')}Z | {message}")


class ListingSubmissionError(Exception):
    """Raised when a listing cannot advance to the next workflow state."""


class ListingSubmissionService:
    """Minimal orchestrator that enforces Fair Housing checks at submit time."""

    def __init__(self, checker: ComplianceChecker | None = None):
        self.checker = checker or ComplianceChecker()
        self.review_queue: list[Listing] = []
        self.published: list[Listing] = []

    def submit(self, listing: Listing) -> Listing:
        report = self.checker.check_listing(listing.description)
        listing.compliance_report = report
        listing.log(
            f"Compliance scan: errors={report.error_count} "
            f"warnings={report.warning_count} info={report.info_count}"
        )

        if report.error_count > 0:
            listing.status = "blocked"
            details = "; ".join(
                f"[{v.severity}] {v.category}: {v.matched_text} -> {v.suggestion}"
                for v in report.violations
                if v.severity == ERROR
            )
            listing.log(f"Blocked: {details}")
            raise ListingSubmissionError(
                f"Listing {listing.listing_id} blocked by Fair Housing compliance: {details}"
            )

        if report.warning_count > 0:
            listing.status = "pending_review"
            self.review_queue.append(listing)
            listing.log(
                "Sent to human review: "
                + "; ".join(
                    f"{v.category}: {v.matched_text}"
                    for v in report.violations
                    if v.severity == WARNING
                )
            )
            return listing

        listing.status = "published"
        self.published.append(listing)
        if report.violations:
            listing.log(
                "Published with info-level notes: "
                + "; ".join(f"{v.category}: {v.matched_text}" for v in report.violations)
            )
        else:
            listing.log("Published (clean)")
        return listing

    def reviewer_approve(self, listing: Listing, reviewer: str) -> Listing:
        if listing.status != "pending_review":
            raise ListingSubmissionError(
                f"Listing {listing.listing_id} is not in pending_review state"
            )
        listing.status = "published"
        self.review_queue.remove(listing)
        self.published.append(listing)
        listing.log(f"Approved by {reviewer}")
        return listing


def _demo() -> None:
    service = ListingSubmissionService()

    samples = [
        Listing(
            listing_id="L-001",
            address="123 Oak St, Austin, TX",
            price_usd=625_000,
            description=(
                "3 bed, 2 bath craftsman bungalow with hardwood floors, "
                "a renovated kitchen, and a large fenced backyard. Close to "
                "transit and the farmers market."
            ),
        ),
        Listing(
            listing_id="L-002",
            address="4 Maple Ct, Seattle, WA",
            price_usd=780_000,
            description=(
                "Two-bedroom condo perfect for young professionals, walk to the "
                "light rail and great schools."
            ),
        ),
        Listing(
            listing_id="L-003",
            address="99 Palm Dr, Irvine, CA",
            price_usd=1_250_000,
            description=(
                "Luxury 4-bed home in a quiet Christian community. Adults only, "
                "no children. No Section 8."
            ),
        ),
    ]

    for listing in samples:
        print("=" * 72)
        print(f"{listing.listing_id} {listing.address}")
        try:
            service.submit(listing)
        except ListingSubmissionError as exc:
            print(f"  BLOCKED: {exc}")
        print(f"  status = {listing.status}")
        if listing.compliance_report:
            print(service.checker.format_report(listing.compliance_report))
        for entry in listing.audit_log:
            print("  log:", entry)

    print()
    print(f"Published: {[l.listing_id for l in service.published]}")
    print(f"Pending review: {[l.listing_id for l in service.review_queue]}")


if __name__ == "__main__":
    _demo()
