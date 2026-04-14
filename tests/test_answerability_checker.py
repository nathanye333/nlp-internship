import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from scripts.answerability_checker import AnswerabilityChecker
from scripts.query_parser import QueryParser


class StubValidator:
    def validate_query(self, filters: dict) -> tuple[bool, list[str]]:
        if filters.get("price_max", 0) and filters.get("price_max", 0) < 100000:
            return False, ["Price maximum is unrealistically low"]
        return True, []


class _NullFrame:
    def __init__(self, size: int, all_null: bool):
        self._size = size
        self._all_null = all_null

    def __len__(self) -> int:
        return self._size

    def isnull(self):
        return self

    def all(self):
        return self

    def __bool__(self) -> bool:
        return self._all_null


def _checker() -> AnswerabilityChecker:
    taxonomy = {"cities": ["Irvine", "Austin", "Seattle"]}
    return AnswerabilityChecker(taxonomy=taxonomy, schema_validator=StubValidator(), parser=QueryParser())


def test_pre_query_rejects_out_of_domain_questions():
    result = _checker().check_pre_query("What is the weather tomorrow?")
    assert result.answerable is False
    assert result.reason == "Out of domain query"
    assert result.details


def test_pre_query_rejects_unsupported_city():
    result = _checker().check_pre_query("Find 3 bed homes in Atlantis under 900k")
    assert result.answerable is False
    assert result.reason == "Unsupported filter values"
    assert "City" in " ".join(result.details)


def test_pre_query_accepts_valid_real_estate_query():
    result = _checker().check_pre_query("Find 3 bed homes in Irvine under 900k with pool")
    assert result.answerable is True
    assert result.reason == "Query is answerable"


def test_post_query_catches_empty_and_null_results():
    checker = _checker()
    empty = checker.check_post_query("Find homes in Irvine", _NullFrame(size=0, all_null=False))
    assert empty.answerable is False
    assert empty.reason == "No listings match criteria"

    null_only = checker.check_post_query("Find homes in Irvine", _NullFrame(size=4, all_null=True))
    assert null_only.answerable is False
    assert null_only.reason == "Results contain no meaningful values"
