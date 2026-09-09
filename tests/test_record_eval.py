"""Unit tests for scripts/record_eval.py's failure_class_distribution().

Denominator is failed episodes, not all episodes -- both are defensible (see
the function's own docstring), which is exactly why this needs pinning down
with a test: a later "simplification" could silently swap the denominator
back to all-episodes without anyone noticing the numbers changed meaning.
"""

import re

from record_eval import failure_class_distribution

from failure_taxonomy_harness.classifier import FailureClass


def _episode(success: bool, failure_class: FailureClass | None = None) -> dict:
    return {"success": success, "failure_class": failure_class}


def _count_for(output: str, cls: FailureClass) -> int:
    match = re.search(rf"^\s*{re.escape(cls.value)}\s+(\d+)\s+\(", output, re.MULTILINE)
    assert match, f"{cls.value} row not found in:\n{output}"
    return int(match.group(1))


def test_denominator_is_failed_not_all_episodes():
    """Percentages must read against the failed count, not the total count."""
    records = [
        _episode(True),
        _episode(True),
        _episode(False, FailureClass.NEVER_PICKED_KEY),
    ]
    output = failure_class_distribution(records)
    assert "1 failed / 3 total" in output
    assert re.search(r"NEVER_PICKED_KEY\s+1\s+\(100\.0%\)", output)


def test_all_four_classes_listed_even_at_zero():
    """A class the run never produced must still appear, at count 0 --
    not disappear from the report."""
    records = [_episode(False, FailureClass.NEVER_PICKED_KEY)]
    output = failure_class_distribution(records)

    for cls in FailureClass:
        if cls is FailureClass.UNCLASSIFIED:
            assert cls.value not in output
        else:
            assert _count_for(output, cls) >= 0  # present and parseable


def test_no_failures_reports_n_a_without_crashing():
    """All-success batch: no division by zero, percentages read n/a."""
    records = [_episode(True), _episode(True)]
    output = failure_class_distribution(records)
    assert "0 failed / 2 total" in output
    assert "n/a" in output


def test_counts_and_percentages_match_multiple_classes():
    records = [
        _episode(True),
        _episode(False, FailureClass.NEVER_PICKED_KEY),
        _episode(False, FailureClass.NEVER_PICKED_KEY),
        _episode(False, FailureClass.NEVER_PICKED_KEY),
        _episode(False, FailureClass.NEVER_UNLOCKED_DOOR),
    ]
    output = failure_class_distribution(records)

    assert "4 failed / 5 total" in output
    assert _count_for(output, FailureClass.NEVER_PICKED_KEY) == 3
    assert _count_for(output, FailureClass.NEVER_UNLOCKED_DOOR) == 1
    assert _count_for(output, FailureClass.NEVER_CROSSED_DOOR) == 0
    assert _count_for(output, FailureClass.NEVER_REACHED_GOAL) == 0
    assert re.search(r"NEVER_PICKED_KEY\s+3\s+\(75\.0%\)", output)
    assert re.search(r"NEVER_UNLOCKED_DOOR\s+1\s+\(25\.0%\)", output)
