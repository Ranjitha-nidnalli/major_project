"""Tests for services/judge_parse.py (TODO #48)."""
from services.judge_parse import parse_judge_score


def test_plain_score():
    assert parse_judge_score("1.0") == 1.0
    assert parse_judge_score(" 0.35\n") == 0.35
    assert parse_judge_score("0") == 0.0


def test_empty_reply_is_a_failure_not_a_zero_score():
    """The live defect: reasoning used the whole budget, content was ''."""
    assert parse_judge_score("") is None
    assert parse_judge_score(None) is None
    assert parse_judge_score("   ") is None


def test_single_number_in_short_wrapper_is_accepted():
    assert parse_judge_score("Score: 0.8") == 0.8


def test_out_of_range_is_rejected():
    """The old fallback took the first number anywhere, so "Score: 10"
    became 10.0 and passed the 0.5 gate."""
    assert parse_judge_score("10") is None
    assert parse_judge_score("Score: 10") is None


def test_ambiguous_multiple_numbers_rejected():
    assert parse_judge_score("between 0.2 and 0.9") is None


def test_non_numeric_rejected():
    assert parse_judge_score("faithful") is None
