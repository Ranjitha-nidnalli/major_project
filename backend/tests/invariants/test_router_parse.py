"""Tests for services/router_parse.py."""
from services.router_parse import parse_router_category


def test_plain_category():
    assert parse_router_category("pest") == "pest"
    assert parse_router_category(" Disease\n") == "disease"


def test_empty_reply_is_a_failure_not_general():
    """The live defect: reasoning used the whole budget, content was ''."""
    assert parse_router_category("") is None
    assert parse_router_category(None) is None


def test_old_pipe_format_and_punctuation_tolerated():
    assert parse_router_category("fertilizer | sugarcane fertilizer splits") == "fertilizer"
    assert parse_router_category("'price'.") == "price"


def test_unknown_category_rejected():
    assert parse_router_category("irrigation") is None
    assert parse_router_category("pest control advice") is None
