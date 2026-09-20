"""
Unit tests for backend/services/gating.py (Task 2).

Pure-function tests: no Qdrant, no embeddings, no LLM. These exercise the
gate's decision logic in isolation, per ARCHITECTURE.md Section 21's
replacement design for the old RRF-score-based gate.
"""
import pytest

from services.gating import GatingConfig, GatingDecision, decide


def test_low_relevance_always_refuses_regardless_of_category():
    """The single hard invariant: below-threshold relevance always refuses,
    for every category, safety-critical or not."""
    config = GatingConfig(default_threshold=0.5, safety_critical_threshold=0.5)
    for category in ["general", "pest", "disease", "fertilizer", "soil", None, "unknown_category"]:
        decision = decide(relevance=0.01, category=category, config=config)
        assert decision.should_answer is False, f"expected refusal for category={category}"


def test_none_relevance_refuses():
    decision = decide(relevance=None, category="general")
    assert decision.should_answer is False
    assert decision.reason == "no_relevance_score"


def test_high_relevance_non_safety_category_answers():
    config = GatingConfig(default_threshold=0.3, safety_critical_threshold=0.6)
    decision = decide(relevance=0.9, category="general", config=config)
    assert decision.should_answer is True


def test_safety_critical_category_uses_stricter_threshold():
    """A relevance score that would pass for 'general' must still refuse for
    a safety-critical category if it's below the stricter threshold."""
    config = GatingConfig(default_threshold=0.3, safety_critical_threshold=0.6)

    general_decision = decide(relevance=0.4, category="general", config=config)
    pest_decision = decide(relevance=0.4, category="pest", config=config)

    assert general_decision.should_answer is True
    assert pest_decision.should_answer is False


def test_per_category_threshold_override():
    config = GatingConfig(
        default_threshold=0.3,
        safety_critical_threshold=0.6,
        category_thresholds={"soil": 0.8},
    )
    decision = decide(relevance=0.5, category="soil", config=config)
    assert decision.should_answer is False
    assert decision.threshold_used == 0.8


def test_category_never_used_to_exclude_retrieved_documents():
    """Category only ever changes the threshold value; it must never be
    capable of forcing a refusal independent of relevance. This guards
    against reintroducing a category hard-filter inside the gate itself."""
    config = GatingConfig(default_threshold=0.1, safety_critical_threshold=0.1)
    for category in ["pest", "disease", "fertilizer", "general", "made_up_category"]:
        decision = decide(relevance=0.99, category=category, config=config)
        assert decision.should_answer is True


def test_entity_match_hook_not_called_when_none():
    """Phase 1 default: no entity check at all, not even a silent pass."""
    calls = []

    def hook(q, r):
        calls.append((q, r))
        return True

    config = GatingConfig(default_threshold=0.1, entity_match_hook=None)
    decide(relevance=0.9, category="general", config=config)
    assert calls == []


def test_entity_match_hook_used_when_provided():
    """Documents the Phase 4 extension point without implementing it."""

    def always_mismatch(query_entities, retrieved_entities):
        return False

    config = GatingConfig(default_threshold=0.1, entity_match_hook=always_mismatch)
    decision = decide(relevance=0.99, category="pest", config=config)
    assert decision.should_answer is False
    assert decision.reason == "entity_mismatch"


def test_decision_is_a_frozen_dataclass_pure_result():
    decision = decide(relevance=0.9, category="general")
    assert isinstance(decision, GatingDecision)
    with pytest.raises(Exception):
        decision.should_answer = False  # frozen -- must not be mutable
