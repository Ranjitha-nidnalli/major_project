"""
Invariants for the tiered answer policy (services/tiers.py, story K-007).
Pure tests: fake fact and RAG functions, no LLM, database or embeddings.
"""
import asyncio

from services import tiers
from services.tiers import (
    TIER_A, TIER_B, TIER_C, answer_tiered, has_dose_like_numbers,
    mentions_dose_or_pesticide, tier_b_blockers, KCC_NUMBER,
)

# Wording used below (Kannada): "when to plant sugarcane", "how much fertiliser per acre".
GENERAL_Q = "ಕಬ್ಬು ಬೆಳೆಯಲು ಯಾವ ಮಣ್ಣು ಸೂಕ್ತ"
DOSE_Q = "ಎಕರೆಗೆ ಎಷ್ಟು ಗೊಬ್ಬರ ಹಾಕಬೇಕು"
PEST_Q = "ಕಬ್ಬಿಗೆ ಬಂದ ಹುಳು ನಿಯಂತ್ರಣ ಹೇಗೆ"

NO_MATCH = {"subjects": [], "intents": [], "organism_word": False, "other_crop": False, "region": None}


def facts_answered():
    async def fn():
        return {"status": "answered", "answer": "RECORD TEXT", "record_ids": ["r1"],
                "citations": ["UAS-B 2025, p. 165"], "match": {**NO_MATCH, "intents": ["variety"]}}
    return fn


def facts_refused(match=None, status="refused_out_of_scope"):
    async def fn():
        return {"status": status, "answer": f"REFUSAL {KCC_NUMBER}", "record_ids": [],
                "citations": [], "match": match or dict(NO_MATCH)}
    return fn


def rag_must_not_run():
    async def fn():
        raise AssertionError("Tier B must not run")
    return fn


def rag_returns(answer="ಕಬ್ಬಿಗೆ ಕೆಂಪು ಮಣ್ಣು ಸೂಕ್ತ.", refused=False, category="general",
                payloads=None):
    async def fn():
        return {"answer": answer, "refused": refused, "category": category,
                "source_payloads": payloads if payloads is not None
                else [{"text": "t", "category": "general", "source": "UAS-B 2025", "page": "12"}]}
    return fn


def run(query, facts_fn, rag_fn, enabled=True):
    return asyncio.run(answer_tiered(query, facts_fn=facts_fn, rag_fn=rag_fn, tier_b_enabled=enabled))


# ---- Tier A is first and cannot be blocked -------------------------------------

def test_tier_a_answers_without_calling_rag():
    r = run(GENERAL_Q, facts_answered(), rag_must_not_run())
    assert r["tier"] == TIER_A and r["answer"] == "RECORD TEXT"


def test_dose_wording_cannot_make_a_confirmed_record_unreachable():
    # The classifier-style checks run only AFTER Tier A refused.
    assert mentions_dose_or_pesticide(DOSE_Q)
    r = run(DOSE_Q, facts_answered(), rag_must_not_run())
    assert r["tier"] == TIER_A


# ---- dose / pesticide / rate questions never reach Tier B -----------------------

def test_dose_question_goes_to_tier_c_even_when_tier_b_is_enabled():
    r = run(DOSE_Q, facts_refused(), rag_must_not_run(), enabled=True)
    assert r["tier"] == TIER_C and "dose_or_pesticide_wording" in r["tier_reason"]


def test_english_dose_wording_is_blocked():
    for q in ("what dose of urea per acre", "which pesticide should I spray", "how much DAP"):
        assert mentions_dose_or_pesticide(q), q
        assert run(q, facts_refused(), rag_must_not_run())["tier"] == TIER_C


def test_pest_or_organism_questions_go_to_tier_c():
    for match in ({**NO_MATCH, "subjects": ["termite"]}, {**NO_MATCH, "organism_word": True}):
        r = run(PEST_Q, facts_refused(match, "refused_unknown_pest"), rag_must_not_run())
        assert r["tier"] == TIER_C


def test_known_topic_without_confirmed_record_does_not_fall_back_to_retrieval():
    r = run(GENERAL_Q, facts_refused({**NO_MATCH, "intents": ["fertilizer"]}, "refused_not_verified"),
            rag_must_not_run())
    assert r["tier"] == TIER_C and "known_topic" in r["tier_reason"]


def test_other_crop_goes_to_tier_c():
    r = run(GENERAL_Q, facts_refused({**NO_MATCH, "other_crop": True}), rag_must_not_run())
    assert r["tier"] == TIER_C


def test_router_category_can_only_tighten():
    assert tier_b_blockers(GENERAL_Q, NO_MATCH, None) == []
    for cat in ("pest", "disease", "fertilizer", "price"):
        assert tier_b_blockers(GENERAL_Q, NO_MATCH, cat)
    assert tier_b_blockers(GENERAL_Q, NO_MATCH, "general") == []


# ---- Tier B is off by default and strict when on --------------------------------

def test_tier_b_disabled_by_default_goes_to_tier_c():
    r = asyncio.run(answer_tiered(GENERAL_Q, facts_fn=facts_refused(), rag_fn=rag_must_not_run()))
    assert r["tier"] == TIER_C and r["tier_reason"] == "tier_b_disabled"


def test_tier_b_answers_general_question_with_citation_and_kcc_line():
    r = run(GENERAL_Q, facts_refused(), rag_returns())
    assert r["tier"] == TIER_B
    assert r["citations"] == ["UAS-B 2025, p. 12"]
    assert "UAS-B 2025, p. 12" in r["answer"] and KCC_NUMBER in r["answer"]


def test_tier_b_without_a_citable_source_is_refused():
    # Today's corpus chunks carry no `source` field, so this is the real behaviour.
    r = run(GENERAL_Q, facts_refused(), rag_returns(payloads=[{"text": "t", "category": "general"}]))
    assert r["tier"] == TIER_C and "no_citable_source" in r["tier_reason"]
    r = run(GENERAL_Q, facts_refused(), rag_returns(payloads=[]))
    assert r["tier"] == TIER_C


def test_partly_cited_answer_is_refused():
    payloads = [{"text": "a", "source": "UAS-B 2025"}, {"text": "b"}]
    assert run(GENERAL_Q, facts_refused(), rag_returns(payloads=payloads))["tier"] == TIER_C


def test_tier_b_answer_with_a_dose_like_number_becomes_tier_c():
    r = run(GENERAL_Q, facts_refused(), rag_returns(answer="ಎಕರೆಗೆ 50 ಕೆ.ಜಿ ಹಾಕಿ"))
    assert r["tier"] == TIER_C and "dose_like_number" in r["tier_reason"]


def test_tier_b_rejected_when_retrieval_shows_a_pest_or_fertilizer_card():
    for cat in ("pest", "disease", "fertilizer", "price"):
        assert run(GENERAL_Q, facts_refused(), rag_returns(category=cat))["tier"] == TIER_C


def test_tier_b_refusal_or_error_is_tier_c():
    assert run(GENERAL_Q, facts_refused(), rag_returns(refused=True))["tier"] == TIER_C

    async def boom():
        raise RuntimeError("llm down")
    r = run(GENERAL_Q, facts_refused(), boom)
    assert r["tier"] == TIER_C and r["tier_reason"].startswith("tier_b_error")


def test_tier_c_always_has_the_kcc_number():
    async def facts():
        return {"status": "refused_out_of_scope", "answer": "no number here", "record_ids": [],
                "citations": [], "match": dict(NO_MATCH)}
    r = run(DOSE_Q, facts, rag_must_not_run())
    assert KCC_NUMBER in r["answer"]


# ---- dose-like number detector ---------------------------------------------------

def test_dose_like_numbers_detected():
    for text in ("2 kg", "0.5 ml per litre", "50%", "೨ ಕೆ.ಜಿ", "5 ಗ್ರಾಂ", "10-12 ಲೀಟರ್", "250 ppm"):
        assert has_dose_like_numbers(text), text


def test_plain_numbers_and_months_are_not_dose_like():
    for text in ("12-14 ತಿಂಗಳು", "5 lakh farmers", "ಜೂನ್-ಆಗಸ್ಟ್ ನಾಟಿ", "Co 62175"):
        assert not has_dose_like_numbers(text), text


def test_general_question_has_no_dose_wording():
    assert mentions_dose_or_pesticide(GENERAL_Q) is None
    assert mentions_dose_or_pesticide("when is the best season to plant") is None


def test_module_stays_pure():
    import services.tiers as t
    src = open(t.__file__, encoding="utf-8").read()
    for heavy in ("torch", "qdrant", "llm_client", "chat_db", "vector_db", "rag_service"):
        assert f"import {heavy}" not in src and f"from {heavy}" not in src
