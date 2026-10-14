"""
services/canonicalizer.py and the topic_fallback hook in fact_answer.answer:
the "LLM only understands the question" middle ground.

The model is faked here; these tests pin what the model is ALLOWED to do,
whatever it outputs: pick topic ids only, never a pest/disease or dose, and
only when the keyword matcher found nothing. No network, no LLM.
"""
import asyncio
import csv
import json

import pytest

from services.canonicalizer import canonicalize, json_schema, parse_topics, NONE_LABEL
from services.fact_answer import Knowledge, answer, record_hash, topic_labels

K = Knowledge.load(reviewed_sheet="__no_such_file__")


def _confirm(tmp_path, rows):
    """Write a review sheet confirming the given {id: overrides} rows (as in test_fact_answer)."""
    path = tmp_path / "reviewed.csv"
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["id", "REVIEW: status (CONFIRMED / WRONG / UNSURE)",
                    "REVIEW: pesticide currently registered? (yes / no / n.a.)", "record_hash (do not edit)"])
        for rid, over in rows.items():
            rec = K.records[rid]
            w.writerow([rid, "CONFIRMED", over.get("registered", "yes"), record_hash(rec["_source"], rec)])
    return Knowledge.load(reviewed_sheet=str(path))


LABELS = topic_labels(K)
UNMATCHED = "ಬೆಳೆ ಚೆನ್ನಾಗಿ ಬರಲು ಏನು ಮಾಡಬೇಕು?"   # no keyword the matcher knows


def _fixed(topics):
    calls = []

    def fallback(query):
        calls.append(query)
        return topics
    fallback.calls = calls
    return fallback


# --- the closed label set --------------------------------------------------

def test_labels_exclude_every_pest_and_disease_subject():
    assert not set(LABELS) & set(K.intents["subjects"])


def test_no_label_reaches_a_pest_or_disease_record():
    for label in LABELS:
        for rid in K.intents["intents"][label]["records"]:
            assert K.records[rid]["topic"] not in ("pest", "disease"), (label, rid)


def test_schema_is_an_enum_over_labels_plus_none():
    enum = json_schema(LABELS)["properties"]["topics"]["items"]["enum"]
    assert set(enum) == set(LABELS) | {NONE_LABEL}


# --- strict parsing ----------------------------------------------------------

@pytest.mark.parametrize("reply", [
    None, "", "seed_rate", "{bad json", '["seed_rate"]',
    '{"topics": []}', '{"topics": ["none"]}', '{"topics": ["seed_rate", "none"]}',
    '{"topics": ["early_shoot_borer"]}',            # a subject id, not a topic
    '{"topics": ["seed_rate"], "dose": "10 kg"}',   # extra keys
    '{"topics": "seed_rate"}',
    'Sure! {"topics": ["seed_rate"]}',
])
def test_parse_rejects_anything_but_known_topics(reply):
    assert parse_topics(reply, LABELS) == []


def test_parse_accepts_known_topics():
    assert parse_topics('{"topics": ["seed_rate", "irrigation"]}', LABELS) == ["seed_rate", "irrigation"]


def test_canonicalize_fails_closed_on_exception():
    async def boom(messages, json_schema):
        raise RuntimeError("network down")
    assert asyncio.run(canonicalize(K, LABELS, UNMATCHED, boom)) == []


def test_canonicalize_sends_the_schema():
    seen = {}

    async def fake(messages, json_schema):
        seen["schema"] = json_schema
        return json.dumps({"topics": ["seed_rate"]})
    assert asyncio.run(canonicalize(K, LABELS, UNMATCHED, fake)) == ["seed_rate"]
    assert seen["schema"] == json_schema(LABELS)


# --- when the fallback may be consulted ------------------------------------

@pytest.mark.parametrize("query", [
    "ಕಬ್ಬಿನಲ್ಲಿ ಕರಿ ದುಂಬಿ ಬಂದಿದೆ, ಏನು ಔಷಧಿ ಹೊಡೆಯಬೇಕು?",   # unknown pest -> organism word
    "ಟೊಮೆಟೊಗೆ ಗೊಬ್ಬರ ಎಷ್ಟು ಹಾಕಬೇಕು?",                   # other crop
    "ಕಬ್ಬಿನ ಬೆಲೆ ಈಗ ಎಷ್ಟು?",                              # refused topic (price)
    "ಕಬ್ಬಿಗೆ ಎಷ್ಟು ನೀರು ಹಾಯಿಸಬೇಕು?",                      # already matched
])
def test_fallback_not_consulted_when_matcher_found_anything(query):
    fb = _fixed(["seed_rate"])
    answer(K, query, topic_fallback=fb)
    assert fb.calls == []


def test_unknown_pest_stays_refused_even_if_model_would_pick_a_topic():
    r = answer(K, "ಕಬ್ಬಿನಲ್ಲಿ ಕರಿ ದುಂಬಿ ಬಂದಿದೆ, ಏನು ಔಷಧಿ ಹೊಡೆಯಬೇಕು?",
               include_drafts=True, topic_fallback=_fixed(["sett_treatment", "weed"]))
    assert r["status"] == "refused_unknown_pest"


# --- what the fallback can and cannot do ------------------------------------

def test_fallback_topic_answers_from_confirmed_records(tmp_path):
    k = _confirm(tmp_path, {"uasb-seed-rate": {"registered": "n.a."}})
    r = answer(k, UNMATCHED, topic_fallback=_fixed(["seed_rate"]))
    assert r["status"] == "answered"
    assert r["record_ids"] == ["uasb-seed-rate"]
    assert r["match"].get("via_fallback") is True


def test_fallback_cannot_bypass_review():
    r = answer(K, UNMATCHED, topic_fallback=_fixed(["seed_rate"]))   # nothing confirmed
    assert r["status"] == "refused_not_verified"


@pytest.mark.parametrize("picked", [
    ["early_shoot_borer"], ["termite"], ["made_up_topic"], [NONE_LABEL], [],
])
def test_fallback_output_outside_label_set_is_ignored(picked):
    r = answer(K, UNMATCHED, include_drafts=True, topic_fallback=_fixed(picked))
    assert r["status"] == "refused_out_of_scope"
    assert r["record_ids"] == []


def test_fallback_exception_is_a_refusal():
    def boom(query):
        raise RuntimeError("model down")
    r = answer(K, UNMATCHED, include_drafts=True, topic_fallback=boom)
    assert r["status"] == "refused_out_of_scope"


def test_fallback_refused_topic_still_refuses():
    r = answer(K, UNMATCHED, include_drafts=True, topic_fallback=_fixed(["price"]))
    assert r["status"] == "refused_not_verified"


def test_fallback_answers_never_include_pest_or_disease_records():
    for label in LABELS:
        r = answer(K, UNMATCHED, include_drafts=True, topic_fallback=_fixed([label]))
        for rid in r["record_ids"]:
            assert K.records[rid]["topic"] not in ("pest", "disease"), (label, rid)


def test_fallback_applies_composites():
    r = answer(K, UNMATCHED, include_drafts=True, topic_fallback=_fixed(["ratoon", "fertilizer"]))
    assert "ratoon_fertilizer" in r["match"]["intents"]
