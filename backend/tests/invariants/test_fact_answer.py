"""
services/fact_answer.py and knowledge/intents.json: the v2 answer path.

These pin the safety rules: pest/disease doses only through a named pest,
only CONFIRMED records, confirmation voided by any edit, registration
required for products, regions never merged. Pure: reads the committed
knowledge files plus a temporary review sheet, no LLM, no database.
"""
import csv
import json

import pytest

from services.fact_answer import (
    Knowledge, answer, match, normalize, record_hash, KNOWLEDGE_DIR,
)

K = Knowledge.load(reviewed_sheet="__no_such_file__")


def _confirm(tmp_path, rows):
    """Write a review sheet confirming the given {id: overrides} rows."""
    path = tmp_path / "reviewed.csv"
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["id", "REVIEW: status (CONFIRMED / WRONG / UNSURE)",
                    "REVIEW: pesticide currently registered? (yes / no / n.a.)", "record_hash (do not edit)"])
        for rid, over in rows.items():
            rec = K.records[rid]
            w.writerow([rid, over.get("status", "CONFIRMED"), over.get("registered", "yes"),
                        over.get("hash", record_hash(rec["_source"], rec))])
    return Knowledge.load(reviewed_sheet=str(path))


# --- the knowledge files themselves -----------------------------------------

def test_every_referenced_record_exists():
    tables = list(K.intents["subjects"].values()) + list(K.intents["intents"].values())
    missing = {r for t in tables for r in t["records"] if r not in K.records}
    assert not missing


def test_every_record_is_reachable():
    tables = list(K.intents["subjects"].values()) + list(K.intents["intents"].values())
    reachable = {r for t in tables for r in t["records"]}
    assert set(K.records) - reachable == {"uasd-pesticide-safety"}   # appended, not asked for


def test_pest_and_disease_doses_only_reachable_through_a_named_subject():
    for key, intent in K.intents["intents"].items():
        for rid in intent["records"]:
            rec = K.records[rid]
            if rec["topic"] in ("pest", "disease"):
                assert not rec.get("product_kn"), f"intent {key} reaches dose record {rid}"


def test_no_alias_is_a_generic_organism_word():
    generic = {normalize(w) for w in K.intents["organism_words"]}
    for key, subj in K.intents["subjects"].items():
        for alias in subj["aliases"]:
            assert normalize(alias.lstrip("=")) not in generic, (key, alias)


# --- matching -------------------------------------------------------------------

@pytest.mark.parametrize("query", [
    "ಕಬ್ಬಿನ ಹೊಲದಲ್ಲಿ ಕಬ್ಬಿನ ಜಿಗಣೆ ಹುಳು ಬಂದಿದೆ, ಏನು ಮಾಡಬೇಕು?",          # H-38
    "ಕಬ್ಬಿನಲ್ಲಿ ಕಪ್ಪು ದುಂಬಿ ಕೀಟ ಕಂಡುಬಂದರೆ ಯಾವ ಔಷಧಿ ಸಿಂಪಡಿಸಬೇಕು?",     # pest-5
    "ಬಿಳಿ ನೊಣಕ್ಕೆ ಅಟ್ರಾಜಿನ್ ಹಾಕಬಹುದಾ?",                               # unknown pest + weed keyword
    "ಭತ್ತಕ್ಕೆ ಕಂದು ಚುಕ್ಕೆ ರೋಗ ಬಂದರೆ ಏನು ಮಾಡಬೇಕು?",
])
def test_unknown_pest_is_refused_even_with_drafts(query):
    r = answer(K, query, include_drafts=True)
    assert r["status"] == "refused_unknown_pest"
    assert r["record_ids"] == []


def test_longest_phrase_wins_and_consumes():
    m = match(K, "ಹಸಿರೆಲೆ ಗೊಬ್ಬರ ಹೇಗೆ?")
    assert m["intents"] == ["green_manure"]        # not also generic "fertilizer"


def test_neutral_phrase_is_not_an_organism_word():
    assert match(K, "ರೋಗರಹಿತ ಬಿತ್ತನೆ ತುಂಡು ಉಪಚಾರ")["organism_word"] is False


def test_named_variety_overrides_generic_variety():
    assert match(K, "ಸಿ.ಒ 86032 ತಳಿ ಬಗ್ಗೆ ಹೇಳಿ")["intents"] == ["variety_co86032"]


def test_composite_ratoon_fertilizer():
    assert match(K, "ಕೂಳೆ ಬೆಳೆಗೆ ಎಷ್ಟು ಗೊಬ್ಬರ ಹಾಕಬೇಕು?")["intents"] == ["ratoon_fertilizer"]


def test_region_detected_from_district():
    assert match(K, "ಬೆಳಗಾವಿಯಲ್ಲಿ ಗೆದ್ದಲು")["region"] == "north_karnataka"


def test_whole_word_keyword_does_not_match_inside_a_word():
    assert "price" not in match(K, "ತಿನ್ನುವುದರಿಂದ")["intents"]


# --- confirmation rules ---------------------------------------------------------

def test_nothing_is_answered_before_review():
    r = answer(K, "ಗೆದ್ದಲು ಹುಳುವಿಗೆ ಏನು ಮಾಡಬೇಕು?")
    assert r["status"] == "refused_not_verified"


def test_confirmed_record_is_shown_verbatim(tmp_path):
    k = _confirm(tmp_path, {"uasd-termite": {}})
    r = answer(k, "ಗೆದ್ದಲು ಹುಳುವಿಗೆ ಏನು ಮಾಡಬೇಕು?")
    assert r["status"] == "answered"
    assert r["record_ids"] == ["uasd-termite"]
    assert K.records["uasd-termite"]["text_kn"] in r["answer"]
    assert "1800-180-1551" in r["answer"]


def test_editing_a_record_voids_its_confirmation(tmp_path):
    k = _confirm(tmp_path, {"uasd-termite": {"hash": "0000000000000000"}})
    assert answer(k, "ಗೆದ್ದಲು")["status"] == "refused_not_verified"


def test_product_needs_registration_answer(tmp_path):
    k = _confirm(tmp_path, {"uasd-termite": {"registered": ""}})
    assert answer(k, "ಗೆದ್ದಲು")["status"] == "refused_not_verified"
    k = _confirm(tmp_path, {"uasd-termite": {"registered": "no"}})
    assert answer(k, "ಗೆದ್ದಲು")["status"] == "refused_not_verified"


def test_wrong_or_unsure_is_not_used(tmp_path):
    for status in ("WRONG", "UNSURE", ""):
        k = _confirm(tmp_path, {"uasd-termite": {"status": status}})
        assert answer(k, "ಗೆದ್ದಲು")["status"] == "refused_not_verified"


def test_regions_are_shown_separately_never_merged(tmp_path):
    k = _confirm(tmp_path, {"uasb-harvest": {"registered": "n.a."}, "uasd-harvest": {"registered": "n.a."}})
    r = answer(k, "ಕಟಾವು ಯಾವಾಗ?")
    assert r["answer"].count("📘") == 2
    south_only = answer(k, "ಮಂಡ್ಯದಲ್ಲಿ ಕಟಾವು ಯಾವಾಗ?")
    assert south_only["record_ids"] == ["uasb-harvest"]


def test_known_but_uncovered_pest_says_so():
    r = answer(K, "ಕೆಂಪು ಕೊಳೆ ರೋಗ ಬಂದರೆ?", include_drafts=True)
    assert r["status"] == "refused_not_verified"
    assert "ಕೆಂಪು ಕೊಳೆ" in r["answer"]


def test_draft_preview_is_labelled():
    r = answer(K, "ಗೆದ್ದಲು", include_drafts=True)
    assert r["answer"].startswith("⚠️") and "DRAFT" in r["answer"]


def test_review_sheet_hashes_match_records():
    """review_sheet.csv must be regenerated (verify_facts.py) whenever a record changes."""
    with open(f"{KNOWLEDGE_DIR}/review_sheet.csv", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    col = next(c for c in rows[0] if c.startswith("record_hash"))
    for row in rows:
        rec = K.records[row["id"]]
        assert row[col] == record_hash(rec["_source"], rec), row["id"]


def test_rejected_pesticides_can_never_be_shown():
    """Phorate and metasystox are not registered in India (owner + expert, 2026-09-30)."""
    import glob
    rejected = set()
    for path in glob.glob(f"{KNOWLEDGE_DIR}/facts_*.json"):
        with open(path, encoding="utf-8") as f:
            rejected |= {r["id"] for r in json.load(f).get("rejected_records", [])}
    assert {"uasd-wwa-phorate", "uasd-wwa-metasystox"} <= rejected
    assert not rejected & set(K.records)
    for text in (r["text_kn"] + " " + r.get("product_kn", "") for r in K.records.values()):
        for banned in ("ಫೊರೇಟ್", "ಮೆಟಾಸಿಸ್ಟಾಕ್ಸ್", "ಬೆನೋಮಿಲ್"):
            assert banned not in text
