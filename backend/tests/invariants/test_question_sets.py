"""
eval/question_sets.py: the loader both eval scripts use for the original
questions.json and the expert's held-out CSV.
"""
import os

import pytest

from eval.question_sets import load_questions, QuestionSetError

PHRASE = "ಕ್ಷಮಿಸಿ, ಈ ಮಾಹಿತಿ ನಮ್ಮ ಡೇಟಾಬೇಸ್ನಲ್ಲಿ ಲಭ್ಯವಿಲ್ಲ."
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HEADER = ("id,category (pest / disease),question_kannada,should_the_system_refuse (yes / no),"
          "expected_answer_kannada (leave empty if it should refuse),source_you_used,notes\n")


def _write(tmp_path, body, name="q.csv"):
    p = tmp_path / name
    p.write_text("﻿" + HEADER + body, encoding="utf-8")
    return str(p)


def test_refusal_phrase_copy_matches_rag_service():
    """run_no_retrieval_baseline.py duplicates the phrase to avoid loading models."""
    with open(os.path.join(BACKEND_DIR, "rag_service.py"), encoding="utf-8") as f:
        assert f'NOT_IN_CONTEXT_PHRASE = "{PHRASE}"' in f.read()
    with open(os.path.join(BACKEND_DIR, "eval", "run_no_retrieval_baseline.py"), encoding="utf-8") as f:
        assert f'NOT_IN_CONTEXT_PHRASE = "{PHRASE}"' in f.read()


def test_original_json_set_infers_refusals():
    questions, raw = load_questions(os.path.join(BACKEND_DIR, "eval", "questions.json"), PHRASE)
    assert len(questions) == 18 and raw
    assert {q["id"] for q in questions if q["expected_refusal"]} == {"price-1", "pest-5", "general-5"}


def test_csv_reads_expert_columns_and_skips_blank_rows(tmp_path):
    path = _write(tmp_path, "H-01,Pest,ಪ್ರಶ್ನೆ ಒಂದು,no,ಉತ್ತರ,book p.3,\nH-02,price,ಪ್ರಶ್ನೆ ಎರಡು,YES,,,\nH-03,,,,,,\n")
    questions, _ = load_questions(path, PHRASE)
    assert [q["id"] for q in questions] == ["H-01", "H-02"]
    assert questions[0] == {"id": "H-01", "category": "pest", "question": "ಪ್ರಶ್ನೆ ಒಂದು",
                            "expected_answer": "ಉತ್ತರ", "expected_refusal": False}
    assert questions[1]["expected_refusal"] is True
    assert questions[1]["expected_answer"] == PHRASE


@pytest.mark.parametrize("body", [
    "H-01,pest,ಪ್ರಶ್ನೆ,maybe,ಉತ್ತರ,,\n",   # refuse column not yes/no
    "H-01,pest,ಪ್ರಶ್ನೆ,no,,,\n",           # answerable without an expected answer
    ",pest,ಪ್ರಶ್ನೆ,no,ಉತ್ತರ,,\n",          # missing id
    "H-01,pest,ಒಂದು,yes,,,\nH-01,pest,ಎರಡು,yes,,,\n",  # duplicate id
    "H-01,,,,,,\n",                        # nothing filled in
])
def test_csv_fails_loudly_instead_of_guessing(tmp_path, body):
    with pytest.raises(QuestionSetError):
        load_questions(_write(tmp_path, body), PHRASE)
