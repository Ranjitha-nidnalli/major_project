"""
Load an eval question set: the original eval/questions.json, or an
expert-written CSV made from eval/expert/heldout_questions_TEMPLATE.csv.

Both come back as the same records: id, category, question,
expected_answer, expected_refusal. For the JSON set, expected_refusal is
inferred as before (the expected answer is the refusal phrase). For the
CSV it is the expert's own yes/no column, and a should-refuse row gets the
refusal phrase as its expected answer so downstream scoring is unchanged.

Fails loudly on a malformed row rather than guessing: an expert answer
must never be filled in or "fixed up" by code.
"""
import csv
import json


class QuestionSetError(ValueError):
    pass


def _column(fieldnames, prefix):
    # The template's headers carry inline instructions, e.g.
    # "should_the_system_refuse (yes / no)"; match on the leading name.
    for name in fieldnames:
        if name.strip().lstrip("﻿").startswith(prefix):
            return name
    raise QuestionSetError(f"missing column starting with {prefix!r}")


def _load_csv(text, refusal_phrase):
    reader = csv.DictReader(text.splitlines())
    cols = {k: _column(reader.fieldnames, k) for k in (
        "id", "category", "question_kannada", "should_the_system_refuse", "expected_answer_kannada")}
    questions = []
    for line_no, row in enumerate(reader, start=2):
        question = (row[cols["question_kannada"]] or "").strip()
        if not question:
            continue  # unused template row
        qid = (row[cols["id"]] or "").strip()
        refuse = (row[cols["should_the_system_refuse"]] or "").strip().lower()
        expected = (row[cols["expected_answer_kannada"]] or "").strip()
        if not qid:
            raise QuestionSetError(f"line {line_no}: missing id")
        if refuse not in ("yes", "no"):
            raise QuestionSetError(f"line {line_no} ({qid}): should_the_system_refuse must be yes or no, got {refuse!r}")
        if refuse == "no" and not expected:
            raise QuestionSetError(f"line {line_no} ({qid}): answerable question has no expected answer")
        questions.append({
            "id": qid,
            "category": (row[cols["category"]] or "").strip().lower(),
            "question": question,
            "expected_answer": refusal_phrase if refuse == "yes" else expected,
            "expected_refusal": refuse == "yes",
        })
    return questions


def load_questions(path, refusal_phrase):
    """Returns (questions, raw_bytes); raw_bytes is what the run hashes."""
    with open(path, "rb") as f:
        raw = f.read()
    text = raw.decode("utf-8-sig")
    if path.lower().endswith(".csv"):
        questions = _load_csv(text, refusal_phrase)
    else:
        questions = json.loads(text)
        for q in questions:
            q["expected_refusal"] = refusal_phrase in q["expected_answer"]
    if not questions:
        raise QuestionSetError(f"no questions in {path}")
    ids = [q["id"] for q in questions]
    if len(ids) != len(set(ids)):
        raise QuestionSetError(f"duplicate ids in {path}")
    return questions, raw
