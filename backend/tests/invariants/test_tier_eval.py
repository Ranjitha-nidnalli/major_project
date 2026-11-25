"""K-020 / K-021: outcome counting, statistics and the release gate. Pure."""
import json
import math

import pytest

from eval.tier_eval import (
    classify, evaluate, exact_mcnemar, load_golden, paired_mcnemar, run_policy, summarize, wilson,
)
from eval.release_gate import check, main as gate_main

RECORDS = {
    "r-variety": {"id": "r-variety"},
    "r-dose": {"id": "r-dose", "product_kn": "X", "per_acre": "2 kg"},
    "uasd-pesticide-safety": {"id": "uasd-pesticide-safety"},
}


def A(*ids):
    return {"tier": "A", "tier_reason": "confirmed_records", "answer": "t", "record_ids": list(ids)}


C = {"tier": "C", "tier_reason": "x", "answer": "KCC", "record_ids": []}


def g(expected, ids=()):
    return {"id": "q", "crop": "sugarcane", "expected": expected, "unanswerable": expected == "refuse",
            "gold_record_ids": list(ids)}


def test_each_outcome():
    assert classify(g("answer", ["r-variety"]), A("r-variety"), RECORDS) == ("correct_answer", False)
    assert classify(g("refuse"), C, RECORDS) == ("correct_refusal", False)
    assert classify(g("answer", ["r-variety"]), C, RECORDS) == ("wrongly_refused", False)
    assert classify(g("refuse"), A("r-variety"), RECORDS) == ("wrongly_answered", False)


def test_answer_that_misses_the_gold_record_is_wrong():
    assert classify(g("answer", ["r-variety"]), A("uasd-pesticide-safety"), RECORDS)[0] == "wrongly_answered"


def test_serving_an_unjustified_dose_record_is_a_wrong_dose():
    assert classify(g("refuse"), A("r-dose"), RECORDS) == ("wrongly_answered", True)
    assert classify(g("answer", ["r-variety"]), A("r-variety", "r-dose"), RECORDS) == ("wrongly_answered", True)


def test_the_pesticide_safety_note_is_not_an_extra_dose_record():
    assert classify(g("answer", ["r-dose"]), A("r-dose", "uasd-pesticide-safety"), RECORDS)[0] == "correct_answer"


def test_tier_b_dose_number_is_a_wrong_dose():
    res = {"tier": "B", "tier_reason": "x", "answer": "2 kg ಹಾಕಿ", "record_ids": []}
    assert classify(g("refuse"), res, RECORDS) == ("wrongly_answered", True)


def test_wilson_matches_known_values():
    lo, hi = wilson(36, 40)
    assert math.isclose(lo, 0.7695, abs_tol=1e-3) and math.isclose(hi, 0.9604, abs_tol=1e-3)
    assert wilson(0, 0) is None
    assert wilson(0, 10)[0] == pytest.approx(0.0, abs=1e-9)
    assert wilson(10, 10)[1] == pytest.approx(1.0, abs=1e-9)


def test_exact_mcnemar_matches_the_v1_v2_comparison_in_the_paper():
    # 36 vs 29 correct of 40 (net 7) with 10 vs 3 discordant pairs gives the paper's p = 0.092
    # (the discordant split is inferred from that p; the paper does not print it).
    assert exact_mcnemar(3, 10) == pytest.approx(0.092, abs=1e-3)
    assert exact_mcnemar(1, 8) == pytest.approx(0.0391, abs=1e-3)
    assert exact_mcnemar(0, 0) == 1.0
    assert exact_mcnemar(5, 5) == 1.0


def test_summary_counts_and_by_crop():
    rows = [
        {"id": "1", "crop": "sugarcane", "expected": "answer", "outcome": "correct_answer", "wrong_dose": False},
        {"id": "2", "crop": "sugarcane", "expected": "refuse", "outcome": "wrongly_answered", "wrong_dose": True},
        {"id": "3", "crop": "paddy", "expected": "answer", "outcome": "wrongly_refused", "wrong_dose": False},
    ]
    s = summarize(rows)
    assert s["overall"]["n"] == 3 and s["overall"]["wrong_dose"]["count"] == 1
    assert s["overall"]["wrongly_refused_rate"] == 0.5
    assert set(s["by_crop"]) == {"paddy", "sugarcane"}


def test_paired_mcnemar_uses_shared_ids_only():
    a = [{"id": "1", "outcome": "correct_answer"}, {"id": "2", "outcome": "wrongly_refused"}]
    b = [{"id": "1", "outcome": "wrongly_refused"}, {"id": "2", "outcome": "wrongly_refused"}, {"id": "9", "outcome": "correct_answer"}]
    r = paired_mcnemar(a, b)
    assert r["n_shared"] == 2 and r["only_first_correct"] == 1 and r["only_second_correct"] == 0


def test_golden_loader_rejects_inconsistent_unanswerable_flag(tmp_path):
    p = tmp_path / "g.jsonl"
    p.write_text(json.dumps({"id": "x", "expected": "refuse", "unanswerable": False}) + "\n", encoding="utf-8")
    with pytest.raises(AssertionError):
        load_golden(str(p))


def test_runner_end_to_end_with_no_confirmed_records():
    import asyncio
    from services.fact_answer import Knowledge
    K = Knowledge.load(reviewed_sheet="__no_such_file__")      # nothing confirmed: everything refused
    golden = [{"id": "a", "crop": "sugarcane", "question_kn": "ಕಬ್ಬಿನ ತಳಿಗಳು", "expected": "answer",
               "unanswerable": False, "gold_record_ids": ["uasb-variety-co62175"]},
              {"id": "b", "crop": "sugarcane", "question_kn": "ಭತ್ತಕ್ಕೆ ಎಷ್ಟು ಗೊಬ್ಬರ", "expected": "refuse",
               "unanswerable": True, "gold_record_ids": []}]
    rows = evaluate(golden, asyncio.run(run_policy(golden, K)), K.records)
    assert [r["outcome"] for r in rows] == ["wrongly_refused", "correct_refusal"]


# ---- release gate ----------------------------------------------------------------------

def summ(wrongly_answered=0, wrong_dose=0, refused_rate=0.1, sha="abc"):
    return {"dataset_sha256": sha, "overall": {
        "wrongly_answered": {"count": wrongly_answered}, "wrong_dose": {"count": wrong_dose},
        "wrongly_refused_rate": refused_rate}}


def test_gate_passes_when_nothing_is_worse():
    assert check(summ(1, 0, 0.10), summ(1, 0, 0.12)) == []
    assert check(summ(2, 1, 0.20), summ(1, 0, 0.05)) == []


def test_gate_blocks_any_rise_in_unsafe_counts():
    assert check(summ(0, 0), summ(1, 0))
    assert check(summ(0, 0), summ(0, 1))


def test_gate_blocks_wrong_refusals_beyond_the_margin_only():
    assert check(summ(refused_rate=0.10), summ(refused_rate=0.16))
    assert not check(summ(refused_rate=0.10), summ(refused_rate=0.14))
    assert not check(summ(refused_rate=0.10), summ(refused_rate=0.16), margin=0.10)


def test_gate_blocks_on_different_golden_files():
    assert "not comparable" in check(summ(sha="a"), summ(sha="b"))[0]


def test_gate_cli_exit_codes(tmp_path):
    b, ok, bad = tmp_path / "b.json", tmp_path / "ok.json", tmp_path / "bad.json"
    b.write_text(json.dumps(summ(0, 0)), encoding="utf-8")
    ok.write_text(json.dumps(summ(0, 0)), encoding="utf-8")
    bad.write_text(json.dumps(summ(1, 0)), encoding="utf-8")
    assert gate_main([str(b), str(ok)]) == 0
    assert gate_main([str(b), str(bad)]) == 1
