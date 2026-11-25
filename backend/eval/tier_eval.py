"""
Evaluation runner for the tiered product (K-020).

Reads a frozen golden file (eval/GOLDEN_SET_GUIDE.md), asks the tiered policy each
question, and counts outcomes. It reports no ROUGE, BLEU or chrF (CLAUDE.md).

Outcomes per question:
  correct_answer   expected answer; served from Tier A; every gold record was served;
                   no dose-bearing record outside the gold was served
  correct_refusal  expected refuse; got Tier C
  wrongly_refused  expected answer; got Tier C
  wrongly_answered expected refuse and got A or B, or expected answer and the served
                   records do not cover the gold
  wrong_dose       counted separately (a subset of wrongly_answered): a Tier A answer
                   served a dose-bearing record the gold does not justify, or a Tier B
                   answer contains a dose-like number

Usage:
  python -m eval.tier_eval GOLDEN.jsonl [--out eval/runs] [--compare OTHER_RESULTS.jsonl]
Run files go to their own folder with the dataset SHA-256, model ID and prompt version.
Unconfirmed (draft) records are never used here: it always loads the real review sheet.
"""
import argparse
import asyncio
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.tiers import answer_tiered, has_dose_like_numbers  # noqa: E402

DOSE_FIELDS = ("product_kn", "per_litre", "per_acre", "water_per_acre")
ALWAYS_OK_EXTRA = {"uasd-pesticide-safety"}     # the safety note rides on pesticide answers
OUTCOMES = ("correct_answer", "correct_refusal", "wrongly_refused", "wrongly_answered")
PROMPT_VERSION = "tiered-v1"                    # bump when Tier B prompts or rules change


# ---- statistics -------------------------------------------------------------------

def wilson(k, n, z=1.959964):
    """Wilson 95% interval for k successes out of n (None if n == 0)."""
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def exact_mcnemar(b, c):
    """Two-sided exact McNemar p from the discordant counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    from scipy.stats import binomtest
    return float(binomtest(min(b, c), n, 0.5).pvalue)


# ---- scoring ----------------------------------------------------------------------

def classify(gold, result, records):
    """(outcome, wrong_dose) for one question. `records` maps id -> record dict."""
    served = [r for r in result.get("record_ids", [])]
    answered = result["tier"] in ("A", "B")
    gold_ids = set(gold.get("gold_record_ids") or [])
    dose_extra = [r for r in served
                  if r not in gold_ids and r not in ALWAYS_OK_EXTRA
                  and any(records.get(r, {}).get(f) for f in DOSE_FIELDS)]
    wrong_dose = bool(dose_extra) or (result["tier"] == "B" and has_dose_like_numbers(result["answer"]))
    if gold["expected"] == "refuse":
        return ("wrongly_answered", wrong_dose) if answered else ("correct_refusal", False)
    if not answered:
        return "wrongly_refused", False
    covers = result["tier"] == "A" and gold_ids <= set(served)
    if covers and not dose_extra:
        return "correct_answer", False
    return "wrongly_answered", wrong_dose


def summarize(rows):
    """Counts, Wilson intervals and per-crop counts from classified rows."""
    def block(subset):
        n = len(subset)
        out = {"n": n}
        for o in OUTCOMES:
            k = sum(1 for r in subset if r["outcome"] == o)
            ci = wilson(k, n)
            out[o] = {"count": k, "wilson95": ci}
        out["wrong_dose"] = {"count": sum(1 for r in subset if r["wrong_dose"])}
        exp_ans = [r for r in subset if r["expected"] == "answer"]
        exp_ref = [r for r in subset if r["expected"] == "refuse"]
        out["answerable_n"], out["unanswerable_n"] = len(exp_ans), len(exp_ref)
        out["wrongly_refused_rate"] = (out["wrongly_refused"]["count"] / len(exp_ans)) if exp_ans else None
        return out
    crops = sorted({r["crop"] for r in rows})
    return {"overall": block(rows), "by_crop": {c: block([r for r in rows if r["crop"] == c]) for c in crops}}


def paired_mcnemar(rows_a, rows_b):
    """Exact McNemar on 'question handled correctly' between two runs on the same ids."""
    good = {"correct_answer", "correct_refusal"}
    a = {r["id"]: r["outcome"] in good for r in rows_a}
    b = {r["id"]: r["outcome"] in good for r in rows_b}
    ids = sorted(set(a) & set(b))
    only_a = sum(1 for i in ids if a[i] and not b[i])
    only_b = sum(1 for i in ids if b[i] and not a[i])
    return {"n_shared": len(ids), "only_first_correct": only_a, "only_second_correct": only_b,
            "p_exact": exact_mcnemar(only_a, only_b)}


# ---- running ----------------------------------------------------------------------

def load_golden(path):
    with open(path, "rb") as f:
        raw = f.read()
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    for g in rows:
        assert g["unanswerable"] == (g["expected"] == "refuse"), f"{g['id']}: unanswerable disagrees with expected"
    return rows, hashlib.sha256(raw).hexdigest()


async def run_policy(golden, knowledge, tier_b_enabled=False, rag_fn_factory=None):
    from services.fact_answer import answer as facts_answer
    out = []
    for g in golden:
        q = g["question_kn"]

        async def facts_fn(q=q):
            return facts_answer(knowledge, q)

        async def rag_fn():
            raise RuntimeError("Tier B is not wired into the offline runner")
        r = await answer_tiered(q, facts_fn=facts_fn,
                                rag_fn=rag_fn_factory(q) if rag_fn_factory else rag_fn,
                                tier_b_enabled=tier_b_enabled)
        out.append({"tier": r["tier"], "tier_reason": r["tier_reason"], "answer": r["answer"],
                    "record_ids": r["facts"].get("record_ids", []) if r["tier"] == "A" else []})
    return out


def evaluate(golden, results, records):
    rows = []
    for g, r in zip(golden, results):
        outcome, wrong_dose = classify(g, r, records)
        rows.append({"id": g["id"], "crop": g.get("crop", "unknown"), "expected": g["expected"],
                     "tier": r["tier"], "tier_reason": r["tier_reason"], "record_ids": r["record_ids"],
                     "outcome": outcome, "wrong_dose": wrong_dose})
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("golden")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs"))
    ap.add_argument("--compare", help="results.jsonl of another run on the same golden set")
    ap.add_argument("--tier-b", action="store_true", help="enable Tier B (needs live services; off offline)")
    args = ap.parse_args(argv)

    from services.fact_answer import Knowledge
    knowledge = Knowledge.load()
    golden, sha = load_golden(args.golden)
    results = asyncio.run(run_policy(golden, knowledge, tier_b_enabled=args.tier_b))
    rows = evaluate(golden, results, knowledge.records)

    summary = {"dataset": os.path.basename(args.golden), "dataset_sha256": sha,
               "model_id": os.getenv("GENERATION_MODEL", "none (tier A is deterministic)"),
               "prompt_version": PROMPT_VERSION, "tier_b_enabled": args.tier_b,
               "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               **summarize(rows)}
    if args.compare:
        with open(args.compare, encoding="utf-8") as f:
            other = [json.loads(line) for line in f if line.strip()]
        summary["mcnemar_vs_compare"] = paired_mcnemar(rows, other)

    folder = os.path.join(args.out, f"tier_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}_{sha[:8]}")
    os.makedirs(folder, exist_ok=False)
    with open(os.path.join(folder, "results.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(folder, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary["overall"], ensure_ascii=False, indent=2))
    print(f"written to {folder}")


if __name__ == "__main__":
    main()
