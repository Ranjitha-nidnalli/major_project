"""
Weekly audit (K-046): a random sample of live answers into a review sheet, and the reviewer's
marks counted back in.

  python -m eval.weekly_audit sample  [--days 7] [--n 25] [--seed 1] [--out audit_YYYY-WW.csv]
  python -m eval.weekly_audit import  audit_YYYY-WW.csv     (stores the marks, prints counts)

The sheet shows the question and the answer served, so a reviewer can mark each one OK, WRONG
or UNSURE against the cited record. Any WRONG goes through the incident process: switch the
record off and add a regression test.
"""
import argparse
import csv
import os
import random
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

VERDICT_COL = "REVIEW: verdict (OK / WRONG / UNSURE)"
NOTE_COL = "REVIEW: note"
REVIEWER_COL = "REVIEW: reviewer"
COLUMNS = ["answer_id", "ts", "channel", "tier", "tier_reason", "question", "answer", "record_ids",
           "sources", VERDICT_COL, NOTE_COL, REVIEWER_COL]
VALID = {"OK", "WRONG", "UNSURE"}


def sample(store, now, days=7, n=25, seed=None):
    rows = store.answers_since(now - timedelta(days=days))
    rng = random.Random(seed)
    return rng.sample(rows, min(n, len(rows)))


def write_sheet(rows, path):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        for r in rows:
            sources = "; ".join(f"{p.get('document')} p.{p.get('page')}" if p.get("page") else str(p.get("document"))
                                for p in r["provenance"])
            w.writerow([r["answer_id"], r["ts"], r["channel"], r["tier"], r["tier_reason"], r["question"],
                        r["answer_text"], " ".join(r["record_ids"]), sources, "", "", ""])


def import_marks(store, path, week):
    """Store the reviewer's marks; returns counts. Unmarked rows are skipped, bad marks rejected."""
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            verdict = (row.get(VERDICT_COL) or "").strip().upper()
            if not verdict:
                continue
            if verdict not in VALID:
                raise ValueError(f"{row['answer_id']}: verdict must be OK, WRONG or UNSURE, got {verdict!r}")
            store.save_mark(row["answer_id"], verdict, (row.get(REVIEWER_COL) or "").strip(),
                            (row.get(NOTE_COL) or "").strip(), week)
    return store.mark_counts()


def main(argv=None):
    from services.store import Store
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--days", type=int, default=7)
    s.add_argument("--n", type=int, default=25)
    s.add_argument("--seed", type=int)
    s.add_argument("--out")
    i = sub.add_parser("import")
    i.add_argument("sheet")
    args = ap.parse_args(argv)
    store, now = Store(), datetime.now(timezone.utc)
    week = now.strftime("%G-W%V")
    if args.cmd == "sample":
        rows = sample(store, now, args.days, args.n, args.seed)
        out = args.out or f"audit_{week}.csv"
        write_sheet(rows, out)
        print(f"{len(rows)} answers written to {out}")
    else:
        print(import_marks(store, args.sheet, week))


if __name__ == "__main__":
    main()
