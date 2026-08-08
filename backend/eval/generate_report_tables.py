"""
Generate the retrieval ablation summary table for your report.

Usage:
    cd backend && python eval/generate_report_tables.py

Outputs markdown tables ready to paste into your report.
"""
import os
import sys
import json
import statistics
import math
from collections import defaultdict

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "retrieval_results.jsonl")

CONFIG_ORDER = [
    "dense", "dense+rerank",
    "sparse", "sparse+rerank",
    "hybrid", "hybrid+rerank",
]

METRICS = ["recall@1", "recall@3", "recall@5", "recall@10", "mrr", "ndcg@5"]


def mean_std_ci(values):
    n = len(values)
    mean = statistics.mean(values)
    std = statistics.stdev(values) if n > 1 else 0.0
    ci95 = 1.96 * std / math.sqrt(n) if n > 1 else 0.0
    return mean, std, ci95, n


def percentile(values, p):
    s = sorted(values)
    idx = min(int(len(s) * p), len(s) - 1)
    return s[idx]


def main():
    if not os.path.exists(RESULTS_PATH):
        print(f"❌ {RESULTS_PATH} not found. Run run_retrieval_ablation.py first.")
        sys.exit(1)

    raw = []
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                raw.append(json.loads(line))

    per_config = defaultdict(lambda: defaultdict(list))
    per_config_lat = defaultdict(list)

    for r in raw:
        cfg = r["config"]
        for m in METRICS:
            per_config[cfg][m].append(r[m])
        per_config_lat[cfg].append(r["latency_seconds"])

    # --- Markdown table for report ---
    print("\n## Retrieval Ablation Results\n")
    print("| Config | recall@1 | recall@3 | recall@5 | recall@10 | MRR | nDCG@5 | Latency (mean / p95) |")
    print("|--------|----------|----------|----------|-----------|-----|--------|----------------------|")

    for cfg in CONFIG_ORDER:
        if cfg not in per_config:
            continue
        vals = per_config[cfg]
        cells = []
        for m in METRICS:
            mean, _, ci95, n = mean_std_ci(vals[m])
            cells.append(f"{mean:.3f}±{ci95:.3f}")
        lats = per_config_lat[cfg]
        lat_mean = statistics.mean(lats)
        lat_p95 = percentile(lats, 0.95)
        cells.append(f"{lat_mean:.2f}s / {lat_p95:.2f}s")
        print(f"| {cfg} | " + " | ".join(cells) + " |")

    print("\n*n = 15 answerable questions; 95% CI shown. Unanswerable question (price-1) excluded from IR metrics.*")

    # --- Per-question breakdown for appendix ---
    print("\n---\n")
    print("## Per-Question Breakdown (appendix)\n")
    by_q = defaultdict(list)
    for r in raw:
        by_q[r["id"]].append(r)

    for qid in sorted(by_q.keys()):
        print(f"\n**{qid}** ({by_q[qid][0]['category']})")
        print("| Config | recall@5 | RR | nDCG@5 | Latency |")
        print("|--------|----------|----|--------|---------|")
        for r in by_q[qid]:
            print(f"| {r['config']} | {r['recall@5']:.0f} | {r['reciprocal_rank']:.3f} | {r['ndcg@5']:.3f} | {r['latency_seconds']:.2f}s |")


if __name__ == "__main__":
    main()
