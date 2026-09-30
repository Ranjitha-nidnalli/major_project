"""
Check draft fact records against the converted source text, then write the
review sheet a person uses to confirm or reject each record.

Two mechanical checks per record (they catch transcription mistakes, not
agricultural ones; only the person reviewing the printed page can do that):
  1. every excerpt appears verbatim on the stated page of the converted text;
  2. every number in the farmer-facing fields (text_kn, product_kn,
     per_litre, per_acre, water_per_acre) appears in the record's excerpts.

Usage:
    cd backend && python knowledge/verify_facts.py
Needs backend/corpus/converted/*.txt (gitignored; regenerate as described in
backend/corpus/SOURCES.md). Exits non-zero if any record fails.
"""
import csv
import glob
import json
import os
import re
import sys

KNOWLEDGE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(KNOWLEDGE_DIR)
REPO_DIR = os.path.dirname(BACKEND_DIR)
REVIEW_SHEET = os.path.join(KNOWLEDGE_DIR, "review_sheet.csv")

CHECKED_FIELDS = ("text_kn", "product_kn", "per_litre", "per_acre", "water_per_acre")
_ZERO_WIDTH = re.compile("[​‌‍﻿]")
_WS = re.compile(r"\s+")
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
_PAGE_HEADER = re.compile(r"^######## \S+ \| pdf_page_index=(\d+) \|", re.M)


def normalize(text):
    return _WS.sub(" ", _ZERO_WIDTH.sub("", text)).strip()


def numbers(text):
    """Numbers as written, with thousands separators removed ('10,000' -> '10000')."""
    return {n.replace(",", "") for n in _NUMBER.findall(text or "")}


def split_pages(converted_text):
    """{pdf_page_index: normalized page text} from a converted-chapter file."""
    pages = {}
    matches = list(_PAGE_HEADER.finditer(converted_text))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(converted_text)
        pages[int(m.group(1))] = normalize(converted_text[m.end():end])
    return pages


def check_record(record, pages):
    """List of problems with one record; empty means both checks pass."""
    problems = []
    # A record normally cites one page; a table entry that continues on the
    # next page lists both in pdf_page_indexes, and each excerpt must then
    # appear on one of them.
    indexes = record.get("pdf_page_indexes") or [record["pdf_page_index"]]
    missing_pages = [i for i in indexes if i not in pages]
    if missing_pages:
        return [f"page(s) {missing_pages} not in converted text"]
    excerpts = record.get("excerpts") or []
    if not excerpts:
        problems.append("no excerpts")
    for ex in excerpts:
        if not any(normalize(ex) in pages[i] for i in indexes):
            problems.append(f"excerpt not found on page: {ex[:60]!r}")
    source_numbers = set().union(*(numbers(ex) for ex in excerpts)) if excerpts else set()
    for field in CHECKED_FIELDS:
        missing = numbers(record.get(field)) - source_numbers
        if missing:
            problems.append(f"{field} has numbers not in excerpts: {sorted(missing)}")
    return problems


def main():
    all_rows, failures, seen = [], 0, set()
    for path in sorted(glob.glob(os.path.join(KNOWLEDGE_DIR, "facts_*.json"))):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        with open(os.path.join(REPO_DIR, data["converted_text"]), encoding="utf-8") as f:
            pages = split_pages(f.read())
        for r in data["records"]:
            if r["id"] in seen:
                problems = [f"duplicate id {r['id']}"]
            else:
                problems = check_record(r, pages)
            seen.add(r["id"])
            if problems:
                failures += 1
                print(f"FAIL {r['id']}: " + "; ".join(problems))
            all_rows.append((data, r, problems))

    with open(REVIEW_SHEET, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["id", "source", "region", "printed_page", "topic", "subject_en", "subject_kn",
                    "product_kn", "per_litre", "per_acre", "water_per_acre", "text_kn (what the bot will say)",
                    "source_excerpts", "flags (read these)", "auto_check",
                    "REVIEW: status (CONFIRMED / WRONG / UNSURE)", "REVIEW: correct text or dose if WRONG",
                    "REVIEW: pesticide currently registered? (yes / no / n.a.)", "REVIEW: notes"])
        for data, r, problems in all_rows:
            w.writerow([r["id"], data["source"], data["region"], r["printed_page"], r["topic"],
                        r.get("subject_en", ""), r.get("subject_kn", ""), r.get("product_kn", ""),
                        r.get("per_litre", ""), r.get("per_acre", ""), r.get("water_per_acre", ""),
                        r["text_kn"], "\n".join(r["excerpts"]), "\n".join(r.get("flags", [])),
                        "PASS" if not problems else "FAIL: " + "; ".join(problems),
                        "", "", "" if r.get("product_kn") else "n.a.", ""])
    print(f"{len(all_rows)} records, {failures} failing. Review sheet: {REVIEW_SHEET}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
