"""
Build a local HTML "review pack" for the person checking the fact records:
every printed page the records cite, rendered upright as an image, with the
records that cite it listed underneath (row number in review_sheet.csv,
what the bot will say, doses, flags).

The person still records decisions in review_sheet.csv; this only saves
them from hunting through the PDFs. Output goes under
backend/corpus/converted/review_pack/ (gitignored: page images are derived
from the downloaded source PDFs).

Usage:
    cd backend && python knowledge/build_review_pack.py [--ai-prereview PATH]
--ai-prereview shows an AI pass's suggestion next to each record, clearly
labelled as a suggestion. It never counts as a confirmation.
"""
import argparse
import csv
import html
import json
import os
import glob

import fitz  # PyMuPDF

KNOWLEDGE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(KNOWLEDGE_DIR)
OUT_DIR = os.path.join(BACKEND_DIR, "corpus", "converted", "review_pack")
SHEET = os.path.join(KNOWLEDGE_DIR, "review_sheet.csv")
PRIORITY_WORDS = ("table", "interleaved", "column", "scrambled", "SUSPECT", "merged", "SAFETY")


def _upright_matrix(page, zoom=1.8):
    """Rotate pages whose text runs vertically (the landscape pest/disease tables)."""
    dirs = [l["dir"] for b in page.get_text("dict")["blocks"] if "lines" in b for l in b["lines"]]
    vertical = sum(1 for _, y in dirs if abs(y) > 0.9)
    m = fitz.Matrix(zoom, zoom)
    return m.prerotate(90) if dirs and vertical > len(dirs) / 2 else m


def _column(row, prefix):
    return next((row[c] for c in row if c and c.startswith(prefix)), "") or ""


def main(ai_prereview=None):
    os.makedirs(os.path.join(OUT_DIR, "pages"), exist_ok=True)
    with open(SHEET, encoding="utf-8-sig") as f:
        sheet_rows = {r["id"]: i + 2 for i, r in enumerate(csv.DictReader(f))}   # Excel row numbers
    ai = {}
    if ai_prereview:
        with open(ai_prereview, encoding="utf-8-sig") as f:
            ai = {r["id"]: r for r in csv.DictReader(f)}

    sections, n_priority = [], 0
    for path in sorted(glob.glob(os.path.join(KNOWLEDGE_DIR, "facts_*.json"))):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        doc = fitz.open(os.path.join(BACKEND_DIR, data["pdf"]))
        by_page = {}
        for r in data["records"]:
            for idx in r.get("pdf_page_indexes") or [r["pdf_page_index"]]:
                by_page.setdefault(idx, []).append(r)
        for idx in sorted(by_page):
            img = f"pages/{data['source']}_{idx}.png"
            page = doc[idx]
            page.get_pixmap(matrix=_upright_matrix(page)).save(os.path.join(OUT_DIR, img))
            rows_html = []
            for r in by_page[idx]:
                flags = r.get("flags", [])
                priority = bool(r.get("product_kn")) or any(w in f for f in flags for w in PRIORITY_WORDS)
                n_priority += priority
                dose = " · ".join(x for x in (r.get("product_kn"), r.get("per_litre") and f"per litre {r['per_litre']}",
                                              r.get("per_acre") and f"per acre {r['per_acre']}",
                                              r.get("water_per_acre") and f"water {r['water_per_acre']}/acre") if x)
                ai_row = ai.get(r["id"])
                ai_html = (f"<div class='ai'>AI suggestion only (not a confirmation): "
                           f"{html.escape(_column(ai_row, 'REVIEW: status'))}</div>") if ai_row else ""
                rows_html.append(
                    f"<tr class='{'prio' if priority else ''}'><td>{sheet_rows.get(r['id'], '?')}</td>"
                    f"<td><b>{html.escape(r['id'])}</b>{'<br>⭐ check first' if priority else ''}</td>"
                    f"<td class='kn'>{html.escape(r['text_kn'])}"
                    f"{'<div class=dose>' + html.escape(dose) + '</div>' if dose else ''}</td>"
                    f"<td>{'<br>'.join('• ' + html.escape(x) for x in flags)}{ai_html}</td></tr>")
            sections.append(
                f"<section><h2>{html.escape(data['source_title'])}: PDF page index {idx}, "
                f"printed page {html.escape(str(by_page[idx][0]['printed_page']))}</h2>"
                f"<img src='{img}' alt='page'><table><tr><th>Sheet row</th><th>Record</th>"
                f"<th>What the bot will say</th><th>Flags</th></tr>{''.join(rows_html)}</table></section>")

    page_html = f"""<!doctype html><html lang="kn"><head><meta charset="utf-8">
<title>Fact Review Pack</title><style>
body{{font-family:system-ui,'Noto Sans Kannada',sans-serif;margin:16px;max-width:1200px}}
img{{width:100%;border:1px solid #999;margin:8px 0}}
table{{border-collapse:collapse;width:100%}} td,th{{border:1px solid #ccc;padding:6px;vertical-align:top;font-size:14px}}
tr.prio{{background:#fff4d6}} .kn{{font-size:16px;line-height:1.5}} .dose{{margin-top:6px;font-weight:600}}
.ai{{margin-top:6px;color:#666;font-style:italic}} section{{margin-bottom:48px}}
</style></head><body>
<h1>Fact review pack</h1>
<p>For each page: look at the page image, then check every record under it.
Is each number in the right row and column for that pest, crop stage or variety? Does the text say what the page says?
Record your decision in <code>backend/knowledge/review_sheet.csv</code> at the given sheet row
(CONFIRMED / WRONG / UNSURE), then save it as <code>backend/knowledge/review_sheet_REVIEWED.csv</code>.
⭐ rows ({n_priority}) have doses or table-layout risks: check those first.
Grey "AI suggestion" lines are from an AI pre-review; they are not confirmations.</p>
{''.join(sections)}</body></html>"""
    with open(os.path.join(OUT_DIR, "index.html"), "w", encoding="utf-8") as f:
        f.write(page_html)
    print(f"{len(sections)} pages, {n_priority} priority records -> {os.path.join(OUT_DIR, 'index.html')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ai-prereview")
    main(ap.parse_args().ai_prereview)
