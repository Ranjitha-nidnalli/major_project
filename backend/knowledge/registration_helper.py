"""
Registration helper for the person filling in "currently registered?" in
review_sheet.csv. For every chemical our fact records mention, it lists the
SUGARCANE entries in the CIB&RC "Major Uses of Pesticides" lists (as on
31.03.2026) and any mention in the banned/refused/restricted list (as on
31.07.2026), with page numbers, so the person can check them quickly.

It does NOT decide anything: text extraction from these PDFs can split or
merge table cells, so every hit (and every "no hit") must be confirmed on
the PDF page. It also never compares or converts doses: CIB&RC gives
per-hectare product rates, the UAS books per-acre ones.

Usage:
    cd backend && python knowledge/registration_helper.py
Needs the PDFs in backend/corpus/cibrc/ (gitignored; URLs in
backend/corpus/SOURCES.md). Writes knowledge/registration_helper.csv.
"""
import csv
import os
import re

import fitz  # PyMuPDF

KNOWLEDGE_DIR = os.path.dirname(os.path.abspath(__file__))
CIBRC_DIR = os.path.join(os.path.dirname(KNOWLEDGE_DIR), "corpus", "cibrc")
OUT = os.path.join(KNOWLEDGE_DIR, "registration_helper.csv")
MUP_FILES = {
    "insecticide": "updated_mup_insecticide_as_on_31.03.2026_c.pdf",
    "fungicide": "2._chemical_mup_fungicide_as_on_31.03.2026_0.pdf",
    "herbicide": "4._herbicides_mup_as_on_31.03.2026.pdf",
}
BANNED_FILE = "list_of_pesticides_which_are_banned_refused_registration_and_restricted_in_use.pdf"

# Chemical -> (search pattern, fact record ids that use it).
CHEMICALS = {
    "Chlorpyrifos": (r"chlorpyri?phos|chlorpyrifos", ["uasb-esb-chlorpyrifos", "uasb-scale-mealybug-chlorpyrifos", "uasb-wwa-chlorpyrifos", "uasd-termite", "uasd-sett-treatment", "uasd-wwa-chlorpyrifos"]),
    "Chlorantraniliprole": (r"chlorantraniliprole", ["uasb-esb-chlorantraniliprole"]),
    "Fipronil": (r"fipronil", ["uasb-esb-fipronil"]),
    "Dimethoate": (r"dimethoate", ["uasb-scale-mealybug-dimethoate", "uasb-wwa-dimethoate"]),
    "Acephate": (r"acephate", ["uasd-wwa-acephate"]),
    "Malathion": (r"malathion", ["uasd-wwa-malathion", "uasd-wwa-malathion-dust"]),
    "Thiamethoxam": (r"thiamethoxam", ["uasd-wwa-thiamethoxam"]),
    "Carbendazim": (r"carbendazim", ["uasb-sett-treatment", "uasb-pineapple-disease", "uasd-sett-treatment", "uasd-smut", "uasd-wilt"]),
    "Copper oxychloride": (r"copper\s*oxy\s*chloride", ["uasb-leaf-spot-copper"]),
    "Mancozeb": (r"mancozeb", ["uasb-leaf-spot-mancozeb"]),
    "Atrazine": (r"atrazine", ["uasb-weed-atrazine", "uasd-weed-atrazine", "uasd-ratoon-weed"]),
    "Metribuzin": (r"metribuzin", ["uasb-weed-metribuzin"]),
    "Diuron": (r"diuron", ["uasb-weed-diuron"]),
    "2,4-D": (r"2\s*,\s*4\s*-?\s*d\b", ["uasb-weed-24d-post", "uasd-weed-24d"]),
}
# A product header line: a name followed by a strength and formulation code, e.g. "Bifenthrin 10 %EC".
_PRODUCT = re.compile(r"^[A-Z0-9][^\n]{1,80}?\d+(?:\.\d+)?\s*%\s*(?:w/w|w/v)?\s*[A-Z]{1,4}\b")
_SUGARCANE = re.compile(r"sugar\s*-?\s*cane", re.I)


def _lines(path):
    doc = fitz.open(path)
    for pno in range(doc.page_count):
        for line in doc[pno].get_text().splitlines():
            line = line.strip()
            if line:
                yield pno + 1, line


def sugarcane_entries(path):
    """[(product_header, page, the ~6 lines after the 'Sugarcane' line)]"""
    entries, product, pending = [], None, []
    lines = list(_lines(path))
    for i, (page, line) in enumerate(lines):
        if _PRODUCT.match(line):
            product = line
        if _SUGARCANE.search(line) and product:
            following = " | ".join(l for _, l in lines[i + 1:i + 7])
            entries.append((product, page, following))
    return entries


def main():
    entries = []
    for kind, fname in MUP_FILES.items():
        entries += [(kind, *e) for e in sugarcane_entries(os.path.join(CIBRC_DIR, fname))]
    banned_pages = {}
    for page, line in _lines(os.path.join(CIBRC_DIR, BANNED_FILE)):
        banned_pages.setdefault(page, []).append(line)

    rows = []
    for chem, (pattern, record_ids) in CHEMICALS.items():
        rx = re.compile(pattern, re.I)
        hits = [e for e in entries if rx.search(e[1])]
        banned = [f"p.{p}: {l}" for p, ls in banned_pages.items() for l in ls if rx.search(l)]
        if not hits:
            rows.append([chem, "; ".join(record_ids), "NO sugarcane entry found (check the PDF)", "", "", "",
                         " || ".join(banned) or "not found"])
        for kind, product, page, following in hits:
            rows.append([chem, "; ".join(record_ids), product, kind, page, following,
                         " || ".join(banned) or "not found"])
    with open(OUT, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["chemical", "our records using it", "CIB&RC product (sugarcane entry)", "list",
                    "PDF page", "text after 'Sugarcane' (pest, a.i. g/ha, product g-or-ml/ha, water L/ha, waiting days)",
                    "banned/refused/restricted list mentions"])
        w.writerows(rows)
    found = sorted({r[0] for r in rows if not r[2].startswith("NO")})
    missing = sorted({r[0] for r in rows if r[2].startswith("NO")})
    print(f"{len(rows)} rows -> {OUT}\nwith sugarcane entries: {found}\nNO sugarcane entry: {missing}")


if __name__ == "__main__":
    main()
