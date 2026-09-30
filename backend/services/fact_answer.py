"""
v2 answer path: answer only from person-confirmed fact records.

No LLM is involved. A question is matched against the pest/disease names
and topic keywords in knowledge/intents.json; the matching records are
shown exactly as stored (text, doses, source, page). Anything else is
refused with the Kisan Call Centre number.

Safety rules this module enforces (tests/invariants/test_fact_answer.py):
  - Pest/disease records are reachable ONLY through a named pest/disease
    alias. A question that mentions an organism (worm, pest, fly...) but
    names no pest/disease we know is refused, even if a topic keyword also
    matched (the H-38 / pest-5 failure: a generic word
    like "worm" must never select another pest's dose).
  - Only records a person marked CONFIRMED in the review sheet are used,
    and only while the record still has the content that was confirmed
    (record_hash). Editing a record voids its confirmation.
  - A record with a product needs the reviewer's "currently registered"
    answer to be yes (or n.a. for biological agents).
  - South (UAS-B) and North (UAS-D) recommendations are shown under
    separate headings and never merged.
"""
import csv
import glob
import hashlib
import json
import os
import re

KNOWLEDGE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "knowledge")
REVIEWED_SHEET = os.path.join(KNOWLEDGE_DIR, "review_sheet_REVIEWED.csv")

# User-facing Kannada strings written by Claude. They need a Kannada
# speaker's review like everything else the bot says (TODO.md).
KCC_LINE = "ಹೆಚ್ಚಿನ ಸಹಾಯಕ್ಕಾಗಿ, ಕಿಸಾನ್ ಕಾಲ್ ಸೆಂಟರ್‌ಗೆ ಕರೆ ಮಾಡಿ: 1800-180-1551 ಅಥವಾ ಹತ್ತಿರದ ರೈತ ಸಂಪರ್ಕ ಕೇಂದ್ರವನ್ನು ಸಂಪರ್ಕಿಸಿ."
MSG_UNKNOWN_PEST = (
    "ಕ್ಷಮಿಸಿ, ನೀವು ಕೇಳಿದ ಕೀಟ ಅಥವಾ ರೋಗದ ಬಗ್ಗೆ ಪರಿಶೀಲಿತ ಮಾಹಿತಿ ನಮ್ಮಲ್ಲಿ ಇಲ್ಲ. "
    "ತಪ್ಪು ಔಷಧಿ ಬಳಸುವುದು ಬೆಳೆಗೂ ನಿಮಗೂ ಅಪಾಯಕಾರಿ, ಆದ್ದರಿಂದ ಊಹಿಸಿ ಉತ್ತರ ಕೊಡುವುದಿಲ್ಲ."
)
MSG_NOT_VERIFIED = (
    "ಕ್ಷಮಿಸಿ, ಈ ವಿಷಯದ ಬಗ್ಗೆ ತಜ್ಞರಿಂದ ಪರಿಶೀಲನೆಯಾದ ಮಾಹಿತಿ ಇನ್ನೂ ನಮ್ಮಲ್ಲಿ ಇಲ್ಲ."
)
MSG_OUT_OF_SCOPE = (
    "ಕ್ಷಮಿಸಿ, ಈ ಪ್ರಶ್ನೆಗೆ ನನ್ನಲ್ಲಿ ಮಾಹಿತಿ ಇಲ್ಲ. ನಾನು ಕಬ್ಬು ಬೆಳೆಯ ಈ ವಿಷಯಗಳ ಬಗ್ಗೆ ಮಾತ್ರ ಉತ್ತರಿಸುತ್ತೇನೆ: "
    "ತಳಿಗಳು, ನಾಟಿ ಕಾಲ, ಬಿತ್ತನೆ ಪ್ರಮಾಣ, ಬೀಜೋಪಚಾರ, ಗೊಬ್ಬರ, ನೀರಾವರಿ, ಕಳೆ ನಿಯಂತ್ರಣ, "
    "ಪ್ರಮುಖ ಕೀಟ ಮತ್ತು ರೋಗಗಳು, ಕಟಾವು ಮತ್ತು ಕೂಳೆ ಬೆಳೆ."
)
MSG_BOTH_REGIONS = (
    "ದಕ್ಷಿಣ ಮತ್ತು ಉತ್ತರ ಕರ್ನಾಟಕದ ಶಿಫಾರಸ್ಸುಗಳು ಬೇರೆ ಬೇರೆ ಇವೆ. ನಿಮ್ಮ ಪ್ರದೇಶದ ಶಿಫಾರಸ್ಸನ್ನು ಮಾತ್ರ ಅನುಸರಿಸಿ."
)
MSG_DRAFT_BANNER = (
    "⚠️ ಪರೀಕ್ಷಾ ಮೋಡ್: ಕೆಳಗಿನ ಮಾಹಿತಿ ಇನ್ನೂ ಪರಿಶೀಲನೆಯಾಗಿಲ್ಲ (DRAFT). ರೈತರು ಇದನ್ನು ಬಳಸಬಾರದು."
)
MSG_MISSING_SUBJECT = "ಈ ಕೀಟ/ರೋಗದ ಬಗ್ಗೆ ಪರಿಶೀಲಿತ ಮಾಹಿತಿ ಇಲ್ಲ: {names}."

SAFETY_RECORD_ID = "uasd-pesticide-safety"
HASHED_FIELDS = ("text_kn", "product_kn", "per_litre", "per_acre", "water_per_acre", "printed_page", "subject_kn")

_ZERO_WIDTH = re.compile("[​‌‍﻿]")
_WS = re.compile(r"\s+")
_WORD_CHAR = "[ಀ-೿A-Za-z0-9]"


def normalize(text):
    """Matching form: zero-width joiners removed, lower-cased, spaces collapsed."""
    return _WS.sub(" ", _ZERO_WIDTH.sub("", text or "")).strip().lower()


def record_hash(source, record):
    """Fingerprint of what a reviewer confirms; changes if any farmer-visible field changes."""
    payload = {"source": source, "id": record["id"], **{f: record.get(f, "") for f in HASHED_FIELDS}}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:16]


class Knowledge:
    """Records, sources, intents and review statuses, loaded from knowledge/."""

    def __init__(self, records, sources, intents, statuses):
        self.records = records          # id -> record (with "_source" added)
        self.sources = sources          # source key -> source header dict
        self.intents = intents          # parsed intents.json
        self.statuses = statuses        # id -> review row dict
        self._entries = self._build_entries()

    @classmethod
    def load(cls, knowledge_dir=KNOWLEDGE_DIR, reviewed_sheet=REVIEWED_SHEET):
        records, sources = {}, {}
        for path in sorted(glob.glob(os.path.join(knowledge_dir, "facts_*.json"))):
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            sources[data["source"]] = {k: v for k, v in data.items() if k != "records"}
            for r in data["records"]:
                records[r["id"]] = {**r, "_source": data["source"]}
        with open(os.path.join(knowledge_dir, "intents.json"), encoding="utf-8") as f:
            intents = json.load(f)
        statuses = {}
        if os.path.exists(reviewed_sheet):
            with open(reviewed_sheet, encoding="utf-8-sig") as f:
                for row in csv.DictReader(f):
                    statuses[row["id"]] = row
        return cls(records, sources, intents, statuses)

    def _build_entries(self):
        entries = []
        for key, subj in self.intents["subjects"].items():
            entries += [(a, "subject", key) for a in subj["aliases"]]
        for key, intent in self.intents["intents"].items():
            entries += [(k, "intent", key) for k in intent["keywords"]]
        entries += [(w, "organism", None) for w in self.intents["organism_words"]]
        entries += [(w, "neutral", None) for w in self.intents["neutral_phrases"]]
        entries += [(w, "other_crop", None) for w in self.intents.get("other_crops", [])]
        entries += [(w, "own_crop", None) for w in self.intents.get("own_crop", [])]
        compiled = []
        for kw, kind, owner in entries:
            whole = kw.startswith("=")
            text = normalize(kw[1:] if whole else kw)
            pattern = re.escape(text)
            if whole:
                pattern = f"(?<!{_WORD_CHAR}){pattern}(?!{_WORD_CHAR})"
            compiled.append((len(text), re.compile(pattern), kind, owner))
        compiled.sort(key=lambda e: -e[0])     # longest phrase first
        return compiled

    def usable(self, record_id, include_drafts=False):
        """True if a person confirmed this exact record (and its product is registered)."""
        rec = self.records.get(record_id)
        if rec is None:
            return False
        if include_drafts:
            return True
        row = self.statuses.get(record_id)
        if not row:
            return False
        status = _column(row, "REVIEW: status").strip().upper()
        if status != "CONFIRMED":
            return False
        if _column(row, "record_hash").strip() != record_hash(rec["_source"], rec):
            return False
        if rec.get("product_kn"):
            registered = _column(row, "REVIEW: pesticide currently registered").strip().lower()
            if registered not in ("yes", "n.a.", "na", "n.a"):
                return False
        return True


def _column(row, prefix):
    for k, v in row.items():
        if k and k.startswith(prefix):
            return v or ""
    return ""


def match(knowledge, query):
    """
    Which pest/disease subjects, topic intents and region the query names.
    Longest phrases are matched first and consume their characters, so a
    shorter phrase inside an already-matched one cannot match again.
    """
    q = normalize(query)
    taken = [False] * len(q)
    subjects, intents, organism, other_crop, own_crop = [], [], False, False, False
    for _, pattern, kind, owner in knowledge._entries:
        for m in pattern.finditer(q):
            if any(taken[m.start():m.end()]):
                continue
            for i in range(m.start(), m.end()):
                taken[i] = True
            if kind == "subject" and owner not in subjects:
                subjects.append(owner)
            elif kind == "intent" and owner not in intents:
                intents.append(owner)
            elif kind == "organism":
                organism = True
            elif kind == "other_crop":
                other_crop = True
            elif kind == "own_crop":
                own_crop = True
    for comp in knowledge.intents["composites"]:
        if all(w in intents for w in comp["when"]):
            intents = [i for i in intents if i not in comp["when"]] + [comp["use"]]
    helpers = {k for k, v in knowledge.intents["intents"].items() if v.get("helper")}
    intents = [i for i in intents if i not in helpers]
    if any(i.startswith("variety_") for i in intents):
        intents = [i for i in intents if i != "variety"]   # a named variety beats the generic word
    region = None
    for reg, names in knowledge.intents["regions"].items():
        if any(normalize(n) in q for n in names):
            region = reg if region is None else "both"
    return {"subjects": subjects, "intents": intents, "organism_word": organism,
            "other_crop": other_crop and not own_crop,
            "region": None if region == "both" else region}


def answer(knowledge, query, region=None, include_drafts=False):
    """
    Returns {"answer", "status", "record_ids", "citations", "match"}.
    status: answered | refused_unknown_pest | refused_not_verified | refused_out_of_scope
    """
    m = match(knowledge, query)
    region = region or m["region"]
    table = knowledge.intents["intents"]

    if m["other_crop"]:
        return _refusal(MSG_OUT_OF_SCOPE, "refused_out_of_scope", m)
    if any(table[i].get("refuse") for i in m["intents"]):
        return _refusal(MSG_NOT_VERIFIED, "refused_not_verified", m)

    if m["subjects"]:
        wanted = _records_for(knowledge.intents["subjects"], m["subjects"])
        wanted += _records_for(knowledge.intents["intents"], m["intents"])
        uncovered = [knowledge.intents["subjects"][s]["name_kn"] for s in m["subjects"]
                     if not knowledge.intents["subjects"][s]["records"]]
    elif m["organism_word"]:
        return _refusal(MSG_UNKNOWN_PEST, "refused_unknown_pest", m)
    elif m["intents"]:
        wanted = _records_for(knowledge.intents["intents"], m["intents"])
        uncovered = []
    else:
        return _refusal(MSG_OUT_OF_SCOPE, "refused_out_of_scope", m)

    usable = [r for r in _dedupe(wanted) if knowledge.usable(r, include_drafts)]
    if region in ("south_karnataka", "north_karnataka"):
        usable = [r for r in usable
                  if knowledge.sources[knowledge.records[r]["_source"]]["region"] == region]
    if not usable:
        msg = MSG_NOT_VERIFIED
        if uncovered:
            msg = MSG_MISSING_SUBJECT.format(names=", ".join(uncovered)) + " " + MSG_UNKNOWN_PEST
        return _refusal(msg, "refused_not_verified", m)
    return _render(knowledge, usable, m, uncovered, include_drafts)


def _records_for(table, keys):
    out = []
    for k in keys:
        out += table[k]["records"]
    return out


def _dedupe(ids):
    seen, out = set(), []
    for i in ids:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def _refusal(message, status, m):
    return {"answer": f"{message}\n\n{KCC_LINE}", "status": status,
            "record_ids": [], "citations": [], "match": m}


def _render(knowledge, record_ids, m, uncovered, include_drafts):
    by_source = {}
    for rid in record_ids:
        by_source.setdefault(knowledge.records[rid]["_source"], []).append(rid)
    parts, citations = [], []
    if include_drafts:
        parts.append(MSG_DRAFT_BANNER)
    if len(by_source) > 1:
        parts.append(MSG_BOTH_REGIONS)
    has_product = False
    for source, ids in by_source.items():
        src = knowledge.sources[source]
        lines = [f"📘 {src['display_name_kn']} ({src['region_kn']}):"]
        for rid in ids:
            rec = knowledge.records[rid]
            has_product = has_product or bool(rec.get("product_kn"))
            lines.append(f"• {rec['text_kn']} (ಪುಟ {rec['printed_page']})")
            citations.append(f"{src['source_title']}, p. {rec['printed_page']} [{rid}]")
        parts.append("\n".join(lines))
    if uncovered:
        parts.append(MSG_MISSING_SUBJECT.format(names=", ".join(uncovered)))
    if has_product and knowledge.usable(SAFETY_RECORD_ID, include_drafts):
        parts.append("⚠️ " + knowledge.records[SAFETY_RECORD_ID]["text_kn"])
    parts.append(KCC_LINE)
    return {"answer": "\n\n".join(parts), "status": "answered",
            "record_ids": record_ids, "citations": citations, "match": m}
