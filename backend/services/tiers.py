"""
Tiered answer policy (story K-007): one answer() with three tiers.

  A  Verified records. Dose, pesticide, fertiliser and seed-rate answers come
     ONLY from person-confirmed, hash-checked, registered fact records
     (services/fact_answer.py). No LLM writes a number.
  B  General advice. Retrieval + abstention gate + numeric check (the v1
     pipeline), for questions that name no pest, no dose and no product, and
     only when every retrieved chunk has a citable source. Off by default.
  C  Refuse and give the Kisan Call Centre number. Also the fail-closed result
     for anything unsure or any dependency error.

Safety rules this module enforces (tests/invariants/test_tiers.py):
  - Tier A runs first on every question. A classifier or router can never make
    a confirmed record unreachable, because nothing is consulted before it.
  - Tier B is considered only AFTER Tier A refused, and only when the fact
    matcher found no pest/disease, no topic, no organism word and no other
    crop, and the question has no dose/pesticide/fertiliser/rate wording.
    Dose, pesticide, rate and fertiliser questions never reach Tier B.
  - A router or retrieval category can only TIGHTEN: pest, disease, fertilizer
    or price from either source sends the question to Tier C.
  - A Tier B answer that contains a dose-like number (a number with a weight,
    volume, percent or ppm unit) is replaced by Tier C, whatever the context says.
  - A Tier B answer needs a citation for every chunk it used; a chunk with no
    source field is not citable, so the answer becomes Tier C.
  - Any exception in Tier B is Tier C.

This module is pure: no LLM, database or embedding imports. The caller passes
in async functions, so the policy is unit-testable with fakes.
"""
import re

from services.fact_answer import KCC_LINE, MSG_NOT_VERIFIED, normalize

TIER_A = "A"
TIER_B = "B"
TIER_C = "C"

KCC_NUMBER = "1800-180-1551"

# Mirrors rag_service.SAFETY_CRITICAL_CATEGORIES plus price (never answered here).
NEVER_TIER_B_CATEGORIES = frozenset({"pest", "disease", "fertilizer", "price"})

# Wording that asks for a dose, a product, a rate or a fertiliser. Kannada terms
# are stems matched as substrings (the language is agglutinative); short English
# tokens are matched as whole words. When in doubt the list errs towards
# refusing. These lists need a Kannada speaker's review (TODO.md).
_KN_TERMS = (
    "ಔಷಧ", "ಕೀಟನಾಶಕ", "ಶಿಲೀಂಧ್ರನಾಶಕ", "ಕಳೆನಾಶಕ", "ಸಿಂಪಡ", "ಸಿಂಪರಣ",
    "ಪ್ರಮಾಣ", "ಡೋಸ್", "ಮಿ.ಲೀ", "ಮಿಲಿ", "ಗ್ರಾಂ", "ಕೆ.ಜಿ", "ಕೆಜಿ", "ಕಿಲೋ",
    "ಲೀಟರ್", "ಎಕರೆಗೆ", "ಹೆಕ್ಟೇರ", "ಗೊಬ್ಬರ", "ಯೂರಿಯಾ", "ಪೊಟ್ಯಾ", "ಡಿ.ಎ.ಪಿ",
    "ಡಿಎಪಿ", "ರಂಜಕ", "ಸಾರಜನಕ", "ಪೊಟ್ಯಾಷ", "ಬೀಜ ದರ", "ಬೀಜದ ದರ", "ಎಷ್ಟು",
)
_EN_PHRASES = (
    "pesticide", "insecticide", "fungicide", "herbicide", "weedicide", "fertili",
    "per acre", "per hectare", "how much", "dosage",
)
_EN_WORDS = re.compile(
    r"\b(dose|spray|sprays|chemical|chemicals|urea|dap|npk|kg|kgs|gram|grams|"
    r"ml|litre|litres|liter|liters|rate)\b"
)

_NUM = r"[0-9೦-೯]+(?:[.,][0-9೦-೯]+)?"
_UNITS = (
    r"(?:kgs?|gms?|grams?|mg|ml|lit(?:re|er)?s?|ppm|%|percent|tonnes?|tons?|quintals?|"
    r"ಕೆ\.?\s?ಜಿ|ಕೆಜಿ|ಕಿಲೋ|ಗ್ರಾಂ|ಗ್ರಾಮ್|ಮಿ\.?\s?ಲೀ|ಮಿಲಿ|ಲೀಟರ್|ಲೀ|ಟನ್|ಕ್ವಿಂಟಾಲ್|ಶೇ|"
    r"ಪಿ\.?\s?ಪಿ\.?\s?ಎಂ|g|l)"
)
_DOSE_LIKE = re.compile(
    rf"{_NUM}(?:\s*[-–]\s*{_NUM})?\s*{_UNITS}(?![A-Za-z])", re.IGNORECASE
)


def mentions_dose_or_pesticide(query):
    """The first dose/pesticide/fertiliser/rate term found in the query, or None."""
    q = normalize(query)
    for term in _KN_TERMS:
        if term in q:
            return term
    for term in _EN_PHRASES:
        if term in q:
            return term
    found = _EN_WORDS.search(q)
    return found.group(0) if found else None


def tier_b_blockers(query, fact_match, router_category=None):
    """
    Reasons the question must NOT reach Tier B (empty list = Tier B is allowed).
    `fact_match` is the `match` dict from services.fact_answer. The router
    category can only add a blocker, never remove one.
    """
    reasons = []
    m = fact_match or {}
    if m.get("subjects"):
        reasons.append("names_pest_or_disease")
    if m.get("organism_word"):
        reasons.append("organism_word")
    if m.get("intents"):
        reasons.append("known_topic_without_confirmed_record")
    if m.get("other_crop"):
        reasons.append("other_crop")
    term = mentions_dose_or_pesticide(query)
    if term:
        reasons.append(f"dose_or_pesticide_wording:{term}")
    if router_category in NEVER_TIER_B_CATEGORIES:
        reasons.append(f"category:{router_category}")
    return reasons


def has_dose_like_numbers(text):
    """True if the text has a number with a weight, volume, percent or ppm unit."""
    return bool(_DOSE_LIKE.search(normalize(text)))


def citations_from_payloads(payloads):
    """
    One citation string per chunk that has a source, or None if ANY chunk lacks
    one (a partly-cited answer is not citable). The current corpus has no
    `source` field, so today this returns None and Tier B refuses.
    """
    cites = []
    for p in payloads or []:
        source = str((p or {}).get("source") or "").strip()
        if not source:
            return None
        page = str((p or {}).get("page") or "").strip()
        cites.append(f"{source}, p. {page}" if page else source)
    return cites or None


def check_tier_b_answer(answer, payloads, final_category):
    """(ok, reason, citations) for a generated Tier B answer."""
    if not (answer or "").strip():
        return False, "empty_answer", []
    if final_category in NEVER_TIER_B_CATEGORIES:
        return False, f"retrieval_category:{final_category}", []
    if has_dose_like_numbers(answer):
        return False, "dose_like_number_in_answer", []
    cites = citations_from_payloads(payloads)
    if not cites:
        return False, "no_citable_source", []
    return True, "ok", cites


def _with_kcc(text):
    return text if KCC_NUMBER in text else f"{text}\n\n{KCC_LINE}"


async def answer_tiered(query, *, facts_fn, rag_fn, tier_b_enabled=False, classify_fn=None):
    """
    The one policy. `facts_fn` is `async () -> dict` shaped like
    services.fact_answer.answer(); `rag_fn` is `async () -> dict` shaped like
    the v1 pipeline result (answer, refused, category, source_payloads).
    `classify_fn` (optional, `async (query) -> category | None`, e.g. a small
    LLM) is consulted only after Tier A refused and Tier B is otherwise
    allowed. It can only tighten: a pest/disease/fertilizer/price category, or
    any error, sends the question to Tier C. It can never reach Tier A.
    Fail closed (K-010): if the fact lookup itself raises, the answer is
    Tier C with the KCC line, never an exception.
    Returns {"tier", "tier_reason", "answer", "citations", "facts", "rag"}.
    """
    try:
        facts = await facts_fn()
    except Exception as exc:  # knowledge files, canonicaliser or sheet unavailable
        facts = {"status": "error", "answer": MSG_NOT_VERIFIED, "record_ids": [],
                 "citations": [], "match": {}}
        return {"tier": TIER_C, "tier_reason": f"facts_error:{type(exc).__name__}",
                "answer": _with_kcc(facts["answer"]), "citations": [], "facts": facts, "rag": None}
    if facts.get("status") == "answered":
        return {"tier": TIER_A, "tier_reason": "confirmed_records",
                "answer": facts["answer"], "citations": facts.get("citations", []),
                "facts": facts, "rag": None}

    def tier_c(reason, rag=None):
        return {"tier": TIER_C, "tier_reason": reason, "answer": _with_kcc(facts["answer"]),
                "citations": [], "facts": facts, "rag": rag}

    blockers = tier_b_blockers(query, facts.get("match"))
    if blockers:
        return tier_c(";".join(blockers))
    if not tier_b_enabled:
        return tier_c("tier_b_disabled")
    if classify_fn is not None:
        try:
            category = await classify_fn(query)
        except Exception as exc:
            return tier_c(f"classifier_error:{type(exc).__name__}")
        extra = tier_b_blockers(query, facts.get("match"), category)
        if extra:
            return tier_c(";".join(extra))

    try:
        rag = await rag_fn()
    except Exception as exc:  # fail closed
        return tier_c(f"tier_b_error:{type(exc).__name__}")
    if rag.get("refused"):
        return tier_c("tier_b_refused", rag)
    ok, reason, cites = check_tier_b_answer(
        rag.get("answer"), rag.get("source_payloads"), rag.get("category"))
    if not ok:
        return tier_c(f"tier_b_rejected:{reason}", rag)
    sources = "\n".join(f"- {c}" for c in cites)
    return {"tier": TIER_B, "tier_reason": "retrieval_gate_passed",
            "answer": _with_kcc(f"{rag['answer']}\n\nಮೂಲ:\n{sources}"),
            "citations": cites, "facts": facts, "rag": rag}
