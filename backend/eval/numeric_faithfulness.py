"""
numeric_faithfulness.py

Extracts numbers + units from generated answers and cross-references them
against numbers + units found in the retrieved context.

Closes the highest-consequence safety gap: an LLM can be semantically faithful
("it's a dosage") while numerically wrong ("100 grams" vs "10 grams").

Usage:
    from numeric_faithfulness import check_numeric_faithfulness
    score, violations = check_numeric_faithfulness(context_text, answer_text)
    # score: 1.0 = all numbers in answer appear in context
    # violations: list of (answer_number, context_numbers_found)
"""
import re
from typing import List, Tuple, Dict, Set

# Regex for numbers: Arabic numerals (with optional decimals) and Kannada digits
_ARABIC_NUM = r"[0-9]+(?:\.[0-9]+)?"
_KANNADA_DIGITS = r"[೦೧೨೩೪೫೬೭೮೯]+(?:\.[೦೧೨೩೪೫೬೭೮೯]+)?"
_NUMBER_PATTERN = re.compile(rf"(?:{_ARABIC_NUM}|{_KANNADA_DIGITS})")

# Common agricultural units in Kannada / English / mixed
_UNITS = [
    # Weight
    r"ಗ್ರಾಂ", r"ಗ್ರಾಮ", r"gram", r"grams", r"g",
    r"ಕೆಜಿ", r"kg",
    r"ಟನ್", r"ton", r"tons", r"tonne", r"tonnes",
    # Lowercase only: the corpus writes tonnes as "100t" / "(5-7 t)", but
    # its varietal tables use "T Ha" as a field LABEL ("Ccs Percent: 14.2
    # Ccs T Ha: 17.5"), and a case-insensitive "t" bound 14.2 to tonnes.
    # Written pre-bounded, since _bounded() only wraps plain-letter units.
    r"(?<![A-Za-z'’])(?-i:t)(?![A-Za-z])",
    r"ಕ್ವಿಂಟಾಲ್", r"quintal", r"quintals",
    # Volume
    r"ಲೀಟರ್", r"ಲೀ", r"litre", r"liter", r"litres", r"liters", r"l",
    r"ಮಿಲಿ", r"ml",
    # Area
    r"ಎಕರೆ", r"acre", r"acres",
    r"ಹೆಕ್ಟೇರ್", r"hectare", r"hectares",
    r"ಗುಂಟ", r"ಗುಂಟೆ", r"guntas?",
    # Concentration / ratio
    r"ಪರ್ಸೆಂಟ್", r"%", r"percent",
    r"ಪಿ\.ಪಿ\.ಎಮ್", r"ppm",
    r"ಡಬ್ಲ್ಯೂ\.ಪಿ", r"ಇ\.ಸಿ", r"ಜಿ",
    # Time
    r"ನಿಮಿಷ", r"minute", r"minutes", r"mins?",
    r"ಗಂಟೆ", r"hour", r"hours",
    r"ದಿನ", r"day", r"days",
    r"ವಾರ", r"week", r"weeks",
    r"ತಿಂಗಳು", r"month", r"months",
    # Count
    r"ಸೆಟ್ಸ್", r"setts?",
    r"ಸಸಿಗಳು", r"seedlings?",
]

def _bounded(unit: str) -> str:
    """
    Latin-script units must stand alone, not match inside a word. Without
    this, single-letter units matched inside ordinary English: in
    "10 min is crucial" the "l" of "crucial" bound 10 to litre (TODO #49).
    An apostrophe before the unit also blocks it, so the "t" in "don't" is
    not read as tonnes. Kannada units are left unbounded: Python's \\w
    does not treat Kannada vowel signs as word characters, so \\b-style
    boundaries would split real Kannada words.
    """
    if re.fullmatch(r"[A-Za-z?]+", unit):
        return rf"(?<![A-Za-z'’]){unit}(?![A-Za-z])"
    return unit


_UNIT_PATTERN = re.compile(
    r"(?:" + "|".join(_bounded(u) for u in _UNITS) + r")",
    re.IGNORECASE,
)

# A number's unit must appear before the next number or a sentence boundary --
# "up to N words away" is not a safe proxy for "belongs to this number", since
# it lets a regex bind a number to a unit that actually belongs to a
# different, later number/quantity in the same sentence.
_SENTENCE_BOUNDARY = re.compile(r"[.!?।]")

# A field label ending in a unit, directly before a number: "Duration
# Months: 10", "Ccs Percent: 14.2", "Seed Rate Tonnes: 5". "T Ha" (the
# varietal tables' tonnes-per-hectare label) is group 1.
_LABEL_UNIT_BEFORE = re.compile(
    r"(?:(?<![A-Za-z])(T\s+Ha)"
    r"|(?<![A-Za-z])(months?|percent|tonnes?|tons?|days?|weeks?|hours?|minutes?))"
    r"\s*:\s*$",
    re.IGNORECASE,
)

# What may sit between the two ends of a range: "10 - 11", "5–7", "10 to 11".
# Includes U+2010-U+2012: gpt-oss writes ranges with "‑" (non-breaking hyphen).
_RANGE_CONNECTOR = re.compile(r"\s*(?:[-‐‑‒–—]|to)\s*", re.IGNORECASE)


def _normalize_kannada_number(s: str) -> str:
    """Convert Kannada digits to Arabic numerals for comparison."""
    kannada_to_arabic = str.maketrans("೦೧೨೩೪೫೬೭೮೯", "0123456789")
    return s.translate(kannada_to_arabic)


def _canonicalize_unit(unit_str: str) -> str:
    u = unit_str.strip().lower()
    if u in ("ಗ್ರಾಂ", "ಗ್ರಾಮ", "gram", "grams", "g"):
        return "gram"
    if u in ("ಕೆಜಿ", "kg"):
        return "kg"
    if u in ("ಲೀಟರ್", "ಲೀ", "litre", "liter", "litres", "liters", "l"):
        return "litre"
    if u in ("ಮಿಲಿ", "ml"):
        return "ml"
    if u in ("ಎಕರೆ", "acre", "acres"):
        return "acre"
    if u in ("ಹೆಕ್ಟೇರ್", "hectare", "hectares"):
        return "hectare"
    if u in ("%", "ಪರ್ಸೆಂಟ್", "percent"):
        return "percent"
    if u in ("ನಿಮಿಷ", "minute", "minutes", "min", "mins"):
        return "minute"
    if u in ("ಗಂಟೆ", "hour", "hours"):
        return "hour"
    if u in ("ದಿನ", "day", "days"):
        return "day"
    if u in ("ವಾರ", "week", "weeks"):
        return "week"
    if u in ("ತಿಂಗಳು", "month", "months"):
        return "month"
    if u in ("ಸೆಟ್ಸ್", "setts", "sett"):
        return "sett"
    if u in ("ಟನ್", "ton", "tons", "tonne", "tonnes", "t"):
        return "ton"
    if u in ("ಕ್ವಿಂಟಾಲ್", "quintal", "quintals"):
        return "quintal"
    if u in ("ಡಬ್ಲ್ಯೂ.ಪಿ", "wp"):
        return "wp"
    if u in ("ಇ.ಸಿ", "ec"):
        return "ec"
    if u in ("ಜಿ", "g_granule"):
        return "g_granule"
    return u


def extract_number_units(text: str) -> Set[str]:
    """
    Extract all number+unit tuples from text.
    Returns a set of normalized strings like '10.0_gram' for easy comparison.

    A number is bound only to the nearest unit token that appears strictly
    before the NEXT number (or a sentence boundary, or end of string) --
    never to a unit that actually belongs to a different number later in
    the sentence.
    """
    number_matches = list(_NUMBER_PATTERN.finditer(text))
    items = []  # (match, value, unit or "")
    for i, m in enumerate(number_matches):
        num_norm = _normalize_kannada_number(m.group())
        try:
            num_val = float(num_norm)
        except ValueError:
            continue

        # "Label unit: value" layout (TODO #49a): the corpus's structured
        # fields put the unit in the label BEFORE the number ("Duration
        # Months: 10", "Ccs Percent: 14.2", "Cane Yield T Ha: 123.5"). An
        # explicit field label wins over looking ahead.
        label = _LABEL_UNIT_BEFORE.search(text[max(0, m.start() - 30):m.start()])
        if label:
            items.append((m, num_val, "ton" if label.group(1) else _canonicalize_unit(label.group(2))))
            continue

        window_end = number_matches[i + 1].start() if i + 1 < len(number_matches) else len(text)
        window = text[m.end():window_end]

        # Don't just slice the window at the first ".", "!", etc. -- unit
        # abbreviations like "ಡಬ್ಲ್ಯೂ.ಪಿ" (WP) contain a literal period, and
        # slicing there would cut the unit token in half before the unit
        # regex ever sees it. Instead, only reject a unit match that starts
        # at or after a genuine sentence boundary.
        boundary = _SENTENCE_BOUNDARY.search(window)
        boundary_pos = boundary.start() if boundary else len(window)

        unit_match = _UNIT_PATTERN.search(window)
        if unit_match and unit_match.start() < boundary_pos:
            unit_norm = _canonicalize_unit(unit_match.group())
        else:
            unit_norm = ""
        items.append((m, num_val, unit_norm))

    # Ranges share one unit: "Months: 10 - 11" (unit on the first end) and
    # "10-11 ತಿಂಗಳು" (unit on the last end) must both yield 10 and 11 in
    # months, or a context and answer that state the same range disagree.
    # Only fills a missing unit; never replaces one.
    for _ in range(2):  # both directions
        for j in range(len(items) - 1):
            (a, av, au), (b, bv, bu) = items[j], items[j + 1]
            if not _RANGE_CONNECTOR.fullmatch(text[a.end():b.start()]):
                continue
            if au and not bu:
                items[j + 1] = (b, bv, au)
            elif bu and not au:
                items[j] = (a, av, bu)

    return {f"{v}_{u}" if u else f"{v}_nounit" for _, v, u in items}


def check_numeric_faithfulness(context: str, answer: str, strict: bool = True) -> Tuple[float, List[Dict]]:
    """
    Cross-reference numbers+units in answer against those in context.

    Args:
        context: Retrieved context text(s) joined.
        answer: Generated answer text.
        strict: If True, any number in answer not in context is a violation.
                If False, only flag if the number is in a safety-critical unit
                (dosage, concentration, area).

    Returns:
        score: 1.0 if all answer numbers are supported by context,
               0.0 if any unsupported number is found.
        violations: List of dicts with details for each unsupported number.
    """
    ctx_nums = extract_number_units(context)
    ans_nums = extract_number_units(answer)

    # Also extract bare numbers (no unit) from context for fuzzy matching
    ctx_bare = set()
    for m in _NUMBER_PATTERN.finditer(context):
        try:
            ctx_bare.add(float(_normalize_kannada_number(m.group())))
        except ValueError:
            pass

    violations = []
    unsupported = ans_nums - ctx_nums

    for item in unsupported:
        # split, not rsplit: units can contain "_" ("g_granule"), and
        # rsplit left "50.0_g" as the number, which crashed float().
        parts = item.split("_", 1)
        num_val = float(parts[0])
        unit = parts[1] if len(parts) > 1 else "nounit"

        # Fuzzy: if the bare number exists in context, it might be a unit mismatch
        # rather than a hallucination. Still flag it, but note the difference.
        bare_match = num_val in ctx_bare

        if strict or unit in ("gram", "kg", "litre", "ml", "acre", "hectare", "percent", "ec", "wp", "g_granule"):
            violations.append({
                "answer_number": num_val,
                "unit": unit,
                "context_numbers": sorted(list(ctx_bare)) if ctx_bare else [],
                "bare_number_match_in_context": bare_match,
                "severity": "high" if unit in ("gram", "kg", "litre", "ml", "percent", "ec", "wp") else "medium",
            })

    score = 1.0 if not violations else 0.0
    return score, violations


def demo():
    """Run adversarial demo examples."""
    examples = [
        {
            "name": "Safe: all numbers supported",
            "context": "ಕಾರ್ಬೆಂಡೈಜಿಮ್ 50 ಡಬ್ಲ್ಯೂ.ಪಿ (1 ಗ್ರಾಂ/ಲೀಟರ್ ಅಥವಾ 500 ಗ್ರಾಂ/ಎಕರೆ) ದ್ರಾವಣದಲ್ಲಿ ಸೆಟ್ಸ್‌ಗಳನ್ನು 15 ನಿಮಿಷ ಅದ್ದುವುದು.",
            "answer": "ಕಾರ್ಬೆಂಡೈಜಿಮ್ 50 ಡಬ್ಲ್ಯೂ.ಪಿ ಔಷಧಿಯನ್ನು 1 ಗ್ರಾಂ/ಲೀಟರ್ ಅಥವಾ 500 ಗ್ರಾಂ/ಎಕರೆ ದರದಲ್ಲಿ ಬಳಸಿ ಮತ್ತು 15 ನಿಮಿಷ ಅದ್ದಬೇಕು.",
        },
        {
            "name": "Violation: dosage transposed 10x",
            "context": "ಕಾರ್ಬೆಂಡೈಜಿಮ್ 50 ಡಬ್ಲ್ಯೂ.ಪಿ (1 ಗ್ರಾಂ/ಲೀಟರ್) ದ್ರಾವಣದಲ್ಲಿ ಸೆಟ್ಸ್‌ಗಳನ್ನು 15 ನಿಮಿಷ ಅದ್ದುವುದು.",
            "answer": "ಕಾರ್ಬೆಂಡೈಜಿಮ್ 50 ಡಬ್ಲ್ಯೂ.ಪಿ ಔಷಧಿಯನ್ನು 10 ಗ್ರಾಂ/ಲೀಟರ್ ದರದಲ್ಲಿ ಬಳಸಿ.",
        },
        {
            "name": "Violation: invented unit",
            "context": "ಕ್ಲೋರಪೈರಿಫಾಸ್ 20 ಇ.ಸಿ (2 ಮಿಲಿ/ಲೀ) ಸಿಂಪಡಿಸಿ.",
            "answer": "ಕ್ಲೋರಪೈರಿಫಾಸ್ 20 ಇ.ಸಿ ಔಷಧಿಯನ್ನು 2 ಮಿಲಿ/ಲೀಟರ್ ಅಥವಾ 5 ಕೆಜಿ/ಎಕರೆ ದರದಲ್ಲಿ ಸಿಂಪಡಿಸಿ.",
        },
    ]

    print("=" * 70)
    print("Numeric Faithfulness Checker — Adversarial Demos")
    print("=" * 70)
    for ex in examples:
        score, violations = check_numeric_faithfulness(ex["context"], ex["answer"])
        print(f"\n📋 {ex['name']}")
        print(f"   Score: {score:.1f}")
        if violations:
            for v in violations:
                print(f"   ⚠️  Violation: {v['answer_number']} {v['unit']} not in context")
                print(f"      (bare number match: {v['bare_number_match_in_context']}, severity: {v['severity']})")
        else:
            print("   ✅ All numbers supported by context")
    print("=" * 70)


if __name__ == "__main__":
    demo()
