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
    r"ನಿಮಿಷ", r"minute", r"minutes",
    r"ಗಂಟೆ", r"hour", r"hours",
    r"ದಿನ", r"day", r"days",
    r"ವಾರ", r"week", r"weeks",
    r"ತಿಂಗಳು", r"month", r"months",
    # Count
    r"ಸೆಟ್ಸ್", r"setts?",
    r"ಸಸಿಗಳು", r"seedlings?",
]

_UNIT_PATTERN = re.compile(
    r"(?:" + "|".join(_UNITS) + r")",
    re.IGNORECASE,
)

# Combined: number followed (optionally) by unit, with up to 3 words of slack
_NUM_UNIT_RE = re.compile(
    rf"({_NUMBER_PATTERN.pattern})"  # group 1: the number
    rf"(?:\s+{{0,3}}"                # up to 3 whitespace-separated words
    rf"({_UNIT_PATTERN.pattern}))?"   # group 2: the unit (optional)
    rf"(?=\s|$|[^\wಀ-೿])",  # lookahead: word boundary
    re.IGNORECASE,
)


def _normalize_kannada_number(s: str) -> str:
    """Convert Kannada digits to Arabic numerals for comparison."""
    kannada_to_arabic = str.maketrans("೦೧೨೩೪೫೬೭೮೯", "0123456789")
    return s.translate(kannada_to_arabic)


def extract_number_units(text: str) -> Set[str]:
    """
    Extract all number+unit tuples from text.
    Returns a set of normalized strings like '10.0_gram' for easy comparison.
    """
    found = set()
    for match in _NUM_UNIT_RE.finditer(text):
        num_str = match.group(1)
        unit_str = match.group(2)
        num_norm = _normalize_kannada_number(num_str)
        # Try to parse as float for normalization
        try:
            num_val = float(num_norm)
        except ValueError:
            continue
        # Normalize unit
        unit_norm = ""
        if unit_str:
            u = unit_str.strip().lower()
            # Canonicalize common variants
            if u in ("ಗ್ರಾಂ", "ಗ್ರಾಮ", "gram", "grams", "g"):
                unit_norm = "gram"
            elif u in ("ಕೆಜಿ", "kg"):
                unit_norm = "kg"
            elif u in ("ಲೀಟರ್", "ಲೀ", "litre", "liter", "litres", "liters", "l"):
                unit_norm = "litre"
            elif u in ("ಮಿಲಿ", "ml"):
                unit_norm = "ml"
            elif u in ("ಎಕರೆ", "acre", "acres"):
                unit_norm = "acre"
            elif u in ("ಹೆಕ್ಟೇರ್", "hectare", "hectares"):
                unit_norm = "hectare"
            elif u in ("%", "ಪರ್ಸೆಂಟ್", "percent"):
                unit_norm = "percent"
            elif u in ("ನಿಮಿಷ", "minute", "minutes"):
                unit_norm = "minute"
            elif u in ("ಗಂಟೆ", "hour", "hours"):
                unit_norm = "hour"
            elif u in ("ದಿನ", "day", "days"):
                unit_norm = "day"
            elif u in ("ವಾರ", "week", "weeks"):
                unit_norm = "week"
            elif u in ("ತಿಂಗಳು", "month", "months"):
                unit_norm = "month"
            elif u in ("ಸೆಟ್ಸ್", "setts", "sett"):
                unit_norm = "sett"
            elif u in ("ಟನ್", "ton", "tons", "tonne", "tonnes"):
                unit_norm = "ton"
            elif u in ("ಕ್ವಿಂಟಾಲ್", "quintal", "quintals"):
                unit_norm = "quintal"
            elif u in ("ಡಬ್ಲ್ಯೂ\.ಪಿ", "wp"):
                unit_norm = "wp"
            elif u in ("ಇ\.ಸಿ", "ec"):
                unit_norm = "ec"
            elif u in ("ಜಿ", "g_granule"):
                unit_norm = "g_granule"
            else:
                unit_norm = u
        # Store as "value_unit" for set comparison
        key = f"{num_val}_{unit_norm}" if unit_norm else f"{num_val}_nounit"
        found.add(key)
    return found


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
        parts = item.rsplit("_", 1)
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
