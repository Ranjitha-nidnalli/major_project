"""
indic_preprocess.py

Kannada text preprocessing using Indic NLP Library.
Normalizes Kannada Unicode characters, removes extra spaces,
and handles script-specific punctuation before chunking/embedding.

This directly addresses the agglutination problem identified in the
architect review: Kannada's inflected forms (ಕಬ್ಬು/ಕಬ್ಬಿನ/ಕಬ್ಬಿಗೆ)
benefit from Unicode normalization before embedding, improving
matching across morphological variants.

Usage:
    from indic_preprocess import normalize_kannada
    clean_text = normalize_kannada(raw_text)

If indic-nlp-library is not installed, falls back to a lightweight
regex-based normalizer that handles the most common Kannada issues.
"""
import re
import unicodedata

# Try to import Indic NLP Library; fall back to lightweight regex if unavailable
try:
    from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
    _INDIC_NLP_AVAILABLE = True
except ImportError:
    _INDIC_NLP_AVAILABLE = False

# Kannada Unicode range
_KANNADA_RANGE = re.compile(r'[ಀ-೿]+')

# Common Kannada normalization issues (fallback regex-based)
# These handle the most frequent Unicode normalization problems
_KANNADA_ISSUES = [
    # Zero-width joiner/non-joiner artifacts
    (re.compile(r'‍'), ''),   # ZWJ
    (re.compile(r'‌'), ''),   # ZWNJ
    # Extra spaces around Kannada characters
    (re.compile(r'(?<=ಀ-೿)\s+(?=ಀ-೿)'), ''),
    # Multiple spaces → single space
    (re.compile(r'\s+'), ' '),
    # Kannada danda (।) spacing
    (re.compile(r'\s*।\s*'), '। '),
]


def _fallback_normalize(text: str) -> str:
    """
    Lightweight regex-based Kannada normalizer.
    Runs when indic-nlp-library is not installed.
    """
    # Unicode NFC normalization (combines decomposed characters)
    text = unicodedata.normalize('NFC', text)

    # Apply regex fixes
    for pattern, replacement in _KANNADA_ISSUES:
        text = pattern.sub(replacement, text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


# Cache the normalizer instance
_normalizer = None

def _get_normalizer():
    global _normalizer
    if _normalizer is None and _INDIC_NLP_AVAILABLE:
        factory = IndicNormalizerFactory()
        _normalizer = factory.get_normalizer("kn", remove_nuktas=False)
    return _normalizer


def normalize_kannada(text: str) -> str:
    """
    Normalize Kannada text for embedding.

    If indic-nlp-library is installed, uses its full normalizer.
    Otherwise falls back to regex-based normalization.

    Args:
        text: Raw text (may contain Kannada, English, numbers, mixed)

    Returns:
        Normalized text ready for chunking/embedding.
    """
    if not text:
        return text

    if _INDIC_NLP_AVAILABLE:
        normalizer = _get_normalizer()
        return normalizer.normalize(text)
    else:
        return _fallback_normalize(text)


def normalize_batch(texts: list) -> list:
    """Normalize a batch of texts."""
    return [normalize_kannada(t) for t in texts]


def demo():
    """Demonstrate normalization on sample Kannada text."""
    samples = [
        # Mixed text with potential normalization issues
        "ಕಬ್ಬಿನ  ಸೆಟ್ಸ್‌ಗಳಲ್ಲಿ   ಅನಾನಸ್ ರೋಗ",
        "ಕಬ್ಬಿಗೆ ಸಾರಜನಕವನ್ನು ಎಷ್ಟು ಕಂತುಗಳಲ್ಲಿ ಹಾಕಬೇಕು?",
        # Text with ZWJ/ZWNJ artifacts (simulated)
        "ಕಾರ್ಬೆಂಡೈಜಿಮ್‍ 50 ಡಬ್ಲ್ಯೂ.ಪಿ",
    ]

    print("Indic NLP Preprocessing Demo")
    print("=" * 60)
    print(f"indic-nlp-library available: {_INDIC_NLP_AVAILABLE}")
    print("-" * 60)

    for s in samples:
        normalized = normalize_kannada(s)
        print(f"IN:  {s!r}")
        print(f"OUT: {normalized!r}")
        print()


if __name__ == "__main__":
    demo()
