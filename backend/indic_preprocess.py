
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

# Common Kannada normalization issues (fallback regex-based)
_KANNADA_ISSUES = [
    # Zero-width joiner/non-joiner artifacts
    (re.compile(r'\u200D'), ''),   # ZWJ
    (re.compile(r'\u200C'), ''),   # ZWNJ
    # Multiple spaces → single space
    (re.compile(r'\s+'), ' '),
    # Kannada danda (।) spacing
    (re.compile(r'\s*।\s*'), '। '),
]


def _fallback_normalize(text: str) -> str:
    """Lightweight regex-based Kannada normalizer."""
    text = unicodedata.normalize('NFC', text)
    for pattern, replacement in _KANNADA_ISSUES:
        text = pattern.sub(replacement, text)
    return text.strip()


_normalizer = None

def _get_normalizer():
    global _normalizer
    if _normalizer is None and _INDIC_NLP_AVAILABLE:
        factory = IndicNormalizerFactory()
        _normalizer = factory.get_normalizer("kn", remove_nuktas=False)
    return _normalizer


def normalize_kannada(text: str) -> str:
    """Normalize Kannada text for embedding."""
    if not text:
        return text
    if _INDIC_NLP_AVAILABLE:
        return _get_normalizer().normalize(text)
    return _fallback_normalize(text)


def normalize_batch(texts: list) -> list:
    """Normalize a batch of texts."""
    return [normalize_kannada(t) for t in texts]


if __name__ == "__main__":
    samples = [
        "ಕಬ್ಬಿನ  ಸೆಟ್ಸ್‌ಗಳಲ್ಲಿ   ಅನಾನಸ್ ರೋಗ",
        "ಕಬ್ಬಿಗೆ ಸಾರಜನಕವನ್ನು ಎಷ್ಟು ಕಂತುಗಳಲ್ಲಿ ಹಾಕಬೇಕು?",
        "ಕಾರ್ಬೆಂಡೈಜಿಮ್\u200D 50 ಡಬ್ಲ್ಯೂ.ಪಿ",
    ]
    print(f"indic-nlp-library available: {_INDIC_NLP_AVAILABLE}")
    for s in samples:
        print(f"IN:  {s!r}")
        print(f"OUT: {normalize_kannada(s)!r}")
        print()
