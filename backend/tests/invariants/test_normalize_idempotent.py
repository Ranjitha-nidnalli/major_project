"""
Invariant test for indic_preprocess.normalize_kannada (CLAUDE.md rule: "One
Kannada normalisation function for corpus, queries, and eval references").

normalize_kannada is applied at multiple points (corpus chunking in
vector_db.py, query embedding in vector_db.embed_query). If it were not
idempotent, applying it a different number of times to two pieces of text
that should be equivalent could silently produce different embeddings.
This test only checks idempotency: normalize(normalize(x)) == normalize(x).
It does not validate normalization *correctness* (that would require a
Kannada speaker, per CLAUDE.md's rule on human-only tasks).
"""
from indic_preprocess import normalize_kannada


SAMPLES = [
    "",
    "ಕಬ್ಬಿನ  ಸೆಟ್ಸ್‍ಗಳಲ್ಲಿ   ಅನಾನಸ್ ರೋಗ",
    "ಕಬ್ಬಿಗೆ ಸಾರಜನಕವನ್ನು ಎಷ್ಟು ಕಂತುಗಳಲ್ಲಿ ಹಾಕಬೇಕು?",
    "ಕಾರ್ಬೆಂಡೈಜಿಮ್‍ 50 ಡಬ್ಲ್ಯೂ.ಪಿ",
    "plain ascii text",
    "mixed ಕನ್ನಡ and English 123",
    "   leading and trailing spaces   ",
    "multiple\n\nnewlines\tand\ttabs",
    "ಕಬ್ಬು। ಕಬ್ಬಿನ।ಕಬ್ಬಿಗೆ",
]


def test_normalize_is_idempotent():
    for sample in SAMPLES:
        once = normalize_kannada(sample)
        twice = normalize_kannada(once)
        assert twice == once, (
            f"normalize_kannada is not idempotent for {sample!r}: "
            f"normalize(x)={once!r}, normalize(normalize(x))={twice!r}"
        )


def test_normalize_empty_string_is_falsy_safe():
    assert normalize_kannada("") == ""
    assert normalize_kannada(None) is None
