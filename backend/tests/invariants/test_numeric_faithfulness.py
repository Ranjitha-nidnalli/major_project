"""
Regression tests for backend/eval/numeric_faithfulness.py (known defect #3,
CLAUDE.md / ARCHITECTURE.md Section 22).

Verified defect: the old extraction regex bound a number to any unit up to
3 words away, regardless of whether a *different* number sat in between.
This both mis-bound units (a number could pick up a unit belonging to a
later quantity) and dropped number+unit pairs entirely (an earlier match
could consume past them). A faithful paraphrase of dosage text scored 0.0
-- the exact "rejects correct answers" failure CLAUDE.md calls out. These
tests pin the corrected nearest-unit-before-next-number behavior.
"""
from eval.numeric_faithfulness import check_numeric_faithfulness, extract_number_units


def test_adjacent_number_unit_pairs_do_not_cross_bind():
    """
    Two number+unit pairs in the same clause, separated by "or": each number
    must bind to its OWN nearby unit, not the other pair's, and neither may
    be dropped.
    """
    text = "1 ಗ್ರಾಂ/ಲೀಟರ್ ಅಥವಾ 500 ಗ್ರಾಂ/ಎಕರೆ"
    found = extract_number_units(text)
    assert "1.0_gram" in found, f"1 gram pair was dropped or mis-bound: {found}"
    assert "500.0_gram" in found, f"500 gram pair was dropped or mis-bound: {found}"


def test_product_concentration_number_binds_to_its_own_unit():
    """
    "Carbendazim 50 WP (1 gram/litre)" -- the 50 must bind to WP (the
    product concentration label), not to a unit borrowed from the dosage
    that follows in parentheses.
    """
    text = "ಕಾರ್ಬೆಂಡೈಜಿಮ್ 50 ಡಬ್ಲ್ಯೂ.ಪಿ (1 ಗ್ರಾಂ/ಲೀಟರ್)"
    found = extract_number_units(text)
    assert "50.0_wp" in found, f"50 did not bind to WP: {found}"


def test_faithful_paraphrase_scores_full_confidence():
    context = (
        "ಕಾರ್ಬೆಂಡೈಜಿಮ್ 50 ಡಬ್ಲ್ಯೂ.ಪಿ (1 ಗ್ರಾಂ/ಲೀಟರ್ ಅಥವಾ 500 ಗ್ರಾಂ/ಎಕರೆ) "
        "ದ್ರಾವಣದಲ್ಲಿ ಸೆಟ್ಸ್‌ಗಳನ್ನು 15 ನಿಮಿಷ ಅದ್ದುವುದು."
    )
    answer = (
        "ಕಾರ್ಬೆಂಡೈಜಿಮ್ 50 ಡಬ್ಲ್ಯೂ.ಪಿ ಔಷಧಿಯನ್ನು 1 ಗ್ರಾಂ/ಲೀಟರ್ ಅಥವಾ 500 ಗ್ರಾಂ/ಎಕರೆ "
        "ದರದಲ್ಲಿ ಬಳಸಿ ಮತ್ತು 15 ನಿಮಿಷ ಅದ್ದಬೇಕು."
    )
    score, violations = check_numeric_faithfulness(context, answer)
    assert score == 1.0, f"faithful paraphrase was rejected: {violations}"


def test_transposed_dosage_is_flagged():
    context = "ಕಾರ್ಬೆಂಡೈಜಿಮ್ 50 ಡಬ್ಲ್ಯೂ.ಪಿ (1 ಗ್ರಾಂ/ಲೀಟರ್) ದ್ರಾವಣದಲ್ಲಿ ಸೆಟ್ಸ್‌ಗಳನ್ನು 15 ನಿಮಿಷ ಅದ್ದುವುದು."
    answer = "ಕಾರ್ಬೆಂಡೈಜಿಮ್ 50 ಡಬ್ಲ್ಯೂ.ಪಿ ಔಷಧಿಯನ್ನು 10 ಗ್ರಾಂ/ಲೀಟರ್ ದರದಲ್ಲಿ ಬಳಸಿ."
    score, violations = check_numeric_faithfulness(context, answer)
    assert score == 0.0
    assert any(v["answer_number"] == 10.0 and v["unit"] == "gram" for v in violations)
