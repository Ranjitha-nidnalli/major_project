"""K-014: the cause of ROUGE-L's Kannada zeros, pinned (see eval/ROUGE_KANNADA.md)."""
import pytest

rouge_scorer = pytest.importorskip("rouge_score.rouge_scorer")
tokenize = pytest.importorskip("rouge_score.tokenize")


def _self_score(text):
    return rouge_scorer.RougeScorer(["rougeL"]).score(text, text)["rougeL"].fmeasure


def test_default_tokenizer_drops_every_kannada_character():
    assert tokenize.tokenize("ಕಬ್ಬಿಗೆ ಎಕರೆಗೆ ಇಳುವರಿ", None) == []


def test_reference_without_ascii_scores_zero_against_itself():
    assert _self_score("ಕಬ್ಬಿಗೆ ಎಕರೆಗೆ ಇಳುವರಿ") == 0.0


def test_reference_with_a_number_is_scored_on_the_number_only():
    assert tokenize.tokenize("ಕಬ್ಬಿಗೆ ಎಕರೆಗೆ 60 ಟನ್ ಇಳುವರಿ", None) == ["60"]
    a, b = "ಕಬ್ಬಿಗೆ 60 ಟನ್", "ಭತ್ತಕ್ಕೆ 60 ಕೆ.ಜಿ"
    assert rouge_scorer.RougeScorer(["rougeL"]).score(a, b)["rougeL"].fmeasure == 1.0
