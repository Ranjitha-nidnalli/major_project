"""
Unit tests for services/entity_match.py's core mechanics.

Per that module's docstring, the hook is NOT wired into production --
tested against all 18 real questions' actual retrieved contexts and found
to false-positive on 27-47% of genuinely answerable questions depending
on configuration. These tests cover only the underlying extraction/match
primitives, which are independently correct regardless of that larger
deployment-viability finding.
"""
from services.entity_match import extract_chunk_entity_tokens, entity_match


def test_extract_entity_tokens_simple_name():
    text = "Crop: Sugarcane | Topic: Pest Management Name: Termites (ಗೆದ್ದಲು) Recommendations: Chemical: X"
    tokens = extract_chunk_entity_tokens(text)
    assert "termites" in tokens
    assert "ಗೆದ್ದಲು" in tokens


def test_extract_entity_tokens_handles_double_parenthetical():
    """Real corpus case: 'Name: Yellow Leaf Disease (YLD) (ಹಳದಿ ನಂಜು ರೋಗ) Recommendations:'
    -- the whole span between Name: and Recommendations: is tokenized, so
    both parenthetical groups contribute, not just the first."""
    text = "Name: Yellow Leaf Disease (YLD) (ಹಳದಿ ನಂಜು ರೋಗ) Recommendations: Chemical: X"
    tokens = extract_chunk_entity_tokens(text)
    assert "ಹಳದಿ" in tokens
    assert "ನಂಜು" in tokens
    assert "yellow" in tokens


def test_extract_entity_tokens_no_name_field_returns_empty():
    text = "Crop: Sugarcane | Topic: Irrigation Schedule Stage: General Notes: Irrigate every 8 days."
    assert extract_chunk_entity_tokens(text) == set()


def test_extract_entity_tokens_filters_generic_stopwords():
    text = "Name: Pest (ಕೀಟ) Recommendations: X"
    tokens = extract_chunk_entity_tokens(text)
    assert "pest" not in tokens
    assert "ಕೀಟ" not in tokens


def test_entity_match_true_when_no_chunk_declares_entity():
    query = "ಕಬ್ಬು ಬೆಳೆಯಲು ಯಾವ ಮಣ್ಣು ಸೂಕ್ತ?"
    chunks = ["Topic: Climate And Soil Soil Type: Deep red soil, Black soil."]
    assert entity_match(query, chunks) is True


def test_entity_match_true_when_query_names_the_retrieved_entity():
    query = "ಕಬ್ಬಿನ ಗದ್ದೆಯಲ್ಲಿ ಗೆದ್ದಲು ನಿಯಂತ್ರಣ ಹೇಗೆ?"
    chunks = ["Name: Termites (ಗೆದ್ದಲು) Recommendations: Chemical: X"]
    assert entity_match(query, chunks) is True


def test_entity_match_false_when_query_names_a_different_entity_than_retrieved():
    """The motivating case: query about a pest not in the corpus, retrieval
    returns a real but different pest's chunk."""
    query = "ಕಬ್ಬಿನಲ್ಲಿ ಕಪ್ಪು ದುಂಬಿ ಕೀಟ ಕಂಡುಬಂದರೆ ಯಾವ ಔಷಧಿ ಸಿಂಪಡಿಸಬೇಕು?"
    chunks = ["Name: Termites (ಗೆದ್ದಲು) Recommendations: Chemical: Chlorantraniliprole 0.4 G"]
    assert entity_match(query, chunks) is False
