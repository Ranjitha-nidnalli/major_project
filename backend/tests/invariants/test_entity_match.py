"""
Unit tests for services/entity_match.py (abstention gate Layer 3,
ARCHITECTURE.md Section 21). Wired into production for pest/disease since
2026-09-23; the top-chunk rule and card dropping were added 2026-10-01.
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


# --- resolve_entity_category (TODO #47) ---
from services.entity_match import resolve_entity_category

PROTECTED = frozenset({"pest", "disease"})


def test_router_misroute_does_not_disable_protection():
    """The live failure: router says 'general' for a pest question whose top
    retrieved chunk is a pest card. Protection must still apply."""
    assert resolve_entity_category("general", "pest", PROTECTED) == "pest"
    assert resolve_entity_category("fertilizer", "disease", PROTECTED) == "disease"


def test_router_can_add_protection_without_chunk_signal():
    assert resolve_entity_category("pest", "general", PROTECTED) == "pest"
    assert resolve_entity_category("disease", None, PROTECTED) == "disease"


def test_router_label_kept_when_neither_signal_fires():
    assert resolve_entity_category("fertilizer", "general", PROTECTED) == "fertilizer"
    assert resolve_entity_category("general", None, PROTECTED) == "general"


def test_router_label_wins_when_both_are_protected():
    assert resolve_entity_category("pest", "disease", PROTECTED) == "pest"


# --- top-chunk rule and card dropping (2026-10-01, run 2026-09-29) ---
from services.entity_match import drop_mismatched_entity_cards, resolve_final_category

TERMITES = "Topic: Pest Management Name: Termites (ಗೆದ್ದಲು) Recommendations: Chemical: Chlorantraniliprole 0.4 G"
PINEAPPLE = "Topic: Disease Management Name: Pineapple Disease / Sett Rot (ಅನಾನಸ್ ರೋಗ) Recommendations: Chemical: Carbendazim 50 WP"
SMUT = "Topic: Disease Management Name: Smut (ಕಾಡಿಗೆ ರೋಗ) Recommendations: Chemical: Cultural Control"
SEED_RATE = "Topic: Planting Practices Seed Rate: Standard: 25,000-30,000 setts (5-7 t) per hectare"
BLACK_BEETLE_Q = "ಕಬ್ಬಿನಲ್ಲಿ ಕಪ್ಪು ದುಂಬಿ ಕೀಟ ಕಂಡುಬಂದರೆ ಯಾವ ಔಷಧಿ ಸಿಂಪಡಿಸಬೇಕು?"
SETTS_Q = "ಒಂದು ಹೆಕ್ಟೇರಿಗೆ ಎಷ್ಟು ಸೆಟ್ಸ್ ಬೇಕಾಗುತ್ತದೆ?"


def test_refuses_when_top_chunk_is_a_different_pests_card():
    """pest-5: the answer would come from the Termites card."""
    assert entity_match(BLACK_BEETLE_Q, [TERMITES, SEED_RATE]) is False


def test_passes_question_naming_no_pest_when_top_chunk_has_no_entity():
    """general-2: the answer chunk ranks first; an unrelated disease card
    further down must not refuse the whole question."""
    assert entity_match(SETTS_Q, [SEED_RATE, PINEAPPLE]) is True


def test_passes_when_matching_card_is_not_first():
    """disease-3: the top card is another disease, but the Smut card the
    question names is in the context."""
    assert entity_match("ಕಾಡಿಗೆ ರೋಗ ಬಂದ ಕಬ್ಬಿನ ಗಿಡಗಳನ್ನು ಏನು ಮಾಡಬೇಕು?", [PINEAPPLE, SMUT]) is True


def test_empty_context_passes_to_relevance_layer():
    assert entity_match(SETTS_Q, []) is True


def test_drops_unnamed_entity_cards_when_none_match():
    assert drop_mismatched_entity_cards(SETTS_Q, [SEED_RATE, PINEAPPLE, TERMITES]) == [SEED_RATE]


def test_keeps_context_unchanged_when_a_card_matches():
    chunks = [PINEAPPLE, SMUT, SEED_RATE]
    assert drop_mismatched_entity_cards("ಕಾಡಿಗೆ ರೋಗ ಬಂದರೆ ಏನು ಮಾಡಬೇಕು?", chunks) == chunks


def test_final_category_falls_back_when_no_protected_card_left():
    """fertilizer-3: upgraded to pest by the default search, but the
    re-retrieved context has no pest/disease card."""
    assert resolve_final_category("fertilizer", "pest", ["general", "fertilizer", "weed"], PROTECTED) == "fertilizer"


def test_final_category_keeps_upgrade_when_protected_card_remains():
    assert resolve_final_category("general", "pest", ["general", "pest"], PROTECTED) == "pest"


def test_final_category_never_removes_router_protection():
    assert resolve_final_category("disease", "disease", ["general", "general"], PROTECTED) == "disease"
