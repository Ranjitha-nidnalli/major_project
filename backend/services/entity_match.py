"""
services/entity_match.py

Entity-match check for the abstention gate's Layer 3 (ARCHITECTURE.md
Section 21). Refuses when the retrieved cards declare specific named
entities (pest/disease names) but none of them are actually mentioned in
the query, regardless of relevance score.

Motivated by a reproduced, live failure (2026-09-22): a query about a
pest not in the corpus ("black beetle") retrieved a different,
topically-similar pest's chunk (Termites, dense relevance 0.597 -- above
both live gate thresholds), and the LLM confidently presented that
different pest's dosage as the answer instead of refusing. Relevance
alone cannot catch this -- the retrieved chunk IS topically relevant
(pest management), it just isn't about the pest that was asked about.

Deliberately NOT LLM-based -- no new paid API call, pure lexical token
overlap between the query and each retrieved chunk's own declared entity
name (the "Name: X (Y) Recommendations: ..." pattern
vector_db._entity_to_text writes at chunking time for disease_management
and pest_management sections).

STATUS 2026-09-23: WIRED INTO PRODUCTION, SCOPED TO pest/disease ONLY.

First validation attempt (against pre-BM25-fusion retrieval, all of
top-5) found real false positives on 6 of 18 questions across every
rank-cutoff variant tried, including fertilizer-1/2/3 and general-2/3 --
retrieved-but-irrelevant entity chunks (root borer, white woolly aphid,
pineapple disease) got pulled into the top-k by embedding similarity
alongside the actual correct chunk. Not shippable at that point.

Re-tested after wiring BM25+dense fusion into production for pest/
disease queries (rag_service.BM25_FUSION_CATEGORIES, TODO #10): with
that better retrieval, this hook is **9/9 correct for pest+disease
specifically** -- all 8 answerable pest/disease questions pass, and
pest-5 (a pest not in the corpus) correctly triggers entity_mismatch,
confirmed end-to-end against the real GatingConfig/decide() call, not
just this module in isolation. The 6 original false positives were all
in fertilizer/general -- categories that do NOT get BM25 fusion and
were NOT re-tested; do not widen this hook's scope to them without
separately validating, since the same retrieval-quality problem that
caused those false positives is presumably still present there.

rag_service._entity_match_hook enforces this scoping: it returns True
(no-op) for any category outside BM25_FUSION_CATEGORIES, and only calls
entity_match() for pest/disease.

**Related finding, not fixed by this module**: end-to-end testing
surfaced that pest-2 (a genuinely answerable question) gets refused by
the *relevance* gate (Layer 1/2), not this entity check -- its plain
dense-only relevance score (0.497) sits just under the uncalibrated
0.50 safety-critical threshold. entity_match never even runs for that
case, since relevance is checked first and fails. This is exactly the
"do not treat these thresholds as calibrated" risk TODO.md already
flags, now with a concrete instance: a real, correct answer sitting
right at an arbitrary boundary. Not addressed here -- needs the
Phase 2 threshold recalibration with more gold data, not a quick fix.

This is a first-pass heuristic, not a general NER system: it only has
signal for chunks that declare a Name: field (most disease/pest chunks;
general/fertilizer/soil chunks don't), and its stopword list is
hand-curated from this corpus's actual vocabulary.
"""
import re

_TOKEN_RE = re.compile(r"[ಀ-೿a-zA-Z]+")

_NAME_FIELD_RE = re.compile(r"Name:\s*(.+?)\s*Recommendations:")

# Generic category/verb/question words that appear in nearly every
# pest/disease query and chunk -- filtered out so only the actual
# distinguishing entity name drives the match, not shared boilerplate.
_GENERIC_STOPWORDS = {
    # Kannada: category nouns, verbs, question words, crop name
    "ಕೀಟ", "ಕೀಟಕ್ಕೆ", "ರೋಗ", "ರೋಗಕ್ಕೆ", "ನಿಯಂತ್ರಣ", "ನಿಯಂತ್ರಣಕ್ಕೆ",
    "ಔಷಧಿ", "ಔಷಧ", "ಸಿಂಪಡಿಸಬೇಕು", "ಸಿಂಪಡಿಸಿ", "ಕಂಡುಬಂದರೆ", "ಯಾವ",
    "ಏನು", "ಹೇಗೆ", "ಬಳಸಬೇಕು", "ಮಾಡಬೇಕು", "ಬಂದ", "ಬಂದಾಗ", "ಬಾರದಂತೆ",
    "ತಡೆಯುವುದು", "ಕಬ್ಬು", "ಕಬ್ಬಿನ", "ಕಬ್ಬಿಗೆ", "ಕಬ್ಬಿನಲ್ಲಿ", "ಗಿಡಗಳನ್ನು",
    "ಬಂದಾಗ", "ಗಳಲ್ಲಿ",
    # English
    "pest", "disease", "control", "medicine", "spray", "which", "what",
    "how", "should", "use", "sugarcane", "name", "recommendations",
}


def _tokens(text):
    return {
        t.lower() for t in _TOKEN_RE.findall(text)
        if len(t) > 2 and t.lower() not in _GENERIC_STOPWORDS
    }


def extract_chunk_entity_tokens(chunk_text):
    """
    Distinctive tokens from a chunk's own declared entity name. Empty set
    for chunks with no Name: field (most general/fertilizer/soil chunks)
    -- callers must treat an empty set as "no entity signal", not a
    mismatch.
    """
    match = _NAME_FIELD_RE.search(chunk_text)
    if not match:
        return set()
    return _tokens(match.group(1))


def _any_entity_card_matches(query_tokens, chunk_texts):
    return any(query_tokens & extract_chunk_entity_tokens(t) for t in chunk_texts)


def entity_match(query_text, retrieved_chunk_texts):
    """
    services/gating.py's entity_match_hook signature:
    (query_entities, retrieved_entities) -> bool, True = pass.

    True if any retrieved card's declared entity appears in the query, or
    no card declares an entity. When no declared entity matches, refuse
    only if the TOP-ranked chunk is itself an entity card: the answer would
    then come from a card about a different pest/disease (pest-5, "black
    beetle" -> Termites card). If the top chunk declares no entity, the
    question names no pest/disease (general-2 seed rate, general-3 sett
    treatment, run 2026-09-29) and passes; the caller then removes the
    non-matching entity cards with drop_mismatched_entity_cards, so the
    wrong pest's dose never reaches generation.

    Before 2026-10-01 this refused whenever no entity card matched, even
    when a card-less chunk ranked first and held the answer.
    """
    if not retrieved_chunk_texts:
        return True
    query_tokens = _tokens(query_text)
    if _any_entity_card_matches(query_tokens, retrieved_chunk_texts):
        return True
    return not extract_chunk_entity_tokens(retrieved_chunk_texts[0])


def drop_mismatched_entity_cards(query_text, chunk_texts):
    """
    Chunks to keep for generation. When no entity card matches the query,
    every entity card is dropped: none of them is about what was asked.
    When some card matches, the chunks are returned unchanged (behaviour
    validated 2026-09-23 for the answerable pest/disease questions).
    Order is preserved.
    """
    query_tokens = _tokens(query_text)
    if _any_entity_card_matches(query_tokens, chunk_texts):
        return list(chunk_texts)
    return [t for t in chunk_texts if not extract_chunk_entity_tokens(t)]


def resolve_entity_category(router_category, top_chunk_category, protected_categories):
    """
    Category that decides whether the pest/disease protections run (BM25
    fusion, entity-match, the safety-critical threshold). TODO #47.

    The LLM router alone is not trusted with this: in the 2026-09-25 live
    eval it routed 7 of 8 pest/disease questions to "general", which
    silently switched those protections off -- including for pest-5, the
    reproduced wrong-pest hallucination. The top hybrid hit's own corpus
    category (a chunking-time label, not an LLM output) is a second
    signal. Either one can turn protection ON; neither can turn it off.
    The router may add caution, never remove it.

    Known cost (zero-cost gate harness, 2026-09-25): on the 18-question
    set, the chunk signal also fires for fertilizer-3 and general-2, whose
    top hybrid hit is a pest/disease card, and both are then refused.
    Refusal is the safe direction; do not tune this rule on the same 18
    questions to remove them. (2026-10-01: both now pass, but through two
    consistency fixes, not a change to this rule: resolve_final_category
    for fertilizer-3, and entity_match's top-chunk rule for general-2.)
    """
    if router_category in protected_categories:
        return router_category
    if top_chunk_category in protected_categories:
        return top_chunk_category
    return router_category


def resolve_final_category(router_category, upgraded_category, final_chunk_categories, protected_categories):
    """
    Category for the gate once the pest/disease re-retrieval has run.

    resolve_entity_category upgrades on the DEFAULT search's top hit, but
    the pest/disease path then replaces the context with a BM25+dense
    search. If that final context holds no pest/disease card at all, the
    gate was judging a context the model never sees: fertilizer-3 (run
    2026-09-29) was upgraded to "pest", re-retrieved five non-pest chunks,
    and was still gated as pest on the #1 chunk (Ratooning, 0.456) while
    the answer sat at #3. The protections exist to stop a wrong pest's
    dose reaching generation; with no pest/disease card in the context
    there is none to stop, so fall back to the router's category.

    Not a tuned rule: it only makes the gate's category agree with the
    context it gates. The router can still never remove protection: a
    router pest/disease label is always kept.
    """
    if router_category in protected_categories:
        return router_category
    if upgraded_category in protected_categories and not (set(final_chunk_categories) & protected_categories):
        return router_category
    return upgraded_category
