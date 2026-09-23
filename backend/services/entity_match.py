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

STATUS 2026-09-23: PROTOTYPED AND TESTED, NOT WIRED INTO PRODUCTION.
NOT SAFE TO DEPLOY AS-IS. Validated against all 18 real questions'
actual retrieved contexts (eval/contexts.json as of commit b01169d).
Every rank-cutoff variant tried (check all of top-5, top-1 only, top-2,
top-3, top-4) produces real false positives -- genuinely answerable
questions where a retrieved-but-irrelevant entity chunk (root borer,
white woolly aphid, pineapple disease, etc.) got pulled into the top-k
by embedding similarity alongside the actual correct chunk, and this
hook would incorrectly refuse. At the best configuration tested (check
only the single top-ranked chunk), 4 of 15 answerable questions
(disease-3, pest-2, fertilizer-3, general-2) still false-positive --
notably because in all 4, the retrieval's rank-1 hit was not even the
gold chunk (a separate, deeper retrieval-quality question this file
does not attempt to fix), yet generation still succeeded because the
correct chunk was present lower in the top-5. Checking all of top-5
produces even more false positives (5+ of 15).

This module DOES correctly flag the motivating case (pest-5, a pest not
in the corpus, where the top hit was a real-but-wrong pest) -- but a
heuristic that trades a ~27-47% false-refusal rate on this small,
18-question set for catching one hallucination is not a net safety
improvement; false refusals erode trust and utility too. Do not wire
this into services/gating.py's entity_match_hook without either (a) a
genuinely better entity-extraction approach (real NER, or an LLM call
that explicitly cross-checks the query's subject against retrieved
card names), or (b) first improving retrieval quality for named-entity
queries specifically (why rank-1 misses the gold chunk on exact lexical
matches like disease-3's "ಕಾಡಿಗೆ ರೋಗ" appearing verbatim in the Smut
chunk's declared name is itself worth investigating -- possibly a case
where BM25 would outperform dense embedding, worth checking against
TODO #10's bucketed BM25 results once more gold data exists).

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


def entity_match(query_text, retrieved_chunk_texts):
    """
    services/gating.py's entity_match_hook signature:
    (query_entities, retrieved_entities) -> bool, True = pass.

    True if at least one retrieved chunk has no declared entity (nothing
    to check) or matches the query; False only when every retrieved
    chunk that DOES declare an entity fails to match it.
    """
    query_tokens = _tokens(query_text)
    saw_any_entity = False

    for chunk_text in retrieved_chunk_texts:
        entity_tokens = extract_chunk_entity_tokens(chunk_text)
        if not entity_tokens:
            continue
        saw_any_entity = True
        if query_tokens & entity_tokens:
            return True

    if not saw_any_entity:
        return True

    return False
