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
    questions to remove them.
    """
    if router_category in protected_categories:
        return router_category
    if top_chunk_category in protected_categories:
        return top_chunk_category
    return router_category
