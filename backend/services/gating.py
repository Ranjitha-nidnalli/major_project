"""
backend/services/gating.py

Pure abstention gate: decides whether to answer or refuse.

Replaces the old inert confidence gate in rag_service.py, which compared a
Qdrant RRF fusion score against HARD_REFUSAL_THRESHOLD. That score is
rank-derived, not relevance-derived: a document ranked first by both the
dense and sparse signals scores 1.0 regardless of whether it is actually
relevant to the query. The project's own refusal_results.jsonl recorded an
unanswerable price query with search_score: 1.0 -- the gate never fired
(ARCHITECTURE.md Section 21, verified defect).

This module takes relevance as an INPUT rather than computing it, so it has
no dependency on Qdrant, embeddings, or an LLM and can be unit tested in
isolation. Callers (rag_service.py) are responsible for computing relevance
as the max dense cosine similarity over the retrieved cards (or a
cross-encoder score, if the reranker is enabled) -- never the RRF fusion
score. See ARCHITECTURE.md Section 21 for the full defence-in-depth design
this gate is layer 2 of.
"""
from dataclasses import dataclass, field
from typing import Callable, FrozenSet, Optional


# ---------------------------------------------------------------------------
# UNCALIBRATED THRESHOLDS -- placeholders, not calibrated values.
#
# ARCHITECTURE.md Section 21 specifies that the real threshold(s) must be
# derived by sweeping candidate values against a labelled
# answerable/unanswerable question set and picking by F1, with a stricter
# threshold for safety-critical categories (pest, disease, fertilizer).
# backend/eval/threshold_sweep.py already implements that sweep; it has not
# been run against a real relevance signal or a real labelled set yet
# (both are Phase 2 work). Do not report these numbers as calibrated, and do
# not use them as evidence of gate quality in any report or eval output.
# ---------------------------------------------------------------------------
UNCALIBRATED_DEFAULT_THRESHOLD = 0.35          # TODO(Phase 2): replace via threshold_sweep.py
UNCALIBRATED_SAFETY_CRITICAL_THRESHOLD = 0.50  # TODO(Phase 2): replace via threshold_sweep.py

SAFETY_CRITICAL_CATEGORIES = frozenset({"pest", "disease", "fertilizer"})


@dataclass(frozen=True)
class GatingConfig:
    """
    Thresholds and extension hooks for the abstention gate.

    default_threshold / safety_critical_threshold are UNCALIBRATED
    placeholders (see module docstring). category_thresholds lets a caller
    override the threshold for a specific category once calibration data
    exists, without editing this module.

    entity_match_hook is a documented extension point for Phase 4
    (ARCHITECTURE.md Section 21: "the entity check is the highest-value
    addition" -- refuse if the query's entity does not appear in any
    retrieved card, regardless of relevance score). It is NOT implemented in
    Phase 1. When None (the Phase 1 default), entity matching is skipped
    entirely -- it is never silently treated as a pass. Signature, if
    provided: (query_entities, retrieved_entities) -> bool.
    """
    default_threshold: float = UNCALIBRATED_DEFAULT_THRESHOLD
    safety_critical_threshold: float = UNCALIBRATED_SAFETY_CRITICAL_THRESHOLD
    safety_critical_categories: FrozenSet[str] = SAFETY_CRITICAL_CATEGORIES
    category_thresholds: dict = field(default_factory=dict)
    entity_match_hook: Optional[Callable[[object, object], bool]] = None


@dataclass(frozen=True)
class GatingDecision:
    should_answer: bool
    reason: str
    threshold_used: float


def _threshold_for_category(category: Optional[str], config: GatingConfig) -> float:
    if category in config.category_thresholds:
        return config.category_thresholds[category]
    if category in config.safety_critical_categories:
        return config.safety_critical_threshold
    return config.default_threshold


def decide(
    relevance: Optional[float],
    category: Optional[str],
    config: Optional[GatingConfig] = None,
    query_entities: object = None,
    retrieved_entities: object = None,
) -> GatingDecision:
    """
    Decide whether to answer or refuse. Pure function: no I/O, no globals.

    Args:
        relevance: max dense cosine similarity over the retrieved cards.
            Must NOT be an RRF fusion score (see module docstring). None or
            a missing/empty retrieval set should be passed as None, which
            always refuses.
        category: the router's category guess. Used ONLY to select a
            stricter threshold for safety-critical categories -- never to
            exclude retrieved documents (that hard-filter was Task 1's
            fix, and must not be reintroduced here).
        config: GatingConfig with thresholds and the optional entity-match
            hook. Defaults to GatingConfig() (uncalibrated placeholders).
        query_entities / retrieved_entities: optional, forwarded to
            config.entity_match_hook if one is set. Ignored entirely when
            entity_match_hook is None -- there is no implicit entity check
            in Phase 1.

    Returns:
        GatingDecision(should_answer, reason, threshold_used)
    """
    if config is None:
        config = GatingConfig()

    if relevance is None:
        return GatingDecision(False, "no_relevance_score", 0.0)

    threshold = _threshold_for_category(category, config)

    if relevance < threshold:
        return GatingDecision(
            should_answer=False,
            reason=f"relevance {relevance:.3f} below threshold {threshold:.3f} for category={category!r}",
            threshold_used=threshold,
        )

    if config.entity_match_hook is not None:
        if not config.entity_match_hook(query_entities, retrieved_entities):
            return GatingDecision(
                should_answer=False,
                reason="entity_mismatch",
                threshold_used=threshold,
            )

    return GatingDecision(should_answer=True, reason="relevance_ok", threshold_used=threshold)
