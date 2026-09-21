# Krishi Mitra — Architecture

> **Reconstructed 2026-09-21.** CLAUDE.md has pointed to this file as "the source
> of truth for design decisions and the phase roadmap" since it was written, and
> nine commits on `phase1-fixes` (2026-09-20) cite specific section numbers here
> (`Section 12`, `Section 13`, `Section 21`, `Section 31`) in code comments and
> commit messages. The file itself was never committed — checked the full git
> history, it does not exist at any revision. This reconstruction is built from
> those citations (which describe their target section's content in enough
> detail to recover it faithfully) plus `CLAUDE.md`, `TODO.md`, and
> `PROJECT_PLAN.md`. Section numbers below were chosen to match the existing
> citations; content between cited sections is genuine but was not literally
> "recovered" — treat it as current and correct, not as archaeology.
>
> If you find an actual prior version of this file (local backup, chat export,
> etc.), diff it against this one and prefer the original's wording for any
> section already cited elsewhere, since those citations were written against it.

---

## 1. Purpose

Design decisions and their reasoning, numbered so code comments and commit
messages can cite a stable section instead of re-explaining themselves. For
current facts (what's true right now — stack versions, known defects, rules),
see `CLAUDE.md`. For the ranked task backlog, see `TODO.md` and
`PROJECT_PLAN.md`. This file is decisions and *why*; it should change rarely.

## 2. System overview

FastAPI backend + Next.js frontend. Qdrant hybrid vector store (dense + learned
sparse via BGE-M3, fused with Reciprocal Rank Fusion). MongoDB for chat
history. LLM generation behind a backend-agnostic client (`llm_client.py`)
supporting Groq (hosted), OpenRouter (hosted, used as a cross-judge model), and
Ollama (local). See `CLAUDE.md` for exact current versions/models.

## 3. Pipeline

```
query -> LLM router (category + English gloss)
      -> embed query (BGE-M3 dense + sparse)
      -> hybrid retrieval, RRF fusion, top-5           [Section 12]
      -> abstention / confidence gate                   [Section 21]
      -> (optional) cross-encoder rerank                [Section 13]
      -> LLM generation, Kannada, context-only
      -> LLM faithfulness judge + numeric checker        [Section 22]
      -> escalation line appended                        [Section 23]
      -> save to MongoDB, return
```

## 4. Corpus

Single crop (sugarcane), `sugarcanemerged3.json`, provenance undocumented
(known defect, see CLAUDE.md). Chunking is structure-aware
(`_adaptive_chunk_section` / `load_and_chunk_data` in `vector_db.py`). Chunk
IDs are a deterministic content hash, not `uuid4()` — required so gold labels
in eval sets survive re-seeding the collection.

## 5. Embeddings

BAAI/bge-m3 via FlagEmbedding: dense + learned sparse in one model call, not
classical BM25. Two named vectors in Qdrant (`dense`, `sparse`), fused via
`FusionQuery(fusion=Fusion.RRF)`. BM25 as a possible *addition* (not
replacement) is evaluated separately — see Section 11.

## 6. Deployment modes

Qdrant runs via `docker-compose.yml` (server mode) as the primary path. Local
file mode (single-process lock) is a documented fallback only — it was the
original mode and caused real problems (`.git/index.lock`-style single-process
contention), which is why server mode is now primary.

---

## 10. Retrieval — hybrid RRF

Dense + sparse prefetch (limit 15 each) fused via Qdrant's native
`FusionQuery(fusion=Fusion.RRF)`, top-5 returned. RRF is rank-derived: a
document ranked first by both signals scores 1.0 *regardless of actual
relevance*. This is a load-bearing fact for Section 21 — do not reuse the RRF
fusion score as a relevance signal anywhere else in the pipeline.

## 11. BM25 (not wired in)

Not currently part of production retrieval. TODO #10 calls for a bucketed
evaluation (exact-term chemical/dosage queries vs. semantic/general queries)
before deciding whether to add it to the hybrid fusion — a flat aggregate
score would wash out exactly the signal that would justify including it
(BM25 is expected to win on exact-term lookups specifically, not in general).

## 12. Category filtering in retrieval — Option A: remove it

**Verified defect (fixed 2026-09-20, commit `1b57a3e`):** the LLM router
classifies each query into a category (`price` / `disease` / `pest` /
`fertilizer` / `general` — verified 2026-09-21 against the actual router
prompt in `rag_service.py`; `price` is not safety-critical so it's omitted
from `SAFETY_CRITICAL_CATEGORIES`, but it is a real router output). The
original code turned this into a Qdrant
`Filter(should=[FieldCondition(category=guess)])` on the retrieval query.

In Qdrant, a `Filter` with only `should` clauses is **not a soft boost** — it
hard-restricts results to points satisfying at least one `should` clause, when
that filter is applied directly to a search or prefetch leg. (Verified
2026-09-21 in `tests/invariants/test_category_filter.py`: a plain dense search
with an identical should-only filter returns only the matching category. The
*outer* `query_filter` on a `prefetch` + `FusionQuery` call did not reproduce
the exclusion against `qdrant-client` 1.18's `:memory:` backend in the same
shape the original bug used — re-verify against the real Docker Qdrant server
before treating that distinction as settled.)

Because the router can misclassify a query, this made gold chunks in a
*different* category from the router's guess completely unreachable — an LLM
classification made a document unreachable, which CLAUDE.md's rules forbid
outright.

**Option A (chosen):** remove the filter entirely. Retrieval is pure hybrid
dense+sparse RRF; `category` is not used to filter or boost retrieval at all.
Router category is still used downstream, only as an input to the Section 21
gate (stricter threshold for safety-critical categories).

**Option B (deferred, TODO #38):** implement a genuine soft boost — e.g. blend
a category-match bonus into the fusion score, or boost via Qdrant's `should`
inside a `must`-wrapped filter (`must=[Filter(should=[...])]` does not have
the same hard-exclusion behavior as a bare `should`-only filter) — rather than
no category signal in retrieval at all. Not done because it adds complexity
and risk on a codebase that just shipped the Option A fix; revisit once
Option A has run in production/eval without regressions.

## 13. Cross-encoder reranker — default off

`bge-reranker-v2-m3` is loaded in `vector_db.py` at startup but not called in
the live request path (`ENABLE_RERANKER=false` by default). Decision, from the
retrieval ablation: on a single-crop, tens-of-chunks corpus, first-stage
hybrid recall@5 is already near ceiling, leaving little for a reranker to fix.
Reranking is an *ordering* intervention (MRR/nDCG can detect it; recall@k
mostly can't), and even those showed gains within noise at this corpus size,
against ~30s/query CPU latency — real cost for a farmer on patchy rural 4G.

**This conclusion is corpus-size-dependent** and must be re-benchmarked when
the corpus scales to multiple crops (TODO #33), where first-stage recall will
no longer be saturated. The code path is kept alive behind the flag for
exactly that reason — this is a "not yet justified," not "rejected," verdict.

## 14. Query condensation (not done)

Chat history currently reaches the generation prompt but never retrieval —
multi-turn follow-ups ("what about for tomato?") aren't condensed into a
retrievable query. TODO #14; needs 5–10 multi-turn eval cases to demonstrate
before it can be trusted in the safety-critical path.

---

## 20. Generation

`llm_client.py` picks a backend via `LLM_BACKEND` (default `groq`), model via
`GENERATION_MODEL`. No hardcoded model-string fallbacks — changing model is a
one-line `.env` edit (this was itself a fix; see TODO's "Already done" list
for the prior Groq-deprecation incident, which is also why model IDs and dates
should be pinned explicitly in any report — a hosted-API deprecation is a
real, citable reproducibility risk, not a hypothetical one).

## 21. Abstention / confidence gate — defense in depth

**Verified defect (fixed 2026-09-20, commit `f55fb3c`):** the original gate
compared the Qdrant **RRF fusion score** against `HARD_REFUSAL_THRESHOLD`
(0.35). Per Section 10, RRF scores are rank-derived, not relevance-derived — a
top-ranked-but-irrelevant hit can score 1.0. The project's own
`refusal_results.jsonl` recorded an unanswerable price query scoring 1.0: the
gate never fired. This is a safety-critical failure mode in a domain where
"refuse" must always beat "guess."

**Fix:** `backend/services/gating.py` — a pure function
`(relevance, category, config) -> decision`, with no Qdrant/embedding/LLM
dependency (unit-testable in isolation, see `tests/invariants/test_gating.py`).
Callers compute `relevance` as max dense cosine similarity over retrieved
cards (dense vector space is COSINE distance, so this is a genuine relevance
score) — never the RRF fusion score.

**Defense-in-depth layers, by phase:**

1. **Layer 1 (live, Phase 1):** relevance threshold, `default_threshold`.
2. **Layer 2 (live, Phase 1):** stricter `safety_critical_threshold` for
   `SAFETY_CRITICAL_CATEGORIES = {pest, disease, fertilizer}`, plus a
   `category_thresholds` override hook for future per-category calibration.
3. **Layer 3 (Phase 4, not implemented):** `entity_match_hook` extension
   point — refuse if the query's entity (e.g. a specific chemical name) does
   not appear in any retrieved card, regardless of relevance score. This is
   the highest-value addition not yet built: a relevance score alone can be
   fooled by a card that is topically close but doesn't actually name the
   thing the farmer asked about. Signature when implemented:
   `(query_entities, retrieved_entities) -> bool`. `None` (current default)
   means entity matching is skipped entirely — never silently treated as a
   pass.

**All current thresholds are explicitly UNCALIBRATED placeholders.**
`backend/eval/threshold_sweep.py` implements the sweep methodology (pick by
F1 against a labelled answerable/unanswerable set). Fixed 2026-09-21: it
originally scored candidate thresholds against the RRF fusion score -- the
exact wrong signal this section documents -- which would have calibrated a
threshold incompatible with what this gate actually compares against. It
now scores a plain dense-only query, matching `get_dense_relevance()`
exactly. It still has not been *run* against a real labelled gold set --
that's Phase 2 work, blocked on human gold labeling, independent of the
scoring-signal fix. Do not report these numbers, or any eval run using
them, as validated until it has been run against real data.

## 22. Faithfulness judge + numeric checker

LLM faithfulness judge cross-checks generated answers against retrieved
context. `eval/numeric_faithfulness.py` regex-cross-references dosage/quantity
numbers in the answer against the context as a second, non-LLM check.

**Verified defect, fixed 2026-09-21 (commit `1929c7d`):** it previously bound
a number to any unit up to 3 words away with no check that a different
number sat in between, mis-binding units across adjacent dosage pairs and
sometimes dropping pairs entirely -- the checker's own "safe, faithful
paraphrase" demo scored 0.0 instead of 1.0. Fixed to bind each number only
to the nearest unit strictly before the next number or a sentence boundary.
Re-verified 2026-09-21: `python backend/eval/numeric_faithfulness.py` now
scores that demo 1.0 with 0 violations, and 4 dedicated regression tests
pass (`tests/invariants/test_numeric_faithfulness.py`).

Even fixed, this is still regex-based pattern matching, not structured
extraction against a verified fact record -- it can only catch numbers the
generation model invents or alters, not chemical names, and can over-fire
on paraphrased step counts/dates in `strict=True` mode. Not a substitute for
the human dosage-verification step. Replacing it with field-identity
verification against typed fact records is the "structured extraction for
safety-critical fields" upgrade path (TODO #40), not done here.

## 23. Escalation line

Every answer appends the Kisan Call Centre number (1800-180-1551) / nearest
Raitha Samparka Kendra, in Kannada. This is a floor, not a substitute for
Section 21's gate — the point is that even a correctly-gated "I don't know"
still gives the farmer a next step.

Verified/fixed 2026-09-21: refusal and timeout messages already embed the
number literally, but the success path only appended it when confidence was
medium, not on every gate-approved answer -- contradicting this section as
written (and PROJECT_PLAN.md P3.1's explicit "every answer" spec). Fixed in
`rag_service.py`'s generation branch; the now-dead `CONFIDENT_SEARCH_THRESHOLD`
constant and `is_medium_confidence` variable were removed with it.

---

## 30. Repo hygiene

Dead code and stale artifacts get *archived or deleted with a documented
reason*, not silently left in place or silently deleted without checking
importers. Every deletion in the 2026-09-20 kill-list pass was preceded by a
repo-wide grep for importers (per CLAUDE.md's rule: grep before deleting).

## 31. Kill-list & stale-eval policy: regenerate, do not patch

When old eval artifacts describe a corpus state that no longer exists (stale
gold IDs, circular labels, superseded question sets), the fix is to
**regenerate them, not patch their paths to point at the stale data**. Moved
`chunk_id_map.txt`, `gold.jsonl`, `retrieval_results.jsonl`, `contexts.json`,
`results.jsonl`, `refusal_results.jsonl`, `questions_UPDATED.json` to
`backend/eval/_archive_stale/` (with a README explaining why each one is
stale) rather than deleting them outright — they're evidence for the report
(e.g. `refusal_results.jsonl` is the actual record of the Section 21 defect:
an unanswerable query scoring 1.0).

Scripts that read these paths (`run_retrieval_ablation.py`, `refusal_test.py`,
`threshold_sweep.py`, `reranker_diagnostic.py`, `generate_report_tables.py`,
`export_for_human_eval.py`) were **deliberately not repointed** at the
archived copies. They should fail loudly with `FileNotFoundError` until Phase
2 produces a real, current gold set — silently pointing them at circular,
non-resolving stale data would let them keep running and producing numbers
that look valid but aren't. `dump_chunks.py` and `build_contexts.py` write
fresh output to their original (non-archived) paths when run — that's
regeneration, which is fine.

Dead code removed under this policy needs a clean "no importers" grep first
(`semantic_chunker.py`, `unlock_db.py`, the flat loader/sliding-window chunker
in `vector_db.py`, the Tavily web-fallback client instantiation — TODO #37 —
all confirmed orphaned before removal).

## 32. Eval artifact lifecycle

`_archive_stale/` is a graveyard with a README, not a second source of truth.
New eval runs write to their own directory (per CLAUDE.md's rule), recording
dataset hash, model ID, and prompt version, so results are never silently
compared across incompatible runs.

---

## 40. Phase roadmap

Full task-level detail lives in `TODO.md` (ranked, two-track) and
`PROJECT_PLAN.md` (P0–P4 work order with rationale). Summary:

- **Phase 1 (demo-ready):** defects fixed 2026-09-20/21 (Sections 12, 21, 22,
  23, plus dependency/hygiene/eval-pipeline fixes) + Docker preflight script
  (`backend/preflight.py`) -- these are done and verified. Still remaining:
  running the threshold sweep and BM25 bucketed eval against a *real* gold
  set (the tooling is now correct, per Sections 11/21, but hasn't been run
  against real data -- blocked on Phase 2 human gold labeling), a data
  provenance note (blocked on actually knowing the provenance, which nobody
  does yet -- do not fabricate one), and manual failure-mode review (needs
  live LLM outputs + human judgment).
- **Phase 2 (rigor, pre-paper):** real gold-labeled retrieval eval, no-retrieval
  baseline, expanded eval set via KCC, BERTScore/RAGAS, statistical
  significance testing, human evaluation.
- **Phase 4 (scope upgrades, future work only — not built):** entity-match
  gate hook (Section 21), voice input/output, multi-crop expansion (re-run
  Section 13's reranker ablation after — the ceiling effect may not hold),
  WhatsApp channel, image-based diagnosis, live tool-calling (mandi prices,
  weather).
- **Phase 5:** mentioned in eval-script comments as a "model bake-off" (3-model
  generation comparison — gemma/llama/sarvam — on frozen contexts). Corresponds
  to PROJECT_PLAN.md's P2.

## 41. What "done" looks like

See `PROJECT_PLAN.md`'s "Target outcome" section for the full framing. One
line worth repeating here since it drives every decision above: **the honest,
measured negative results are the contribution** — a reranker that doesn't
help *yet*, metrics that were lying and got fixed, thresholds explicitly
marked uncalibrated rather than dressed up as validated. Don't let future
edits quietly upgrade a "not yet justified" into a claimed result without
re-running the evidence that would justify it.
