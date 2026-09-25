# Krishi Mitra — Complete TODO List

Ranked master list of every open task discussed, split into two tracks:
**Track 1** — 2-week plan to have a good, defensible model to show the professor.
**Track 2** — the month before submitting a research paper.

Items already completed are listed at the top for reference and not repeated below.

---

## Already done ✅

- README merge conflict resolved
- Placeholder contact info replaced with real Kisan Call Centre number (1800-180-1551)
- `llm_client.py` and `ChatMessage.tsx` wired into live request/render paths (previously
  built but orphaned)
- ROUGE-L diagnosed as broken on Kannada (self-score 0.625, 6/16 questions scored 0.000);
  replaced with chrF (validated: self-score 1.0, unrelated-pair floor 0.172). **Re-confirmed
  2026-09-21** by re-running `python backend/eval/diagnose_metrics.py` fresh: identical
  numbers reproduced (same 6 questions at rougeL_self=0.000: disease-3, disease-4,
  fertilizer-3, general-1, general-4, price-1).
- ~~Gold-labeled retrieval ablation~~ — **VOID, superseded 2026-09-21/22.** The original
  numbers here predated the normalization fix and the current chunk set. **Do not cite them
  -- see "Real evaluation results" below for the current, verified numbers**, generated
  against gold.jsonl as of commit `a84ba34` and re-run through the pipeline as of commit
  `94e6e42`.
- Reranker decision made and justified (off by default; near-ceiling recall at current
  corpus size; MRR/nDCG gain real but corpus-size-dependent, must re-benchmark on scale-up)
- Hard refusal + faithfulness gate + escalation message live in `rag_service.py`
- Duplicate `refusal_test.py` cleaned up
- Groq deprecation fix (#1): `llm_client.py`'s `_DEFAULT_MODEL` is a single env-driven
  constant (`GENERATION_MODEL`), no hardcoded per-backend fallback strings. **Verified
  2026-09-21** via direct grep of `llm_client.py` — confirmed already fixed prior to this
  session; just re-checked here since it was still listed as open in the ranked list below.
- Dead `tavily_client` (#37) removed. **Verified 2026-09-21** via grep: no
  `ENABLE_WEB_FALLBACK` code remains anywhere in `backend/`, only a comment documenting the
  removal in `rag_service.py`.

### This session (2026-09-21) — fixes made and verified with actual re-runs, not just re-reading commit messages

Safety-critical fixes:
- **Numeric faithfulness checker mis-bound numbers to units** (#5, defect #3). Fixed in
  `backend/eval/numeric_faithfulness.py` (commit `1929c7d`). **Verified fresh 2026-09-21**:
  re-ran `python backend/eval/numeric_faithfulness.py` — the "Safe: all numbers supported"
  case now scores 1.0 / 0 violations (previously scored 2 violations on its own positive
  control). All 4 dedicated regression tests pass
  (`tests/invariants/test_numeric_faithfulness.py`).
- **Escalation line (Kisan Call Centre number) wasn't on every answer**, contradicting its
  own spec. Fixed in `rag_service.py` (commits `6f979f5`, `3295c6f`). **Verified fresh
  2026-09-21** by tracing every code path that produces the final answer string (6 distinct
  paths: gate refusal, semantic/numeric-fail refusal, success path, timeout, generic
  exception, empty-LLM-response fallback) — all 6 contain the number. One nuance found
  during this trace and now documented honestly in the code: the empty-LLM-response fallback
  message could theoretically receive the number twice (harmless duplicate, never missing)
  if the faithfulness judge scores that canned message unexpectedly high; this is
  judge-behavior-dependent, not code-enforced, and the original comment overclaimed a
  guarantee — corrected (commit `10474d9`).
- **Kannada whitespace normalization was a silent no-op** whenever `indic-nlp-library` is
  installed (it is, in this environment and per `requirements.txt`). Fixed in
  `indic_preprocess.py` (commit `f7c12c1`). **Verified fresh 2026-09-21** with an actual
  before/after run: loaded the pre-fix code from git history and the current code
  side by side, ran the same real Kannada string with double/triple spaces through both —
  before left the double-spaces intact, after collapsed them to single spaces. Confirmed
  this does NOT affect the ROUGE-L/chrF self-score numbers above (that script never calls
  `normalize_kannada`), but DOES affect the retrieval ablation numbers (see caveat above).
- `gold.jsonl` circularity safeguard in `auto_relabel_gold.py` (commit `f2baaf3`).
  **Verified fresh 2026-09-21**: confirmed `eval/gold.jsonl` does not exist (only the
  archived stale copy does), confirmed the script's only write target is
  `gold.jsonl.draft` (grepped every reference), and reproduced the actual
  `FileNotFoundError` a downstream script hits with `gold.jsonl` absent.

Eval-pipeline correctness (each of these would have silently produced wrong numbers the
first time it was actually run):
- `threshold_sweep.py` scored candidate thresholds against the RRF fusion score instead of
  the dense-cosine relevance the live gate actually uses — would have calibrated a threshold
  on the wrong scale entirely. Fixed (commit `3f8999e`). Tooling is now correct; still not
  run against real data (blocked on Phase 2 gold labeling, see below).
- `build_contexts.py`'s frozen context format (`<doc>` tags) didn't match what
  `rag_service.py` actually sends in production, so the P2 generation comparison wasn't
  testing what real users see. Fixed (commit `0e873bc`).
- `run_eval.py` appended results across runs to one shared `results.jsonl` with no dataset
  hash or prompt version recorded — a direct violation of this file's own stated rule
  above ("never append to a shared file"). Fixed (commit `f617975`): each run now writes
  its own file under `eval/runs/` with dataset hash, prompt version, and timestamp;
  `export_for_human_eval.py` combines them explicitly and warns on inconsistent runs.
- Two unpaginated `Qdrant.scroll(limit=1000)` calls (`vector_db.py`'s seeding-verification
  step and `dump_chunks.py`) would silently truncate past 1000 chunks — currently dormant
  at ~43 chunks, would bite on multi-crop expansion (#33). Fixed (commit `8d46967`).
- `generate_report_tables.py` formatted the fractional `recall@5` metric with `:.0f`,
  which would round e.g. 0.5 to a misleading 0 or 1. Fixed (commit `391b8db`).
- "Show sources" UI could fragment one real retrieved chunk into multiple fake ones, since
  a chunk's own flattened text can contain an internal blank line and the old code
  recovered sources by splitting on that same delimiter. Fixed (commit `2eb25e5`):
  `rag_service.py` now returns the un-joined chunk list directly.

Frontend:
- `GET /history/{session_id}` was built on the backend, never called from the frontend —
  chat history was lost on every refresh, sidebar chats had no click handler at all. Fixed
  (commit `258d3e6`): session persistence via localStorage, history fetch on load/switch.
  Verified via `tsc --noEmit` (clean) and tracing the response shape
  (`chat_db.get_chat_history()`'s actual fields) against what the frontend expects — could
  not verify interactively (no browser-automation tool available, and the real backend
  needs Docker/Qdrant, not available in this environment). **Still needs a manual
  click-through with the backend actually running before demo.**
- Removed a redundant, English-only "Need help? Kisan Call Centre" footer from
  `ChatMessage.tsx` that violated the Kannada-only user-facing text convention and (after
  the escalation-line fix above) duplicated the number. Fixed (commit `3295c6f`).
- Page metadata advertised "Ragi, Jowar, Sugarcane, Areca nut, Coffee" — corpus is
  sugarcane-only. Fixed (commit `db1134c`).

Docs/hygiene:
- `ARCHITECTURE.md` reconstructed (commit `507e7b6`) — cited by 9 commits and CLAUDE.md,
  never actually existed anywhere in git history.
- Explicit documentation-drift audit performed 2026-09-21 (commit `d8fe992`): grepped every
  factual claim in README.md/ARCHITECTURE.md/AGENTS.md against actual code. Found and fixed:
  ARCHITECTURE.md Section 12 was missing the router's `price` category; Section 22 still
  described the numeric faithfulness checker as broken after it had been fixed earlier in
  the same session; Section 21's `threshold_sweep.py` description predated its own fix;
  Section 40's roadmap still listed two already-done items as remaining; README didn't
  mention `ChatResponse`'s `sources` field or 2 of the project's 6 test files;
  `.env.example` was missing the `JUDGE_MODEL` variable (`run_eval.py` requires it).
  AGENTS.md checked clean (one-line pointer to CLAUDE.md, nothing to drift).
- BLEU added to `diagnose_metrics.py` (#3, commit `1147efb`) and actually **run** 2026-09-21
  (not just added): self-score = 1.000 across all 16 questions (zero failures), unrelated-
  pair floor = 0.028, usable range = 0.972. **This does not match the original framing of
  this task** ("show him why BLEU fails like ROUGE-L") — BLEU does NOT show the same
  structural self-score breakdown ROUGE-L does on this corpus (ROUGE-L: 6/16 questions
  scored 0.000; BLEU: 0/16). Likely explanation: sacrebleu's tokenizer handles Kannada
  script more gracefully than `rouge-score`'s. CLAUDE.md's separate, still-valid claim that
  BLEU/chrF can't detect numeric errors (a 10x overdose can score higher than a valid
  paraphrase) is a different failure mode entirely, untested by this self-score diagnostic.
  **Bring this nuance to the professor, not a flat "BLEU fails" claim.**
- `.env.example` (#6): pre-existing, plus `JUDGE_MODEL` added 2026-09-21 (doc-drift fix
  above).
- Docker preflight script added (#9, commit `a2c6b3a`): `backend/preflight.py` checks
  Qdrant reachability/collection/point-count and MongoDB reachability. Verified when
  authored: ran it against this environment (no Docker running) and confirmed it fails
  cleanly with actionable messages rather than crashing; also verified the connection-
  refused exception path separately.
- 18 new tests added (#30, commits `4db5163`, `920396e`, `ba2bf64`, `f7c12c1`,
  `1929c7d`, `584e1ee`-adjacent): chunking helpers (`flatten_value`, `_infer_category`,
  `_split_large_text`, chunk-ID determinism), normalization (whitespace collapse, strip),
  numeric faithfulness regressions, requirements-import resolution. **Verified fresh
  2026-09-21**: full suite re-run at commit `db1134c` (pushed), 32/32 passed in 0.85s.
- `temperature=0.0` design-choice note (#39), PII/prompt-injection scope note (#41),
  Production Readiness section (#42) — all added to README.md (commit `1147efb`).
- Closed a root-level `.gitignore` gap for `.env` / local Qdrant store (commit `615d655`);
  declared `pymongo` explicitly and dropped a stale `psutil` comment in `requirements.txt`
  (commit `cb7e95a`).

### Real evaluation results (2026-09-21/22) — the first valid numbers this project has had

Every number below was generated by actually running the pipeline this session, not
carried forward from before. Cites the exact commit each artifact came from. **Void any
number anywhere else in this file or README.md that predates commit `1b57a3e`** (the start
of this session's fixes) — none of it was generated against the current chunking, the
current gate, or a non-circular gold set.

**Pipeline determinism (commit `7a41d47`):** `python vector_db.py` run twice fresh. Both
runs: 43 chunks generated = 43 stored, 0 duplicates, exit 0. Point ID sets identical
between runs (`diff` empty). This required fixing a real bug first: qdrant-client's local
file-mode backend does not reliably clear a pre-existing on-disk collection across process
restarts (isolated with a minimal repro) -- a naive re-seed was silently accumulating stale
chunks (43 new + old data = 80 stored points) instead of replacing them.

**Gold labels (commit `a84ba34`):** 18 questions (16 original + 2 added), matched against
all 43 current chunks by reading actual chunk content (not just top-10 retrieved -- see
that commit's message for why). 15 answerable, 3 unanswerable (`price-1`, plus 2 new:
`pest-5` a pest not in the corpus, `general-5` a specific number the corpus's general topic
mention doesn't contain). `python eval/check_gold_integrity.py` confirms all 18
gold_chunk_id references resolve in the current chunk set.

**query_type distribution (commit `a84ba34`):** entity-specific=10, quantity-specific=5,
semantic=2, procedural=1 (was 16/0/0/0 before -- the field was entirely absent from
questions.json).

**Retrieval ablation (commit `b7d3c0c`), n=15 answerable, dense/hybrid/bm25+dense only
(reranker variants skipped per standing decision):**

| Config | recall@5 | MRR | nDCG@5 |
|---|---|---|---|
| dense | 0.933 ± 0.131 | 0.848 ± 0.160 | 0.862 ± 0.154 |
| hybrid | 1.000 ± 0.000 | 0.749 ± 0.167 | 0.817 ± 0.125 |
| bm25+dense | 0.933 ± 0.131 | 0.819 ± 0.162 | 0.842 ± 0.152 |

n=15 is thin -- several bucket-level CIs are very wide (e.g. `hybrid`|quantity-specific
(n=3) nDCG@5 95% CI is ±0.372). Full per-bucket table in `eval/retrieval_results.jsonl`.
Bucketing now actually works (previously silently 16/0, see above).

**Threshold sweep (ran, not committed as a file -- `threshold_sweep.py` only prints), n=18
(15 positive, 3 negative), against gold as of `a84ba34`:** raw dense-cosine relevance
scores ranged 0.497–0.668 for **answerable** questions and 0.499–0.597 for the 3
**unanswerable** ones -- i.e. the score ranges overlap. The script's naive F1-maximizing
recommendation is threshold=0.10, achieving **TNR=0.000** (rejects zero of the 3 negatives)
for both the general and safety-critical subsets. **Do not adopt this.** F1 on 1-3 negative
examples is dominated by the majority class and says nothing about specificity. The
honest reading: with only 1 negative example in the safety-critical subset, no threshold
choice from this data is statistically calibrated. Recommendation: **do not change
`UNCALIBRATED_DEFAULT_THRESHOLD` (0.35) or `UNCALIBRATED_SAFETY_CRITICAL_THRESHOLD` (0.50)
from this sweep** -- collect more negative examples first (Phase 2), and prioritize the
entity-match hook (see the hallucination finding below) over threshold-tuning, since this
sweep shows relevance alone cannot separate "topically similar but wrong" from "correct."

**Generation eval, `GENERATION_MODEL=openai/gpt-oss-20b` (not the deprecated
llama-3.1-8b-instant), self-judged (see below for why not cross-judged), n=18, saved to
`eval/runs/openai-gpt-oss-20b__ad0295196c09__20260921T235104.jsonl`:**

- avg chrF = 0.413 (unrelated-pair floor 0.172 -- signal above floor: **0.241**)
- avg embedding-sim = 0.729 (unrelated-pair floor 0.524 -- signal above floor: **0.205**)
- `fertilizer-1` scored embsim=0.460, *below* the unrelated-pair floor -- the model
  answered from a genuinely different-but-also-corpus-grounded nitrogen schedule (the
  corpus contains two valid schedules; see future-work item on multiple acceptable
  answers). Not a hallucination, but a real multi-answer ambiguity the metric can't see.
- avg faithfulness (self-judged) = 0.611, but **this number is not trustworthy** -- see
  the two judge defects below, both discovered and partially fixed mid-run.

**Critical finding — a live, reproduced hallucination on a safety-critical query.** Asked
about "black beetle" (a pest not in the corpus, `pest-5`), the LLM did not refuse. Instead
it presented the **Termite treatment's exact dosage** (Chlorantraniliprole 0.4G / Fipronil
0.3G) as the answer, because retrieval pulled the Termite chunk in as topically similar
(dense relevance 0.597 -- above *both* current gate thresholds). This is not hypothetical:
it happened, in this exact pipeline, this session. All three current safety layers missed
it: (1) the relevance gate passed it (0.597 > 0.50), (2) the LLM ignored its own
system-prompt instruction to refuse when the answer isn't in context, (3) the numeric
faithfulness checker (fixed earlier this session) would not have caught it either -- "0.4"
and "ಜಿ" genuinely appear in the retrieved context, just attached to the wrong pest. This
is the single strongest piece of evidence yet for ARCHITECTURE.md Section 21's
already-identified but unbuilt Layer 3 (entity-match hook) -- promoting that from
"someday" to the top of Track 1/2 priority (new item #43 below).

**Two judge defects found and partially fixed mid-run (commits `c5300ae`, `94e6e42`):**
`calculate_faithfulness`'s `max_tokens=50` silently returned an EMPTY completion for the
currently-configured model (a reasoning model that spends tokens on internal reasoning
before visible output) -- this would have made the live faithfulness gate refuse **every
single answer** (0.0 always < 0.50). Raised to 300, still failed on 5/18 (longer contexts
need more headroom); raised to 500, verified against the longest real context in this
run. Separately, the judge scored two objectively-correct refusals (`price-1`, `general-5`,
both chrF=1.000) as 0.0 faithfulness -- a real prompt-design gap (no concept of "declining
to answer is correct here"), not fixed, needs its own calibration work (new item #44).

**OpenRouter cross-judge (TODO #4) is blocked by more than paid calls.** A real
`OPENROUTER_API_KEY` exists, but `llm_client.py`'s `LLM_BACKEND` is a single process-wide
setting, not selectable per-call -- there is currently no way to make one call go through
Groq (generation) and another through OpenRouter (judging) in the same run. This needs an
architecture change to `call_llm`/`calculate_faithfulness`, not just an API key (new item
#45).

---

## Ranked master list

Rows are never renumbered — `ARCHITECTURE.md` and code comments cite these numbers
directly (e.g. "TODO #37", "TODO #38"). Completed items are annotated in place rather than
removed or renumbered.

| # | Task | Track |
|---|---|---|
| 1 | ~~Fix Groq deprecation~~ — ✅ **DONE**, verified 2026-09-21 (see Already done) | 1 |
| 2 | Re-run eval against new model; annotate old `results.jsonl` as predating the deprecation | 1 |
| 3 | ~~Run BLEU through the self-score diagnostic~~ — ✅ **DONE + RUN** 2026-09-21 (see Already done — result contradicts the original framing, read the note) | 1 |
| 4 | Add OpenRouter backend branch in `llm_client.py`; use as **cross-judge** model (different provider judging a Groq-generated answer, closing the self-evaluation gap) — **code exists** (`llm_client.py`'s `_openrouter_chat_sync`, `run_eval.py`'s `JUDGE_MODEL`), **never actually run** — blocked on paid API calls | 1 |
| 5 | ~~Numeric faithfulness checker~~ — ✅ **DONE + VERIFIED** 2026-09-21 (see Already done) | 1 |
| 6 | ~~Add `.env.example`~~ — ✅ **DONE** (see Already done; `JUDGE_MODEL` gap closed 2026-09-21) | 1 |
| 7 | ~~Threshold sweep~~ — ✅ **RUN** 2026-09-21/22 against real gold (see "Real evaluation results"). Result: do not adopt the naive F1-optimal threshold (TNR=0 on both subsets); current uncalibrated defaults left unchanged pending more negative examples | 1 |
| 8 | Data provenance note for `sugarcanemerged3.json` — **still open, correctly**: provenance is genuinely unknown, do not fabricate a source/date. **New finding 2026-09-24** (ARCHITECTURE.md Section 4): the file's own embedded `metadata` block claims `region: Karnataka`, contradicted by its own Tamil Nadu-specific content (TNAU variety data, named TN districts) -- narrows but does not resolve the open question; the self-declared metadata is now known to be unreliable, not just silent | 1 |
| 9 | ~~Docker Compose + preflight script~~ — ✅ **DONE** (see Already done) | 1 |
| 10 | ~~BM25 evaluation, bucketed by query type~~ — ✅ **RUN** 2026-09-21/22 (bm25+dense config only, per prior decision -- see "Real evaluation results"). bm25-alone and bm25+hybrid still not run; add if a fuller ablation is wanted. **Root cause found 2026-09-23** for why bm25+dense outperformed dense/hybrid on entity-specific queries like disease-3: BGE-M3's dense embedding doesn't reliably separate structurally-templated corpus entries. Every disease/pest chunk shares the same "Name: X (Y) Recommendations: Chemical: ... Dosage: ... Notes: ..." template, and when two entries also share the same remedy pattern (e.g. both Smut and Yellow Leaf Disease are "Cultural Control, uproot and burn"), dense cosine similarity is dominated by the shared boilerplate -- verified directly by re-embedding query/chunk pairs outside Qdrant: query-vs-wrong-chunk consistently scored higher than query-vs-gold-chunk (Smut case: 0.500 vs 0.546 wrong; Termites case: 0.489 vs 0.497 wrong). BGE-M3's learned-sparse component doesn't reliably fix this either (razor-thin, sometimes wrong-direction margins). ~~This is now a well-evidenced case for wiring BM25 into the production hybrid fusion~~ — ✅ **DONE 2026-09-23** (commit `c9f209e`), at the project owner's explicit go-ahead. `rag_service.py` now fuses BM25+dense (matching the validated ablation config) for `category in {pest, disease}` only -- NOT applied to fertilizer/general/price, since the same ablation showed BM25 fusion hurting the semantic bucket. Verified against the live corpus with the actual production code: disease-3's gold chunk moved from missing (dense) / rank 5 (hybrid) to rank 3; **pest-2 -- the exact case that hallucinated a different pest's dosage in Step 6's eval run -- now retrieves correctly at rank 1** | 1 |
| 11 | Manual failure-mode review on existing eval results — still open, blocked on live LLM outputs + human judgment | 1 |
| 12 | GitHub Action running `refusal_test.py` on push — still open, needs explicit user sign-off (CI/CD change) + secrets configured | 1 |
| 13 | Chunking rebuild: Indic NLP Library preprocessing (agglutination handling) + field-aware chunking, bundled as one pass — relabel `gold.jsonl` and re-run the ablation once, not twice — still open, deliberately deferred as its own scoped conversation | 1 (start) / 2 (finish) |
| 14 | Query condensation for multi-turn conversations (chat history currently reaches generation, never retrieval) + 5–10 multi-turn eval cases to prove it | 1 (if time) / 2 |
| 15 | Streaming responses (Groq SSE + frontend consumer) | 1 (cut first if short on time) |
| 16 | Reframe the paper's thesis around the metric-validity finding ("evaluation pitfalls in Kannada RAG") rather than "chatbot system description" — decide this *before* collecting more data, since it shapes what's worth collecting | 2 |
| 17 | No-retrieval baseline: run eval questions through the generation model with empty context, to prove RAG is actually adding value over parametric knowledge alone | 2 |
| 18 | Expand eval set to n=40–100 using the Kisan Call Centre (KCC) public dataset | 2 |
| 19 | External validity check — prioritize KCC-sourced (independently written) questions over corpus-derived ones, to avoid the appearance of evaluating on easy, leakage-adjacent questions | 2 |
| 20 | Add BERTScore (semantic/embedding-based overlap, complements chrF's lexical overlap) — validate on Kannada with the same self-score discipline before trusting it | 2 |
| 21 | Adopt the RAGAS framework (faithfulness, answer relevancy, context precision/recall) to replace the hand-rolled faithfulness judge with a named, published framework | 2 |
| 22 | Statistical significance testing (bootstrap or Wilcoxon) on ablation deltas, not just eyeballing CI overlap | 2 |
| 23 | Human evaluation on a subset, with a documented evaluator-selection and instruction process (even informal — reviewers expect this noted) | 2 |
| 24 | Reproducibility package — publish gold labels, metric-diagnostic scripts, eval harness alongside the paper | 2 |
| 25 | Data licensing check for KCC / data.gov.in redistribution, before finalizing the eval set as a paper artifact | 2 |
| 26 | Venue selection — Indic-NLP-specific workshops (LREC, ACL/EMNLP low-resource tracks, ICON) rather than a general top-tier venue | 2 |
| 27 | Pin exact model versions and dates throughout; cite the Groq deprecation explicitly as a reproducibility lesson for hosted-API research | 2 |
| 28 | Restore/debug local Ollama inference — now higher priority given the Groq deprecation was a live demonstration of hosted-API fragility | 2 |
| 29 | Resolve the sarvam-2b base-vs-instruct model question (moot until #28 is done) | 2 |
| 30 | ~~Basic unit tests beyond `refusal_test.py`~~ — ✅ **DONE** 2026-09-21 (see Already done) | 2 (if time) |
| 31 | Add a LICENSE file and a `pip freeze`-pinned lockfile for full reproducibility — lockfile ✅ done (`requirements.lock`, from `phase1-fixes`); LICENSE file still open (needs a license choice — ask, don't assume) | 2 (if time) |
| 32 | Voice input/output (STT/TTS) | 2 — future work section only, not built |
| 33 | Multi-crop expansion (ragi, tomato) + re-run the retrieval ablation, since the reranker conclusion is corpus-size-dependent and may flip | 2 — future work section only |
| 34 | WhatsApp channel | 2 — future work section only |
| 35 | Image-based pest/disease diagnosis from photo | 2 — future work section only |
| 36 | Live tool-calling integration (Agmarknet mandi prices, IMD weather) | 2 — future work section only |
| 37 | ~~Web fallback is dead code~~ — ✅ **DONE**, removed in the kill-list commit prior to this session; verified via grep 2026-09-21 (see Already done) | 1 |
| 38 | Wire Qdrant payload filtering by category — still open, **deliberately deferred**: documented as "Option B" in `ARCHITECTURE.md` Section 12, to be revisited once Option A (the hard-filter removal) has run in production/eval without regressions, which hasn't happened yet | 1 (if time) / 2 |
| 39 | ~~Document `temperature=0.0`~~ — ✅ **DONE** 2026-09-21 (see Already done) | 1 |
| 40 | Structured extraction for safety-critical numeric fields (chemical name/dose/unit as a validated schema) rather than free-text generation — a stronger version of the numeric faithfulness checker (#5); note as an extension/upgrade path even if #5 ships first | 2 |
| 41 | ~~PII/prompt-injection limitation note~~ — ✅ **DONE** 2026-09-21 (see Already done) | 1 |
| 42 | ~~Production Readiness section~~ — ✅ **DONE** 2026-09-21 (see Already done) | 1 |
| 43 | ~~Entity-match hook (ARCHITECTURE.md Section 21, Layer 3)~~ — ✅ **SHIPPED 2026-09-23** (commit `45a951b`), scoped to pest/disease only. First attempt (pre-BM25-fusion retrieval) false-positived on 6 of 18 questions and was shelved. Re-tested after #10 (BM25 fusion) landed: **9/9 correct for pest+disease** -- all 8 answerable pass, pest-5 correctly triggers `entity_mismatch`. Verified end-to-end against the real `GatingConfig`/`decide()` call: pest-5 now returns `should_answer=False, reason=entity_mismatch` -- the reproduced hallucination from Step 6 is caught by the live gate. NOT widened to fertilizer/general (not re-tested there, same retrieval-quality problem presumably still present) | 1 |
| 46 | **`pest-2` refused by an unrelated, uncalibrated threshold — found 2026-09-23, root-caused, opt-in fix shipped (default OFF).** Two real bugs found while investigating: (1) the relevance gate was scoring a *different chunk* than the one BM25+dense fusion actually used to build the answer (fixed, commit `9840812`); (2) `rrf_fuse()`'s tie-breaking was non-deterministic across process restarts due to Python's per-process hash randomization -- the exact same query could retrieve a different top chunk on different runs (fixed, commit `a1afe01`). After both fixes, the TRUE finding holds: `pest-2` (correct) scores relevance=0.489, LOWER than `pest-5` (a wrong-entity hallucination) at 0.597 -- no single threshold value separates them, because `entity_match_hook` (validated 9/9) now does that job instead. Added `ENABLE_LENIENT_PEST_DISEASE_THRESHOLD` (`.env`, **default false**) to lower pest/disease's threshold to the general default (0.35) when explicitly opted into -- verified both states: off leaves `pest-2` refused (unchanged), on answers it correctly while `pest-5` still correctly refuses via entity_mismatch. Left off by default deliberately: this weakens a safety-critical refusal control, and that's the project owner's decision to make explicitly, not something to infer from general delegation ("do what's best") -- confirmed by this session's own safety classifier blocking the first attempt to apply it directly | 1 |
| 44 | ~~Faithfulness judge prompt doesn't handle refusals~~ — ✅ **FIXED 2026-09-23** (commit `a330b09`). Extracted the exact instructed refusal phrase into `NOT_IN_CONTEXT_PHRASE`; `generate_from_context()` now recognizes it verbatim and skips the judge for that case (scored 1.0) instead of letting a faithfulness judge mis-score a correct refusal as unfaithful. Verified: full test suite + hand-traced through the downstream semantic_fail/numeric_fail logic. **Not yet verified with a live LLM call** (needs a real eval re-run to confirm end-to-end -- next time #2/#6 runs, check price-1/general-5 specifically) | 1 |
| 45 | ~~OpenRouter cross-judge (#4) needs an architecture change~~ — ✅ **FIXED 2026-09-23** (commit `88a24cd`). Added `backend=` param to `call_llm()`, threaded through `calculate_faithfulness()`/`generate_from_context()`; `run_eval.py` reads an optional `JUDGE_BACKEND` env var. Verified without real API calls: monkeypatched both backend functions, confirmed default routes to `LLM_BACKEND` unchanged and an explicit override routes correctly. **#4 itself (an actual cross-judge run) is still not done** -- that needs paid OpenRouter calls, separately authorized | 2 |
| 47 | **Router rarely emits pest/disease, so #10 (BM25 fusion) and #43 (entity-match) mostly don't run in production — found 2026-09-25 (live-pipeline eval, see dated section below).** 7 of 8 pest/disease questions were routed to `general`; only disease-2 got `disease`. Both protections key off the *router's* category, and the 2026-09-23/24 #43 verifications mocked the router to the correct category, so they never exercised this path. pest-5 (the reproduced hallucination) was routed `general`, passed the relevance gate, and generated the black-beetle→termite answer again. Fix direction: stop gating these protections on the LLM router's label (CLAUDE.md: an LLM classification must never decide what's reachable/checked) — e.g. run entity-match whenever the retrieved chunks are entity cards, regardless of routed category. Needs re-validation for false positives on fertilizer/general (see the 2026-09-24 scope decision). **FIX IMPLEMENTED 2026-09-25** (`services/entity_match.resolve_entity_category`): hybrid retrieval always runs first; the protections apply if the router says pest/disease OR the top hybrid hit's corpus category is pest/disease — the router can add caution, never remove it. Zero-cost harness (real `get_sugarcane_answer`, router mocked, generation blocked): gate decisions are now identical whether the router says `general` for everything or gives the gold label; pest-5 refused via `entity_mismatch` in both (previously passed the gate when routed `general`). Gate decision correct went from 15/18 to 13/18. Cost, all in the refusal direction: fertilizer-3 (top hit a pest card; relevance 0.456 < 0.50) and general-2 (top hit a disease card; entity_mismatch) newly refused, and pest-2 refused by the 0.50 threshold as documented under #46 (it had only passed because the misroute gave it the 0.35 general threshold). A stricter "≥3 of top-5" rule would remove fertilizer-3 but was deliberately NOT adopted, because it would be tuned on the same 18 questions. Live re-run results: see the dated section below | 1 |
| 48 | **Faithfulness judge returns empty output, scored 0.0 — found 2026-09-25.** 4/18 live questions (pest-1, pest-4, general-1, pest-5) logged `Judge returned non-numeric:` with an empty string. Suspected (NOT verified) cause: gpt-oss-20b is a reasoning model and exhausts its token budget before emitting the score. Three were correct answers wrongly refused; the fourth was pest-5, which was refused **only** because of this failure — the safe outcome on the safety-critical case was luck, not a working control. Fix: confirm cause, then handle judge failure explicitly (retry / distinct `judge_error` state) rather than silently scoring 0.0. fertilizer-1 also scored 0.0 with non-empty output, although its "3 ಕಂತುಗಳು" schedule exists in the chunk (see the 2026-09-24 note) — a judge false negative. **FIXED 2026-09-25, cause verified.** Replayed the exact production judge call on the 18 (context, answer) pairs from the 09-21 run, capturing the raw Groq response: pest-4 returned `finish_reason=length` with 498 of its 500 completion tokens spent on hidden reasoning and empty content. Reasoning use ranged 101–612 tokens across calls, with no clear link to context length, and varies between runs even at temperature 0 (pest-1: 389 then 254), which is why failures looked random. Fix: `services/judge_parse.parse_judge_score` returns None (not 0.0) for empty, unparseable, out-of-range, or multi-number replies; `calculate_faithfulness` tries `max_tokens` 1024, then retries once at 4096 (only generated tokens are billed, so the higher cap costs nothing on normal calls); if both attempts fail it still fails closed at 0.0, but logs a judge error. The old fallback also took the first number anywhere, so "Score: 10" would have passed the gate; now rejected. Verified: 6 unit tests, a mocked retry check, and the same 18-call replay (all `finish_reason=stop` on the first attempt; pest-4 needed 612 reasoning tokens and scored 0.8). fertilizer-1's non-empty 0.0 is a separate judge-quality issue, not fixed here. **Also found: the judge scores pest-5's wrong-pest answer 1.0 (faithful).** The answer accurately quotes the termite chunk, so a faithfulness judge cannot detect a wrong-entity answer. Only the #47 gate fix catches this case | 1 |
| 49 | **Numeric checker misses unit abbreviations → false refusals — found 2026-09-25.** Verified against the retrieved context: disease-2 flagged "10 minute" although the context has "10 min"; general-2 flagged "7 ton" although the context has "(5-7 t)". Both correct answers refused (safe direction, but hurts usefulness). Fix: add `min`→minute and `t`→tonne aliases in `eval/numeric_faithfulness.py` with tests; check a bare `t` alias doesn't mis-bind elsewhere. **FIXED 2026-09-25.** Root cause was broader than missing aliases: Latin-script units had no word boundaries, so single letters matched inside ordinary words ("10 min is crucial" → the `l` of "crucial" bound 10 to litre). Fix: Latin units must stand alone (not inside a word, not after an apostrophe); added `min`/`mins`→minute and lowercase-only `t`→tonne. `t` is case-sensitive because the varietal tables use "T Ha" as a field label ("Ccs Percent: 14.2 Ccs T Ha: 17.5"), and a case-insensitive `t` bound the CCS percentage to tonnes, a binding that could let a wrong answer pass. Checked by diffing old vs new extraction over all 43 chunks and every recorded answer: 31/97 texts extract differently, mostly bogus litre/gram bindings from inside words becoming `nounit`; spot-checked every new tonne binding against the chunk text ("100t of cane", "(5-7 t)", "25 t/ha FYM"). Verdicts changed only for the 09-21 run's disease-2 and general-2, both from flagged to passing. 6 regression tests added. **Still open, found while doing this:** (a) ~~the checker assumes a unit FOLLOWS its number, but the corpus's "Label unit: value" layout ("Duration Months: 10 - 11 Cane Yield T Ha: ...") puts it before~~ — fixed 2026-09-25 (branch `fix/numeric-label-units`): a number directly after a label ending in a unit (`Months:`, `Percent:`, `Tonnes:`, `Days:`, `T Ha:`→tonne) takes that unit, and the two ends of a range (`-`, U+2010–U+2014, `to`) share a unit when only one end has it. Checked every range in the corpus by hand. Across 72 recorded answers from 4 runs, no pass/fail result changed; a first attempt without U+2011 had wrongly failed the 09-21 run's pest-3 ("500‑1000 / acre"), which is now a regression test. Also fixed a crash: `"50.0_g_granule".rsplit("_", 1)` made `float("50.0_g")` fail, so any answer with an unsupported `ಜಿ` number raised an exception; (b) ~~Kannada units are still unbounded~~ — fixed 2026-09-25 (branch `fix/kannada-unit-boundaries`): a Kannada unit can't be preceded by a Kannada letter (suffixes after it, like "ನಿಮಿಷಗಳ", are still allowed), and the shortest (`ಜಿ`, `ಲೀ`) must stand alone on both sides. Real bugs it fixed: `ಜಿ` matched inside ಕಾರ್ಬೆಂಡೈಜಿಮ್ and ಅಟ್ರಾಜಿನ್, binding "50 WP"/"80 WP" to granules. Only those 2 of 115 texts extract differently; no pass/fail change across recorded answers; (c) ~~`run_live_eval.py` stores the refusal message, not the model's pre-refusal answer~~ — fixed 2026-09-25 (commit `f687e9d`): records now include `raw_generation` and the router's raw label. The two 2026-09-25 live runs predate this, so disease-2 is untested against this fix until the next live run | 1 |

### 2026-09-24 — gold-label second-pass verification + fertilizer/general scope decision closed

**Gold-label second AI pass (not human verification — see "Still blocked" #4 below).**
Independently re-derived the correct chunk match for all 18 `questions.json` entries
straight from `chunks.jsonl` content (not from memory of the original `a84ba34` labeling
pass), then re-ran `python eval/check_gold_integrity.py`. Result: **no discrepancies
found** — all 15 answerable entries' `gold_chunk_id`s contain text that matches
`expected_answer` word-for-word (dosages, percentages, names checked individually); the 2
multi-chunk entries (disease-1, general-2) and the shared fertilizer-2/fertilizer-3 chunk
were re-confirmed as genuinely independent/legitimate matches, not padding; all 3
unanswerable entries (`price-1`, `pest-5`, `general-5`) re-confirmed genuinely unanswerable
against a full scan of all 43 chunks; integrity check passes (18/18 resolve, 3 marked
unanswerable). This narrows but does not close "Still blocked" #4 — a second AI pass can
catch labeling/typo errors but cannot catch a case where the corpus itself has a wrong
agricultural fact, which only a domain expert can.

**Gold-label third AI pass, independent model (Claude Opus, no shared context with the
above passes).** At the project owner's explicit request, a separate Opus agent re-derived
each of the 18 gold labels from scratch against `chunks.jsonl`/`questions.json`, with no
visibility into the Sonnet passes above. Result: **18/18 confirmed correct, 0
discrepancies** -- verified exact dosages/numbers/names quoted from the chunk text (not
just topic similarity) for every answerable entry, and independently re-swept all 43
chunks for the 3 unanswerable claims. Three independent AI passes (original labeling +
Sonnet audit + Opus audit) now agree with no disagreement found. Same caveat as above,
stated explicitly by the Opus agent itself: this verifies label-to-corpus fidelity, not
whether the corpus's own dosages are agriculturally correct -- human domain-expert sign-off
is still required before publication, and provenance is still unknown (#8).

**Zero-cost live verification of the pest-5 gate refusal.** Ran the actual production
`get_sugarcane_answer()` pipeline against `pest-5` with the LLM router call mocked to a
fixed "pest" classification (its true category, already known from `questions.json`) and a
hard guard that raises instead of making any further LLM call -- so this cannot spend money
even if the gate fails to refuse. Result: gate refuses via `entity_mismatch` before any
generation call, relevance=0.5972 (matches the previously-documented number), confirming
TODO #43's claim end-to-end against the live code path today, not just the isolated module
test from 2026-09-23. Also spot-checked `fertilizer-1`'s already-generated eval answer (no
new call): the model answered "3 installments (6th/10th/14th week)" -- a second schedule
that genuinely exists in the `f69c5f65` chunk, faithfulness=1.0 -- corroborating the
existing "multi-answer ambiguity, not a hallucination" note rather than a new defect.

**Fertilizer/general BM25-fusion + entity-match scope — decision: leave scoped to
pest/disease, do not widen.** This was an open question from TODO #10/#43. Checked
`entity_match.py`'s `_NAME_FIELD_RE` (`"Name: X (Y) Recommendations:"`) against every
fertilizer/general chunk in `chunks.jsonl`: none have that field (they're `Nutrient
Management`/`Planting Practices`/etc. structured text, not entity cards). So widening the
hook's scope to those categories would be a permanent no-op on the *correct* chunk for a
query, and could only ever fire as a **false positive** if a spurious pest/disease chunk
leaks into a fertilizer/general query's top-5 — which is exactly the failure mode the
original 6/18 false-positive test (docstring, `entity_match.py`) already caught before
BM25 fusion landed, and that test was never re-run for these categories. Combined with the
existing ablation (#10) showing BM25 fusion actively *hurts* the semantic bucket, there is
no upside to widening scope today and a demonstrated downside. Closing this out as a
resolved decision, not a someday item — reopen only if fertilizer/general retrieval
quality is separately re-ablated and a real hallucination case is reproduced there (same
evidence bar #43 met for pest/disease).

### 2026-09-25 — first live-pipeline eval (all 18 questions through `get_sugarcane_answer`)

At the project owner's explicit go-ahead (paid Groq calls). New script
`backend/eval/run_live_eval.py`: unlike `run_eval.py` (frozen `contexts.json`, straight
into generation, so no router, BM25 fusion, gate, or entity-match), this sends each question
through the production function and records the gate decision per question. Run file:
`backend/eval/runs/live/openai-gpt-oss-20b__3572a0f3ad5c__20260925T212719.jsonl`.
Config: `GENERATION_MODEL=openai/gpt-oss-20b` (self-judged), Groq, file-mode Qdrant
(43 chunks), reranker off, `ENABLE_LENIENT_PEST_DISEASE_THRESHOLD=false` (it had been
`true` in the local `.env`, contradicting the documented default; set to `false` at the
owner's direction).

Result: correct answer/refuse decision on 12/18 — 3/3 unanswerable refused, 9/15
answerable answered. **The headline number overstates safety**: pest-5 was refused only
because the judge returned empty output (#48); the gate did not catch it, because the router
sent it to `general` and entity-match never ran (#47). All 6 wrongly refused answerable
questions trace to #48 (pest-1, pest-4, general-1), #49 (disease-2, general-2), or a judge
false negative (fertilizer-1). Answers that got through were judged ≥0.95 but have not been
human-reviewed. chrF/embedding similarity are in the file for continuity only — not quality
or safety evidence.

**Re-run after the #47 fix (commit `c2c6e85`), same config.** Run file:
`backend/eval/runs/live/openai-gpt-oss-20b__3572a0f3ad5c__20260925T215356.jsonl`. Correct
answer/refuse decision on **14/18** (was 12/18) — 3/3 unanswerable refused, 11/15 answerable
answered. **pest-5 is now refused by the gate (`entity_mismatch`)** before any generation
call, although the router again said `general` — no longer dependent on a judge failure.
The router again labelled 7 of 8 pest/disease questions `general` (visible in the
"Router said ..." log lines; the run file's `gate.category` is the category *after* the
fix, not the raw router output). The 4 wrongly refused answerable questions: disease-2
(#49 numeric "10 min"), pest-2 (0.50 threshold, #46), fertilizer-3 and general-2 (#47's
known cost). The judge returned no empty outputs this run, so #48 is still open and
unexplained — this run just didn't trigger it. Answers that got through were judged ≥0.95
but have not been human-reviewed.

**Third run, with #47 + #48 + #49 fixes and raw-answer recording (branch
`fix/judge-empty-output`), same config.** Run file:
`backend/eval/runs/live/openai-gpt-oss-20b__3572a0f3ad5c__20260925T222329.jsonl`. Correct
answer/refuse decision on **15/18** (was 12/18 this morning, 14/18 after #47) — 3/3
unanswerable refused, 12/15 answerable answered. **#49 confirmed live:** disease-2's
answer ("10 ನಿಮಿಷ … 0.1 %") now passes the numeric check (score 1.0) and is answered.
**#48:** no judge errors logged; every answer that reached the judge got a usable
score. pest-5 again refused by the gate (`entity_mismatch`) with the router saying
`general`. The router again said `general` for 7 of 8 pest/disease questions. All 3
remaining wrongly refused answerable questions are gate decisions, not judge or numeric
failures: pest-2 (0.50 threshold, #46), fertilizer-3 and general-2 (#47's known cost).
Answers that got through were judged ≥0.85 but have not been human-reviewed.

---

## Still blocked (updated 2026-09-24)

1. **Paid LLM/API calls** — partially unblocked this session at the project owner's
   explicit direction: generation eval (#6/#2, chrF/embsim, see "Real evaluation results")
   and threshold sweep (#7) ran for real. Still not done: manual failure-mode review (#11,
   needs live outputs + human judgment) and OpenRouter cross-judge (#4 -- and per new item
   #45, this is now blocked on an architecture gap too, not just an API key/authorization).
2. **CI/secrets setup** — GitHub Action for `refusal_test.py` (#12). Unchanged: a CI/CD
   pipeline change needs sign-off, and can't actually pass without Groq/OpenRouter keys and
   a live Qdrant/Mongo configured as GitHub secrets.
3. **The chunking rebuild** (#13) — unchanged. Larger scope (Indic NLP preprocessing +
   field-aware chunking), forces relabeling `gold.jsonl` afterward again. Deliberately
   deferred as its own conversation, not a quick pipeline item.
4. **Human gold-labeling** — **partially unblocked this session, with an important
   caveat.** At the project owner's explicit direction, gold.jsonl (commit `a84ba34`) was
   generated by an AI reading actual chunk content against each question (not the
   circular `auto_relabel_gold.py` heuristic) -- full methodology and reasoning are in that
   commit's message. **Re-verified with two more independent AI passes 2026-09-24**
   (a Sonnet audit, then a separate Opus agent with no shared context -- see dated
   sections above) -- both found zero discrepancies, integrity check passes, three
   independent AI passes now agree. This is still NOT independent human domain-expert
   verification, and no number of AI passes can become one -- they can catch a
   labeling/typo error but not a case where the corpus itself states a wrong agricultural
   fact. Before these labels go in
   a paper or report, a human (ideally with agricultural domain knowledge) should
   independently review them, especially the 2 added negative examples (`pest-5`,
   `general-5`).
5. **Corpus provenance** (#8) — unchanged, genuinely unknown where `sugarcanemerged3.json`
   came from. Do not fabricate a source, date, or collection method to close this item.

---

## Track 1 — 2-week plan (demo-ready, defensible)

**Week 1 — critical path**
1. Fix Groq deprecation (#1) — blocks everything else
2. Re-run eval, annotate stale results (#2)
3. BLEU self-score diagnostic (#3)
4. OpenRouter cross-judge branch (#4)
5. Numeric faithfulness checker + demo examples (#5)
6. `.env.example` (#6) — ten minutes, do it same day as #1

**Week 2 — hardening + polish**
7. Threshold sweep against gold labels (#7)
8. Data provenance note (#8)
9. Docker Compose + preflight script (#9)
10. BM25 bucketed evaluation (#10)
11. Manual failure-mode review (#11)
12. GitHub Action for refusal test (#12)
13. Fix or remove dead `tavily_client` (#37)
14. Document temperature=0.0 as a design choice (#39)
15. PII/prompt-injection limitation note (#41)
16. Production Readiness section in the report (#42)
17. Start chunking rebuild if time allows (#13) — better to leave clean for Track 2 than rush it
18. Category payload filtering if time remains (#38)

*Cut first if squeezed: streaming (#15), query condensation (#14), category filtering (#38).*

## Track 2 — month before paper

**Weeks 1–2 — reframing + evaluation depth**
- Decide paper thesis (#16) — do this first, it shapes everything after
- No-retrieval baseline (#17)
- Finish chunking rebuild if not done in Track 1 (#13)
- Query condensation, properly evaluated (#14)
- Expand eval set via KCC, prioritizing independent sourcing (#18, #19)

**Weeks 2–3 — rigor**
- BERTScore + RAGAS adoption (#20, #21)
- Statistical significance testing (#22)
- Human eval with documented process (#23)
- Restore Ollama, resolve sarvam (#28, #29)
- Structured extraction for dosage fields, upgrading the regex-based numeric checker (#40)
- Category payload filtering if not done in Track 1 (#38)

**Week 4 — packaging**
- Reproducibility package (#24)
- Data licensing check (#25) — before finalizing eval set, not after
- Venue selection and formatting (#26)
- Model versions/dates + Groq deprecation as citable lesson (#27)
- Unit tests, LICENSE, lockfile if time allows (#30, #31)
- Future-work section: voice, multi-crop, WhatsApp, image diagnosis, live data (#32–36) —
  write about them, don't build them

  | # | Item                                                        | Why It's Future Work                                                               | Effort if Pursued                             |
| - | ----------------------------------------------------------- | ---------------------------------------------------------------------------------- | --------------------------------------------- |
| 1 | **Multi-crop expansion** (ragi, tomato, paddy)              | Single-crop RAG is a toy example; multi-crop shows generalization                  | 2–3 weeks (data collection + re-labeling)     |
| 2 | **Real farmer queries from Kisan Call Centre (KCC)**        | Synthetic questions are too clean; real queries have typos, code-mixing, vagueness | 1–2 weeks (dataset access + cleaning)         |
| 3 | **Temporal metadata** (season, crop stage, validity period) | "How much fertilizer?" depends on germination vs maturity stage                    | 1 week (schema design + data annotation)      |
| 4 | **Regional metadata** (soil type, district, climate zone)   | North Karnataka red soil vs coastal black soil needs different advice              | 1 week (geotagging existing data)             |
| 5 | **Structured + unstructured mixed corpus**                  | Tables (dosages) + paragraphs (advisory) + images (symptoms)                       | 2–3 weeks (OCR + image captioning)            |
| 6 | **Multiple acceptable answers per question**                | Chemical vs biological control — both valid, eval should accept either             | 2–3 days (schema + code update)               |
| 7 | **Human evaluation panel**                                  | Farmer + agronomist ratings of answer quality and safety                           | 1–2 weeks (recruitment + annotation)          |
| 8 | **Active learning from farmer feedback**                    | Thumbs up/down on answers to improve retrieval over time                           | 2–3 weeks (feedback UI + retraining pipeline) |

