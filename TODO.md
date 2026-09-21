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
- Gold-labeled retrieval ablation with recall@k / MRR / nDCG@5, 95% CI, n=15 answerable
  questions — **caveat added 2026-09-21**: this ran through `normalize_kannada()` (both
  corpus chunking and query preprocessing), which had a whitespace-collapse bug active at
  the time (see this session's entries below). The reranker conclusion below is unaffected
  in kind (it's a corpus-size-ceiling effect, not a normalization artifact) but the exact
  numbers predate the normalization fix and should be re-run before being cited precisely.
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
| 7 | Threshold sweep — **tooling fixed** 2026-09-21 (was scoring the wrong signal, see Already done), **not yet run** — blocked on Phase 2 human gold labeling | 1 |
| 8 | Data provenance note for `sugarcanemerged3.json` — **still open, correctly**: provenance is genuinely unknown, do not fabricate a source/date | 1 |
| 9 | ~~Docker Compose + preflight script~~ — ✅ **DONE** (see Already done) | 1 |
| 10 | BM25 evaluation, bucketed by query type — **ablation tooling ready** (`run_retrieval_ablation.py` already has bm25/bm25+dense/bm25+hybrid configs + manual-label bucketing), **not yet run** — blocked on Phase 2 human gold labeling | 1 |
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

---

## Still blocked (unchanged by this session)

Five categories of work remain genuinely blocked, not just unprioritized. Listed
explicitly so they don't get silently dropped or mistaken for "not needed":

1. **Paid LLM/API calls** — re-running eval on the current model (#2), the OpenRouter
   cross-judge run (#4, code exists, never executed), manual failure-mode review (#11,
   needs live outputs). CLAUDE.md forbids these by default; needs explicit authorization.
2. **CI/secrets setup** — GitHub Action for `refusal_test.py` (#12). A CI/CD pipeline
   change needs sign-off, and can't actually pass without Groq/OpenRouter keys and a live
   Qdrant/Mongo configured as GitHub secrets.
3. **The chunking rebuild** (#13) — larger scope (Indic NLP preprocessing + field-aware
   chunking), forces relabeling `gold.jsonl` afterward. Deliberately deferred as its own
   conversation, not a quick pipeline item.
4. **Human gold-labeling** (Phase 2, blocks #7, #10) — `auto_relabel_gold.py` can draft
   candidates (`gold.jsonl.draft`) but CLAUDE.md explicitly reserves the actual labeling
   decision as a human task, not something to automate and present as verified.
5. **Corpus provenance** (#8) — genuinely unknown where `sugarcanemerged3.json` came from.
   Do not fabricate a source, date, or collection method to close this item.

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

