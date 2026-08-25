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
  replaced with chrF (validated: self-score 1.0, unrelated-pair floor 0.172)
- Gold-labeled retrieval ablation with recall@k / MRR / nDCG@5, 95% CI, n=15 answerable
  questions
- Reranker decision made and justified (off by default; near-ceiling recall at current
  corpus size; MRR/nDCG gain real but corpus-size-dependent, must re-benchmark on scale-up)
- Hard refusal + faithfulness gate + escalation message live in `rag_service.py`
- Duplicate `refusal_test.py` cleaned up

---

## Ranked master list

| # | Task | Track |
|---|---|---|
| 1 | Fix Groq deprecation: switch to `openai/gpt-oss-20b`, remove 4 hardcoded fallback model strings in `llm_client.py` | 1 |
| 2 | Re-run eval against new model; annotate old `results.jsonl` as predating the deprecation | 1 |
| 3 | Run BLEU through the self-score diagnostic, bring the result to the professor (he asked for BLEU by name — show him why it fails, don't just assert it) | 1 |
| 4 | Add OpenRouter backend branch in `llm_client.py`; use as **cross-judge** model (different provider judging a Groq-generated answer, closing the self-evaluation gap) | 1 |
| 5 | Numeric faithfulness checker (regex cross-reference of dosage/quantity numbers in the answer against retrieved context) + 2–3 adversarial demo examples | 1 |
| 6 | Add `.env.example` with placeholder values and inline comments for every env var (`GROQ_API_KEY`, `OPENROUTER_API_KEY`, `GENERATION_MODEL`, `ENABLE_RERANKER`, `LLM_BACKEND`, etc.) | 1 |
| 7 | Threshold sweep: use `gold.jsonl` to empirically justify `HARD_REFUSAL_THRESHOLD` (0.35) and `RERANK_THRESHOLD` (0.5) instead of leaving them as unexplained constants | 1 |
| 8 | Data provenance note for `sugarcanemerged3.json` — one paragraph in README on source/date/collection method | 1 |
| 9 | Docker Compose (MongoDB + Qdrant only, not full stack) + a preflight script (curl Qdrant collection, confirm chunk count, confirm Mongo reachable) — demo-day insurance | 1 |
| 10 | BM25 evaluation — **bucketed by query type** (exact-term chemical/dosage queries vs. semantic/general queries), not a flat 4th ablation row. If it wins on exact-term queries, wire it into the production hybrid fusion, not just the report | 1 |
| 11 | Manual failure-mode review on existing eval results (never done — read ~20 outputs, tag failure categories) | 1 |
| 12 | GitHub Action running `refusal_test.py` on push | 1 |
| 13 | Chunking rebuild: Indic NLP Library preprocessing (agglutination handling) + field-aware chunking, bundled as one pass — relabel `gold.jsonl` and re-run the ablation once, not twice | 1 (start) / 2 (finish) |
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
| 30 | Basic unit tests beyond `refusal_test.py` (chunking boundaries, `flatten_value()`, UUID5 determinism across reseeds) | 2 (if time) |
| 31 | Add a LICENSE file and a `pip freeze`-pinned lockfile for full reproducibility | 2 (if time) |
| 32 | Voice input/output (STT/TTS) | 2 — future work section only, not built |
| 33 | Multi-crop expansion (ragi, tomato) + re-run the retrieval ablation, since the reranker conclusion is corpus-size-dependent and may flip | 2 — future work section only |
| 34 | WhatsApp channel | 2 — future work section only |
| 35 | Image-based pest/disease diagnosis from photo | 2 — future work section only |
| 36 | Live tool-calling integration (Agmarknet mandi prices, IMD weather) | 2 — future work section only |
| 37 | **Web fallback is dead code, not just disabled** — `tavily_client` is instantiated in `rag_service.py` when `ENABLE_WEB_FALLBACK=true`, but no `.search()` call exists anywhere in the file. Same "built but not wired in" bug as `llm_client.py`/`ChatMessage.tsx` before those were fixed. Either wire in a real call (e.g. price queries when retrieval confidence is low) or remove the dead client instantiation for hygiene | 1 |
| 38 | Wire Qdrant payload filtering by category — the router already classifies queries (disease/pest/fertilizer/general) but this is never used to filter/boost retrieval, only vector similarity is. Another instance of the "built but not wired in" pattern | 1 (if time) / 2 |
| 39 | Document `temperature=0.0` as an explicit design choice in the report (deterministic outputs, reduces eval variance, intentional trade-off vs. natural phrasing) — one paragraph | 1 |
| 40 | Structured extraction for safety-critical numeric fields (chemical name/dose/unit as a validated schema) rather than free-text generation — a stronger version of the numeric faithfulness checker (#5); note as an extension/upgrade path even if #5 ships first | 2 |
| 41 | One-line report mention of PII/prompt-injection handling as a named, deliberately out-of-scope limitation | 1 |
| 42 | Add a **"Production Readiness"** section to the report explicitly naming known, deliberately-deferred ops gaps (auth, rate limiting, structured logging beyond `print()`, CI/CD) — shows awareness of the prototype/production delta rather than leaving it unaddressed | 1 |

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
