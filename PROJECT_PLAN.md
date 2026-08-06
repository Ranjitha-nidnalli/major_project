# Kannada Agricultural RAG — Work Order & Target Outcome

Handoff doc for Claude Code. Read this fully before making changes.
Work the tasks **in order**. P0 blocks everything downstream.

---

## 0. Context: where the project stands

**Built and working:**
- FastAPI backend, Next.js frontend, MongoDB chat memory
- Qdrant vector store, BGE-M3 embeddings (dense + sparse)
- Hybrid retrieval via Qdrant native `FusionQuery(fusion=Fusion.RRF)`
- `bge-reranker-v2-m3` cross-encoder rerank stage
- All LLM calls local via **Ollama** (`GENERATION_MODEL`, default `gemma4:e4b`)
- Tavily web fallback behind `ENABLE_WEB_FALLBACK` (default off)
- Eval harness: `backend/eval/questions.jsonl` (n=16), `run_eval.py`, `run_retrieval_ablation.py`
- Corpus: single crop (sugarcane), `sugarcanemerged3.json`

**The blocking problem:** the retrieval ablation returned ROUGE-L ≈ 0.011 across *all six* configs
(dense / sparse / hybrid, each ± reranker) and embedding-similarity in a 0.525–0.565 band.

This was initially read as "the reranker adds nothing." **That conclusion is not supported.**
A metric that returns the same near-zero value regardless of input is not measuring quality.
The correct reading is: **our metrics currently lack the sensitivity to detect any difference
between retrieval configs.** Everything else is downstream of fixing that.

Do not write any of the current ablation numbers into the report until P0 and P1 are done.

---

## P0 — Metric sanity (do this first, it is quick)

### P0.1 Prove the metrics are broken

Create `backend/eval/diagnose_metrics.py`. It must:

1. Load references from `questions.jsonl`.
2. Score **each reference against itself** with the current ROUGE-L function.
   - Expected if healthy: ~1.0
   - If it returns ~0.0 → ROUGE is discarding Kannada tokens. Confirmed broken.
3. Score each reference against a **different, unrelated** reference (shuffle the list).
   - This gives the metric's *noise floor*.
4. Do the same two checks for BGE-M3 embedding-similarity.
   - Expect self-sim ≈ 1.0, and cross-sim likely ≈ 0.5 — i.e. two unrelated Kannada
     agri sentences already score ~0.5 purely from shared script/language/domain.
5. Print a table: `metric | self-score | unrelated-score | usable range`.

**Why:** if the unrelated-pair baseline is ~0.5 and all our configs land in 0.525–0.565,
we were reading noise inside the metric's floor. This script proves it in one run.

### P0.2 Replace ROUGE-L with script-appropriate metrics

ROUGE-L word-tokenises. Kannada is **agglutinative** — ಕಬ್ಬಿಗೆ / ಕಬ್ಬಿನ / ಕಬ್ಬು share a stem but
never match as whole tokens. Word-level overlap metrics are structurally wrong here.

Replace with **character-n-gram metrics**:
- **chrF / chrF++** via `sacrebleu` — this is the primary metric from now on.
- Optionally char-level BLEU: `sacrebleu` with `tokenize='char'`.
- Keep BGE-M3 embedding-sim, but **always report it against the unrelated-pair baseline
  from P0.1**, never as a bare number.

Update `run_eval.py` and `run_retrieval_ablation.py` to emit chrF. Keep ROUGE-L in the output
as a deliberately-retained broken metric **only if** you also log the P0.1 self-score next to it
— it makes a good "why we changed metrics" figure in the report. Otherwise drop it.

### P0.3 Investigate the 30s reranker latency

30s/query for `bge-reranker-v2-m3` on ~30 pairs is slow even on CPU. Before we conclude
anything about its cost/benefit, rule out that the latency is an artifact:

- Is the reranker model **loaded on every query** instead of once at module import / app startup?
- Are pairs scored **one at a time** instead of in a single batched `.compute_score()` call?
- Is `use_fp16` set appropriately for the device?

Fix both if present. Re-time. Plausible target: 2–5s. If it drops that far, the entire
cost/benefit trade-off changes and must be re-evaluated, not assumed.

---

## P1 — Evaluate retrieval with *retrieval* metrics

The core methodological error: we evaluated **retrieval** using **generation** metrics
(ROUGE/emb-sim on retrieved text vs. reference answers). Retrieval needs retrieval metrics,
and those need gold labels.

### P1.1 Build gold chunk labels

The corpus is small (tens of chunks) and n=16. This is ~1 hour of manual work and it is the
single highest-value hour left in the project.

1. Dump every chunk with a stable id: `backend/eval/dump_chunks.py` → `chunks.jsonl`
   (`{chunk_id, category, text}`). **Chunk ids must be stable across rebuilds** — derive
   from a content hash, not `uuid4()`, so labels survive re-seeding. (`vector_db.py` currently
   uses `uuid.uuid4()` — change it to a deterministic hash of the chunk text.)
2. Create `backend/eval/gold.jsonl`: for each question id, a list of chunk_ids that genuinely
   contain the answer. Allow multiple gold chunks per question.
3. If a question has **no** answering chunk in the corpus, label it `gold: []` and mark it
   `unanswerable: true`. These are valuable — they test whether the system correctly refuses.

### P1.2 Score retrieval properly

Rewrite `run_retrieval_ablation.py` to compute, per config (dense / sparse / hybrid, ± rerank):

- **recall@k** for k = 1, 3, 5, 10
- **MRR**
- **nDCG@5**
- latency (mean + p95)
- report **n and variance/CI**, not just point estimates. With n=16, a bare mean is not a result.

**Expected finding — anticipate it:** recall@5 will likely be at or near **1.0** for the
no-reranker baseline. The corpus is one crop; first-stage retrieval over ~tens of chunks
almost always surfaces the right one in the top 15. If so, the reranker *cannot* show a gain
— the metric is **saturated**.

That is the real result, and it is a good one. Frame it as:

> **No headroom, not no benefit.** On a single-crop corpus, hybrid first-stage retrieval
> already achieves recall@5 ≈ 1.0, leaving nothing for a cross-encoder reranker to fix.
> Reranking is an *ordering* intervention; MRR/nDCG are the metrics that can detect it,
> and even those are near-ceiling at this corpus size.

MRR and nDCG@5 may still show a small ordering improvement where recall@k is flat — that is
exactly what those metrics are for. Report them.

### P1.3 Decide the reranker on evidence

Put the reranker behind `ENABLE_RERANKER` (**default off**). Keep the code path alive.

Report language:

> Cross-encoder reranking was implemented and evaluated. On the current single-crop corpus it
> produced no measurable retrieval gain (recall@5 already at ceiling; ΔMRR within noise) at
> Nx latency cost. It is disabled by default and retained behind a feature flag. **This
> conclusion is corpus-dependent and must be re-benchmarked when the corpus scales to multiple
> crops**, where first-stage recall will no longer be saturated.

Latency is not an academic concern here: the end user is a farmer on patchy rural 4G. Say so.

---

## P2 — The 3-model generation comparison

Models: `gemma4:e4b`, `llama3.1:8b`, `gaganyatri/sarvam-2b-v0.5`.

**Do not run retrieval 3×N times.** In a generation comparison, retrieval is a *controlled
variable*. Freeze it.

### P2.1 Freeze contexts

`backend/eval/build_contexts.py`:
- Run retrieval **once** per eval question, in the shipping config.
- Cache to `contexts.json`: `{question_id: {query, retrieved_chunk_ids, context_text}}`.

### P2.2 Replay

Modify `run_eval.py` to read `contexts.json` instead of calling retrieval. Feed **byte-identical**
context to all three models.

Benefits: differences are attributable purely to generation; retrieval cost paid once; the run
is reproducible and a 4th model can be added later without retrieval drift; no Ollama-warmup
latency artifact contaminating the comparison. Hours → minutes.

### P2.3 Judge

Current `calculate_faithfulness` uses the **same local model** that generates. An 8B model
grading its own output is weak evidence (known self-preference bias). Mitigate:

- Use a **different** local model as judge than the one being graded (e.g. judge gemma with
  llama and vice versa), and say so in the report.
- **Human evaluation on a subset (~30–50 answers).** Get an agriculture student, a KVK officer,
  or an agri-science friend to score faithfulness + usefulness on a 1–5 scale. This is the single
  thing that turns the evaluation chapter from "we asked an LLM if it was happy" into real
  science. It does not need to be large; it needs to exist.

### P2.4 Manual failure-mode review

Read ~20 outputs by hand. Categorise failures (retrieval miss / hallucination / bad Kannada /
wrong register / refused-when-it-shouldn't / answered-when-it-shouldn't). A tagged failure
taxonomy with examples is a strong report section and costs an evening.

---

## P3 — Correctness & safety layer (high marks, low effort)

This is an agriculture bot. A hallucinated **pesticide dosage** can destroy a crop, poison a
farmer, or breach pesticide regulation. Examiners will probe this. Implement:

1. **Escalation line on every answer**: Kisan Call Centre `1800-180-1551` / nearest Raitha
   Samparka Kendra. Append in Kannada.
2. **Hard refuse on low-confidence chemical/dosage queries.** Currently a low-confidence query
   falls through to web search. For `pest` / `disease` / `fertilizer` categories involving a
   dosage or a chemical name, a low retrieval score must **refuse**, not improvise. Refusal is
   the correct behaviour; the existing Kannada refusal string is already there.
3. **Timestamp** all market-price and weather data in the answer. Stale prices are worse than
   no prices.
4. Add the `unanswerable: true` eval questions from P1.1 as an explicit **refusal-accuracy**
   metric. "Correctly refuses when the corpus doesn't know" is a reportable number.
5. Units: answers should use **acres / guntas / quintals / ₹ per kg** — not hectares/tonnes.
   Check what the corpus uses and what the model emits; add to the system prompt if needed.

---

## P4 — Scope upgrades (mini-project → final-year project)

In descending order of impact:

1. **Voice input.** The biggest real-world adoption factor — typing Kannada on a phone is
   painful, and many farmers are more comfortable speaking. ASR options: AI4Bharat
   **IndicConformer** / IndicWhisper (tuned for Indian accents and noisy audio) or Whisper
   large-v3. TTS: AI4Bharat Indic-TTS. Even a mic button in the existing chat UI is a large win.
2. **Real farmer queries.** Our 16 eval questions are clean textbook Kannada. Real queries are
   **code-mixed** — "tomato-ge yaava spray madbeku, leaf curl bandide" — with English pesticide
   and brand names in Kannada grammar. Mine the **Kisan Call Centre dataset** (data.gov.in,
   filter Karnataka) for 100–200 real questions. Testing on realistic input is where the true
   failure modes are. This also fixes the n=16 problem.
3. **Crops 2 and 3.** Ragi + tomato alongside sugarcane covers dryland / vegetable / plantation.
   The ingestion code is already crop-agnostic; this is mostly data + prompt work. **Re-run the
   P1 ablation after this** — the reranker conclusion may well flip once recall is off the ceiling.
4. **WhatsApp channel.** Meets farmers where they already are; no app install.
5. Live tool-calling: Agmarknet mandi prices, IMD weather.
6. Image-based pest diagnosis from a leaf photo.

Note `get_sugarcane_answer()` and the hardcoded `"sugarcane {query} Karnataka"` Tavily string
are single-crop assumptions baked into the code. Generalise when you do (3).

---

## Target outcome — what "done" looks like

**Artifact:** a voice-capable Kannada agricultural Q&A system, running fully on local models,
grounded in authoritative Karnataka sources (UAS Package of Practices), covering 2–3 crops,
that refuses rather than guesses on safety-critical questions.

**Report:** the artifact is ~40% of the marks. The other 60% is the evaluation:

- A **metric-validity section** — "we discovered our first-choice metrics were structurally
  unsuited to an agglutinative Dravidian script, diagnosed it, and replaced them." Most student
  projects never notice this. It is a genuine contribution and it is *already yours*.
- A **retrieval ablation table** (dense/sparse/hybrid ± rerank) on proper IR metrics, with n and
  CIs, and the ceiling-effect analysis.
- A **generation comparison** across 3 local models on frozen contexts, with a human-eval subset.
- A **failure taxonomy** with real examples.
- A **responsible-AI section** on dosage safety and refusal behaviour.

**The framing that makes this strong:** the honest, measured negative results *are* the
contribution. "We built a chatbot and it works" is a C. "We built it, measured it rigorously,
found our metrics were lying to us, fixed them, and can now say precisely when a reranker earns
its cost on an Indic-language corpus" is a distinction.

---

## Open questions — answer these and the plan gets sharper

1. **How many chunks are in the Qdrant collection?** (Post-fix, after the stale-42/clean-40 bug.)
   This determines whether the recall-ceiling hypothesis is right.
2. **What's in `questions.jsonl`?** Are the references gold *answers* (free text) or gold
   *chunks*? Where did the 16 questions come from — hand-written, or drawn from the corpus?
   If they were written *from* the corpus, they're easier than real queries and the ceiling
   effect is even more likely.
3. **What exactly does the current ROUGE-L call look like** (which library, which tokenizer)?
4. **CPU-only, or is there a GPU available?** Changes the reranker verdict and the ASR plan.
5. **What's the deadline and how many people are on the team?** Drives how much of P4 is realistic.
