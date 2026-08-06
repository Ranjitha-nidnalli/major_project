# Krishi Mitra

A Kannada-language RAG chatbot that answers sugarcane farming questions for Karnataka farmers. FastAPI backend, Next.js frontend, Qdrant hybrid (dense + sparse) vector store, BGE-M3 embeddings, a BGE cross-encoder reranker, and MongoDB for chat history. All generation runs on a local Ollama model — **no hosted LLM API keys required**.

See [CLAUDE.md](./CLAUDE.md) for a deeper architecture walkthrough (file-by-file responsibilities, request flow).

## Architecture

```
frontend/   Next.js chat UI  ──HTTP──>  backend/   FastAPI
                                          ├── rag_service.py  retrieval + generation pipeline
                                          ├── vector_db.py    Qdrant client, BGE-M3 embed model,
                                          │                   BGE reranker model, + DB seeding script
                                          ├── chat_db.py      MongoDB (Motor) chat history
                                          └── main.py         /chat /history /health endpoints

Qdrant (local, file-mode)   MongoDB (localhost, no auth)   Ollama (localhost:11434)
```

**Request flow**: query → LLM router (category + English gloss) → BGE-M3 hybrid retrieval (dense + sparse, fused with Reciprocal Rank Fusion) → query-expansion retry on weak matches → BGE cross-encoder reranks/filters candidates → optional Tavily web fallback (off by default) → Ollama generates a Kannada answer strictly from context → Ollama judges faithfulness → answer + `search_score` + `accuracy_score` saved to Mongo and returned.

## Prerequisites

- Python 3.10+
- Node.js 18+
- [Ollama](https://ollama.com) installed and running
- MongoDB running locally (no auth needed)

## Setup

### 1. Pull the Ollama model

```bash
ollama pull gemma4:e4b
```

Ollama must be running (`ollama serve`, or the desktop app) and reachable at `http://localhost:11434` before you start the backend.

### 2. Start MongoDB locally

```bash
mongod --dbpath <your-data-dir>
```

No credentials needed — the backend connects to `mongodb://localhost:27017` by default with no auth.

### 3. Backend environment

Create `backend/.env` (gitignored). All variables are optional — sensible local defaults are baked in:

| Variable | Default | Purpose |
|---|---|---|
| `MONGO_URI` | `mongodb://localhost:27017` | MongoDB connection string |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server address |
| `GENERATION_MODEL` | `gemma4:e4b` | Ollama model used for routing, query expansion, generation, and the faithfulness judge — swap it out to A/B test other local models without touching code |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | HF model id (or local path) for dense+sparse embeddings |
| `RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | HF model id for the cross-encoder reranker |
| `ENABLE_RERANKER` | `false` | Set to `true` to rerank retrieved chunks with the cross-encoder. Off by default — see [Evaluation](#evaluation) for why |
| `ENABLE_WEB_FALLBACK` | `false` | Set to `true` to enable Tavily web search fallback for price/low-confidence queries |
| `TAVILY_API_KEY` | — | Only needed if `ENABLE_WEB_FALLBACK=true` |

### 4. Install backend dependencies

```bash
cd backend
pip install -r requirements.txt
```

First run will download the BGE-M3 embedding model and the BGE reranker model from Hugging Face (a few GB total) — this can take a while.

### 5. Seed the Qdrant database

Qdrant runs in local file-mode (no separate server) at `backend/qdrant_sugarcane_db/`. Place your source knowledge file at `backend/sugarcanemerged3.json` (gitignored — not included in the repo), then:

```bash
cd backend
python vector_db.py
```

This chunks the JSON, generates dense + sparse embeddings for each chunk, and upserts them into the `sugarcane_knowledge` Qdrant collection. Re-run it any time you update the source data — it clears and rebuilds the collection.

Qdrant's file-mode store only allows one process to hold it at a time. If you see a "DB SERVER IS LOCKED" error, stop `main.py` (or any other process using it) first; `python unlock_db.py` can kill zombie holders.

### 6. Run the backend

```bash
cd backend
python main.py
```

Serves on `http://localhost:8000`. Endpoints: `POST /chat`, `GET /history/{session_id}`, `GET /health`.

### 7. Run the frontend

```bash
cd frontend
npm install
npm run dev
```

Serves on `http://localhost:3000` and talks to the backend at `http://localhost:8000`.

## Evaluation

All eval code and data lives in `backend/eval/`. `questions.json` (n=16, one deliberately unanswerable) and `gold.jsonl` (gold chunk ids per question) are hand-built from `sugarcanemerged3.json`, so this is a first pass, not a held-out test set — real farmer queries would be harder.

### Metric validity: our first-choice metrics were lying to us

We initially scored retrieved/generated text against gold answers with ROUGE-L and BGE-M3 embedding cosine similarity. `backend/eval/diagnose_metrics.py` checks any metric before trusting it, by scoring each reference against itself (expect ≈1.0) and against an unrelated reference (the metric's noise floor):

| Metric | self-score | unrelated-score | usable range |
|---|---|---|---|
| ROUGE-L | 0.625 | 0.048 | 0.577 |
| chrF | 1.000 | 0.172 | 0.828 |
| embedding-sim | 1.000 | 0.524 | 0.476 |

ROUGE-L isn't just insensitive — it's broken for this corpus. Its tokenizer strips non-Latin script, so 6 of 16 references (pure-Kannada sentences with no Latin substrings) scored **0.000 against themselves**. Kannada is also agglutinative (ಕಬ್ಬಿಗೆ / ಕಬ್ಬಿನ / ಕಬ್ಬು share a stem but never match as whole word tokens), which independently breaks word-level overlap metrics. We replaced it with **chrF** (character-n-gram, `sacrebleu`), which is script-agnostic and scored a clean 1.000 self / 0.172 unrelated baseline. Embedding-similarity is kept, but always reported against its own 0.524 unrelated-pair baseline — a bare cosine number close to that floor isn't signal.

### Retrieval ablation: recall vs. ordering

The first ablation pass (`chrF`/embedding-sim of retrieved text vs. gold *answer*) scored all 6 retrieval configs within the embedding metric's 0.524 noise floor — we were reading noise, not a result. `run_retrieval_ablation.py` was rewritten to score against gold *chunk* labels with proper IR metrics: recall@k, MRR, nDCG@5, latency (mean + p95), reported with n and a 95% CI (n=15 answerable questions; the 16th, a "what's today's market price" question, has no answering chunk by design and is excluded from these metrics — see below).

| Config | recall@1 | recall@3 | recall@5 | recall@10 | MRR | nDCG@5 | latency (mean / p95) |
|---|---|---|---|---|---|---|---|
| dense | 0.667 | 0.800 | 0.933 | **1.000** | 0.763 | 0.738 | 0.005s / 0.007s |
| dense+rerank | 0.800 | 1.000 | 1.000 | 1.000 | 0.878 | 0.855 | 30.5s / 34.3s |
| sparse | 0.333 | 0.667 | 0.733 | **0.800** | 0.503 | 0.555 | 0.007s / 0.011s |
| sparse+rerank | 0.667 | 0.800 | 0.800 | 0.800 | 0.722 | 0.707 | 27.3s / 39.1s |
| hybrid | 0.667 | 0.867 | 0.933 | **1.000** | 0.782 | 0.762 | 0.018s / 0.033s |
| hybrid+rerank | 0.800 | 1.000 | 1.000 | 1.000 | 0.878 | 0.855 | 33.0s / 37.7s |

**No headroom on recall for dense/hybrid, but real headroom on ordering.** Dense-only and hybrid retrieval both already reach recall@10 = 1.0 without any reranking — on a single-crop, ~40-chunk corpus, first-stage retrieval over the top 15 candidates almost always contains the right chunk. Sparse-only does *not* reach ceiling (recall@10 = 0.800, missing 3/15 questions completely) and no amount of reranking can fix that, since reranking only reorders retrieved candidates — it can't surface a chunk that was never retrieved.

Where recall is saturated, reranking's gain shows up exactly where it should: MRR and nDCG@5, not recall. Both dense and hybrid gain +0.10–0.12 in MRR/nDCG with reranking — several questions had the correct chunk retrieved but ranked 2nd–8th, and reranking consistently promoted it to rank 1. This is a genuine, repeatable effect (consistent across two independent retrieval modes), not noise. There was also one honest counter-example (`fertilizer-1`, which has 3 overlapping gold chunks from a redundantly-structured corpus section) where reranking made ordering *worse* — kept in the raw results rather than smoothed over.

**Cost**: reranking adds ~27-33s mean (~34-39s p95) latency per query on this CPU-only machine, vs ~0.02s without — confirmed via a separate micro-benchmark (`diagnose_reranker_latency.py`) to be the intrinsic CPU cost of a 568M-parameter cross-encoder, not a code bug (no reload-per-call, no missed batching, fp32 is already correct for CPU).

**Decision**: `ENABLE_RERANKER` defaults to `false`. On this corpus, the reranker earns a real but modest ordering improvement (MRR/nDCG +0.10-0.12) at a ~1500-6000x latency multiplier, for an end user who may be on patchy rural mobile data. It's retained behind the flag, not deleted — this conclusion is corpus-size-dependent (recall is saturated *because* the corpus is small and single-crop) and should be re-benchmarked if the corpus grows to cover multiple crops, where first-stage recall will no longer be at ceiling and the reranker's value proposition changes.

### What's next

A 3-model generation comparison (`gemma4:e4b` vs `llama3.1:8b` vs `gaganyatri/sarvam-2b-v0.5`) on frozen retrieval contexts, a manual failure-mode review, and a query-rewrite glossary experiment are planned next — see `PROJECT_PLAN.md`.

## Notes

- The eval harness (`evaluate_accuracy.py`, `test_bleu.py`, `test_rag.py`) is not covered by this setup guide — it's being reworked separately and currently has stale imports from the pre-Ollama version of `rag_service.py`.
- `GENERATION_MODEL` is swappable via env var (see table above).
