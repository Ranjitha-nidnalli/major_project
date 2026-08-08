# Krishi Mitra

A Kannada-language RAG chatbot that answers sugarcane farming questions for Karnataka farmers. FastAPI backend, Next.js frontend, Qdrant hybrid (dense + sparse) vector store, BGE-M3 embeddings, a BGE cross-encoder reranker, and MongoDB for chat history.

## Architecture & LLM Backend

The system is designed around a local-first stack: **FastAPI** backend, **Next.js** frontend, **Qdrant** hybrid (dense + sparse) vector store, **BGE-M3** embeddings, and **MongoDB** for chat history. Retrieval and embedding run entirely on the local machine with no external dependencies.

**LLM Generation:** The system was originally implemented against local **Ollama** models for zero-cost, offline operation. Ollama's Python client (v0.6.2) proved unstable in our environment (Windows 11, Python 3.13), crashing on every generation call with a response-parsing error (`'NoneType' object is not subscriptable`). After extensive debugging, local inference could not be restored in the project timeline.

For the prototype evaluation and live demo, generation was switched to **Groq's hosted Llama 3.1 API**. This requires a `GROQ_API_KEY` and internet connectivity. The backend includes an `llm_client.py` abstraction that supports both Groq and Ollama backends via the `LLM_BACKEND` environment variable; restoring fully local inference is future work.

**Request flow:** query → LLM router (category + English gloss) → BGE-M3 hybrid retrieval (dense + sparse, fused with Reciprocal Rank Fusion) → query-expansion retry on weak matches → optional BGE cross-encoder reranks/filters → LLM generates a Kannada answer strictly from context → LLM judges faithfulness → **faithfulness gate blocks low-score answers** → answer + `search_score` + `accuracy_score` + sources saved to MongoDB and returned.

## Prerequisites

- Python 3.10+
- Node.js 18+
- MongoDB running locally (no auth needed)
- Groq API key (free tier at console.groq.com)

## Setup

### 1. Backend environment

Create `backend/.env` (gitignored):

| Variable | Default | Purpose |
|----------|---------|---------|
| `MONGO_URI` | `mongodb://localhost:27017` | MongoDB connection string |
| `LLM_BACKEND` | `groq` | `groq` or `ollama` (ollama untested) |
| `GENERATION_MODEL` | `llama-3.1-8b-instant` | Groq model id |
| `GROQ_API_KEY` | — | Required if `LLM_BACKEND=groq` |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | HF model id for dense+sparse embeddings |
| `RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | HF model id for cross-encoder |
| `ENABLE_RERANKER` | `false` | Set to `true` to rerank retrieved chunks |
| `ENABLE_WEB_FALLBACK` | `false` | Set to `true` to enable Tavily web search fallback |
| `TAVILY_API_KEY` | — | Only needed if `ENABLE_WEB_FALLBACK=true` |

### 2. Install backend dependencies

```bash
cd backend
pip install -r requirements.txt
```

First run will download the BGE-M3 embedding model and the BGE reranker model from Hugging Face (a few GB total).

### 3. Seed the Qdrant database

Qdrant runs in local file-mode (no separate server) at `backend/qdrant_sugarcane_db/`. Place your source knowledge file at `backend/sugarcanemerged3.json`, then:

```bash
cd backend
python vector_db.py
```

This chunks the JSON, generates dense + sparse embeddings for each chunk, and upserts them into the `sugarcane_knowledge` Qdrant collection. Re-run it any time you update the source data.

Qdrant's file-mode store only allows one process to hold it at a time. If you see a "DB SERVER IS LOCKED" error, stop `main.py` first; `python unlock_db.py` can kill zombie holders.

### 4. Run the backend

```bash
cd backend
python main.py
```

Serves on `http://localhost:8000`. Endpoints: `POST /chat`, `GET /history/{session_id}`, `GET /health`.

### 5. Run the frontend

```bash
cd frontend
npm install
npm run dev
```

Serves on `http://localhost:3000` and talks to the backend at `http://localhost:8000`.

## Evaluation

All eval code and data lives in `backend/eval/`.

### Metric validity

We initially scored retrieved/generated text against gold answers with ROUGE-L and BGE-M3 embedding cosine similarity. `backend/eval/diagnose_metrics.py` checks any metric before trusting it, by scoring each reference against itself (expect ≈1.0) and against an unrelated reference (the metric's noise floor):

| Metric | self-score | unrelated-score | usable range |
|--------|-----------|-----------------|-------------|
| ROUGE-L | 0.625 | 0.048 | 0.577 |
| chrF | 1.000 | 0.172 | 0.828 |
| embedding-sim | 1.000 | 0.524 | 0.476 |

ROUGE-L is broken for this corpus — its tokenizer strips non-Latin script, so 6 of 16 pure-Kannada references scored **0.000 against themselves**. We replaced it with **chrF** (character-n-gram, `sacrebleu`), which is script-agnostic and scored a clean 1.000 self / 0.172 unrelated baseline.

### Retrieval ablation

`run_retrieval_ablation.py` scores against gold chunk labels with proper IR metrics: recall@k, MRR, nDCG@5, latency (mean + p95), reported with n and a 95% CI (n=15 answerable questions; price-1 is excluded as unanswerable).

| Config | recall@1 | recall@3 | recall@5 | recall@10 | MRR | nDCG@5 | Latency (mean / p95) |
|--------|----------|----------|----------|-----------|-----|--------|---------------------|
| dense | 0.667±0.247 | 0.800±0.210 | 0.933±0.131 | 1.000±0.000 | 0.763±0.179 | 0.738±0.167 | 0.00s / 0.01s |
| dense+rerank | 0.800±0.210 | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 | 0.878±0.129 | 0.855±0.124 | 30.52s / 34.30s |
| sparse | 0.333±0.247 | 0.667±0.247 | 0.733±0.232 | 0.800±0.210 | 0.503±0.203 | 0.555±0.200 | 0.01s / 0.01s |
| sparse+rerank | 0.667±0.247 | 0.800±0.210 | 0.800±0.210 | 0.800±0.210 | 0.722±0.215 | 0.707±0.209 | 27.31s / 39.08s |
| hybrid | 0.667±0.247 | 0.867±0.178 | 0.933±0.131 | 1.000±0.000 | 0.782±0.167 | 0.762±0.155 | 0.02s / 0.03s |
| hybrid+rerank | 0.800±0.210 | 1.000±0.000 | 1.000±0.000 | 1.000±0.000 | 0.878±0.129 | 0.855±0.124 | 32.98s / 37.70s |

**Key finding:** Dense/hybrid already achieve recall@10 = 1.0 without reranking. Reranking improves MRR/nDCG by +0.10–0.12 but adds ~30s latency on CPU vs. ~0.02s without. `ENABLE_RERANKER` defaults to `false`. This conclusion is corpus-size-dependent and must be re-benchmarked if the corpus scales.

### Generation evaluation

`build_contexts.py` freezes retrieval contexts once. `run_eval.py` replays identical contexts to the generation model, with LLM-judged faithfulness.

**Llama 3.1 8B (Groq) results (n=16):**

| Metric | Average | Notes |
|--------|---------|-------|
| chrF | 0.416 | Moderate lexical overlap; Kannada agglutination lowers n-gram match |
| Embedding sim | 0.730 (+0.206 vs baseline) | Strong semantic alignment with gold answers |
| Faithfulness | 0.765 | High grounding in retrieved context |
| Gen latency | 24.1s | Groq free-tier queueing after initial rapid calls |

**Self-judge caveat:** Due to Groq free-tier rate limits, we used the same model as judge and generator. Self-judging is a known source of bias; faithfulness scores should be read as an internal consistency check, not an absolute quality measure, until a genuinely separate judge model is used.

### Safety

- **Hard refusal** on low-confidence queries (`search_score < 0.35`)
- **Safety-critical gate**: pest/disease/fertilizer queries with medium confidence are refused
- **Faithfulness gate**: generated answers with `accuracy_score < 0.50` are blocked and replaced with a refusal
- **Escalation line** on every answer: Kisan Call Centre `1800-180-1551`

## What's next

- Human evaluation subset (~30–50 answers)
- Manual failure-mode review (tagged taxonomy)
- Refusal-accuracy metric for unanswerable questions
- Additional crops (ragi, tomato) to test reranker off the recall ceiling
- Restore Ollama local-inference path (currently blocked by Windows/Python 3.13 client bug)
