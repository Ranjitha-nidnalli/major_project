# Krishi Mitra

A Kannada-language RAG chatbot that answers sugarcane farming questions for
Karnataka farmers. FastAPI backend, Next.js frontend, Qdrant hybrid
(dense + sparse) vector store, BGE-M3 embeddings, and MongoDB for chat
history.

This README describes what the code in this repository actually does today.
For the target architecture and the phased roadmap toward it, see
`ARCHITECTURE.md`. For current facts and working rules, see `CLAUDE.md`.

## Architecture

- **Backend**: FastAPI (`backend/main.py`), endpoints `POST /chat`,
  `GET /history/{session_id}`, `GET /health`.
- **Frontend**: Next.js (`frontend/`), a chat UI that posts to `/chat`.
- **Vector store**: Qdrant, collection `sugarcane_knowledge`, named vectors
  `dense` + `sparse`. Server mode (Docker) is the target setup; see
  "Running the stack" below. Falls back to a local file-mode store if
  `QDRANT_URL` is unset.
- **Embeddings**: BAAI/bge-m3 via `FlagEmbedding.BGEM3FlagModel`, used for
  both the dense vector and the learned-sparse (lexical) vector — not
  classical BM25.
- **Reranker**: a `bge-reranker-v2-m3` cross-encoder is loaded at startup
  (`backend/vector_db.py`) but is **not used anywhere in the live request
  path**. `ENABLE_RERANKER` and `RERANK_THRESHOLD` are defined in
  `rag_service.py` but nothing reads them yet — reranking is not wired in.
  Treat both as dead configuration until that changes.
- **LLM**: `backend/llm_client.py` abstracts over three backends
  (`groq`, `openrouter`, `ollama`), selected by `LLM_BACKEND` (default
  `groq`). Model ID comes from `GENERATION_MODEL` — there is no hardcoded
  fallback. Kannada generation quality has not been validated.
- **Chat history**: MongoDB, db `sugarcane_chat`, collection `messages`.
- **Corpus**: `sugarcanemerged3.json` at the repo root. Provenance
  unknown — it contains Tamil Nadu content and should not be treated as
  authoritative for Karnataka-specific advice (see `CLAUDE.md`).

## Request flow (`rag_service.get_sugarcane_answer`)

1. **Router** — an LLM classifies the query into a category
   (`price`, `disease`, `pest`, `fertilizer`, `general`) and produces an
   English gloss of the query. The category is used *only* to pick a
   stricter abstention-gate threshold for safety-critical categories
   (pest/disease/fertilizer) — it is never used to filter or exclude
   retrieved documents. (An earlier version applied the category as a hard
   Qdrant filter, which made gold chunks unreachable whenever the router
   misclassified a query; that filter has been removed.)
2. **Hybrid retrieval** — the query is embedded with BGE-M3 into a dense
   vector and sparse (lexical) weights. Both are queried against Qdrant and
   fused with Reciprocal Rank Fusion (RRF), returning the top 5 chunks. No
   category filter, no reranking.
3. **Abstention gate** (`backend/services/gating.py`) — a pure function
   decides whether to answer or refuse, based on the max dense cosine
   similarity over the retrieved chunks (a real relevance score) and the
   query's category (used only to select a stricter threshold for
   safety-critical categories). The threshold is an explicitly
   **uncalibrated placeholder**; it has not been tuned against a labelled
   answerable/unanswerable question set. Do not treat refusal accuracy as
   validated until that calibration happens.
4. **Generation** — an LLM answers strictly from the retrieved context, in
   Kannada, with a fixed refusal string used when it can't find the answer
   in context. The last 3 turns of chat history are replayed as
   conversational memory.
5. **Live-path checks after generation** (interactive requests only):
   - An LLM faithfulness judge scores the answer against the retrieved
     context; scores below `FAITHFULNESS_GATE_THRESHOLD` (0.50) trigger a
     refusal.
   - A regex-based numeric-faithfulness check
     (`backend/eval/numeric_faithfulness.py`) cross-references numbers in
     the answer against numbers in the context. **This check has known
     false-positive and false-negative failure modes** (it can mis-bind a
     number to the wrong unit) and is left as-is in this phase — replacing
     it with field-identity verification against typed fact records is
     later-phase work, not fixed here. Don't treat its output as a
     reliable safety signal.
6. **Persistence** — user and assistant turns are saved to MongoDB.

`ChatResponse` (`main.py`) returns `answer`, `search_score` (the RRF fusion
score — kept for telemetry/display only, **not** used by the abstention
gate), `accuracy_score` (the faithfulness judge score), and `sources` (the
retrieved chunk texts, returned as a list so the frontend doesn't have to
reconstruct source boundaries by splitting a joined string — see
`rag_service.py`'s `source_chunks` field).

## Prerequisites

- Python 3.10+
- Node.js 18+
- Docker (for Qdrant + MongoDB in server mode) — see below for the
  file-mode fallback if you don't want to use Docker
- An API key for whichever `LLM_BACKEND` you configure (Groq is the
  default; a free-tier key works at https://console.groq.com)

## Running the stack

### 1. Start Qdrant and MongoDB

```bash
docker compose up -d
```

This starts Qdrant (`localhost:6333`) and MongoDB (`localhost:27017`) with
persistent named volumes. If you don't want to run Docker, you can omit
this step and leave `QDRANT_URL` unset in your `.env` — `vector_db.py`
will fall back to a local file-mode Qdrant store at
`backend/qdrant_sugarcane_db/`, which only allows one process to hold it
at a time (you cannot run `main.py` and a seeding/eval script
simultaneously in that mode). MongoDB has no equivalent fallback; you need
a MongoDB instance reachable at `MONGO_URI` either way.

### 2. Backend environment

```bash
cd backend
cp .env.example .env
```

Fill in `.env` — see `backend/.env.example` for every variable and what it
does. At minimum you need an API key for your chosen `LLM_BACKEND`.

### 3. Install backend dependencies

```bash
cd backend
pip install -r requirements.txt
```

First run will download the BGE-M3 embedding model and the BGE reranker
model from Hugging Face (several GB total; the reranker is loaded even
though it isn't used in the live path yet — see Architecture above).

### 4. Seed the Qdrant collection

Place your source knowledge file at `backend/sugarcanemerged3.json`, then:

```bash
cd backend
python vector_db.py
```

This chunks the JSON, generates dense + sparse embeddings, and upserts
into the `sugarcane_knowledge` Qdrant collection. Re-run any time the
source data changes — it recreates the collection from scratch.

### 5. Run the backend

```bash
cd backend
python main.py
```

Serves on `http://localhost:8000`.

### 6. Run the frontend

```bash
cd frontend
npm install
npm run dev
```

Serves on `http://localhost:3000` and talks to the backend at
`http://localhost:8000`. Set `CORS_ALLOW_ORIGINS` in the backend `.env` if
you serve the frontend from a different origin.

## Tests

```bash
cd backend
pytest
```

Runs `backend/tests/` only (scoped by `backend/pytest.ini`) — currently 32
tests across 6 files (category filter, gating, chunking helpers, Kannada
normalization, numeric faithfulness, requirements-import resolution), none
of which require Qdrant, MongoDB, or an LLM API key. The category-filter
test uses an in-memory Qdrant client; the chunking-helper tests AST-extract
`vector_db.py`'s pure functions rather than importing that module directly,
since it loads ~7GB of models as an import-time side effect. Scripts like
`quick_test.py` and `backend/eval/refusal_test.py` are manual, live-service
scripts, not part of the automated suite — see their docstrings for what
they need running.

## Evaluation

Eval code lives in `backend/eval/`. Several of its data files
(`gold.jsonl`, `chunks.jsonl`, `contexts.json`, `results.jsonl`, and
others) describe a corpus state that no longer exists and have been moved
to `backend/eval/_archive_stale/` with an explanation of why each is
stale — see the README there. The scripts that read those paths
(`run_retrieval_ablation.py`, `refusal_test.py`, `threshold_sweep.py`, and
others) will fail with `FileNotFoundError` until a current, hand-labelled
gold set is built; this is intentional, not a bug — the alternative would
be silently running against circular, non-resolving labels.

This README does not report retrieval ablation numbers, generation-model
comparison numbers, or metric-validity numbers. Those all depend on the
stale/archived data above and would be misleading if reprinted here
without being regenerated against a current corpus and a current,
non-circular gold set.

## Known limitations (current, not aspirational)

- The abstention gate's relevance threshold is uncalibrated.
- The numeric-faithfulness check (fixed 2026-09-21: numbers no longer
  mis-bind to a neighboring quantity's unit) is still regex-based, not a
  structured extraction against a verified fact record — it can only catch
  numbers the generation model invents or alters, not chemical names, and
  in `strict=True` mode any bare number in the answer without a matching
  context number is flagged, which can over-fire on paraphrased step counts
  or dates. Not a substitute for the human dosage-verification step.
- The corpus's provenance is unknown and it contains non-Karnataka
  content.
- Kannada generation quality has not been validated against native
  speakers.
- The reranker is loaded but not used in the live request path.
- `ollama` and `openrouter` backends in `llm_client.py` are less tested
  than the default `groq` path; `ollama` in particular is documented as
  untested on Windows+Python 3.13.
- **PII / prompt injection**: no input sanitization, prompt-injection
  defenses, or PII handling are implemented. Deliberately out of scope for
  this phase, not an oversight — noted here so it isn't mistaken for one.
- **Generation determinism**: `temperature=0.0` everywhere `call_llm` is
  invoked (`rag_service.py`, `llm_client.py`'s default). Deliberate: reduces
  eval variance and makes a given (query, context) pair reproducible, at
  the cost of more mechanical-sounding Kannada phrasing than a natural
  temperature would produce. A safety-critical dosage bot should prefer
  reproducible over natural.

### Production readiness (explicitly deferred, not addressed here)

This is a prototype, not a deployable service. Known, deliberately-deferred
operational gaps:

- No authentication or authorization on any endpoint.
- No rate limiting — a single client can exhaust the Groq/OpenRouter quota
  or hammer the local Qdrant/Mongo instances.
- Logging is `print()` statements, not structured logging; no log
  aggregation, no request tracing.
- No CI/CD pipeline (`backend/tests/` runs locally via `pytest` only).
- No secrets management beyond a local `.env` file.
- No monitoring, alerting, or uptime guarantees.

See `ARCHITECTURE.md` for the phased plan that addresses the retrieval/
safety limitations above (the production-readiness gaps are out of scope
for the phases currently planned, not just not-yet-reached).
