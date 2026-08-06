# Krishi Mitra

Kannada-language RAG chatbot for sugarcane farmers in Karnataka. FastAPI backend, Next.js frontend, Qdrant hybrid vector store, BGE-M3 embeddings, MongoDB for chat history. Generation runs on local Ollama — no hosted LLM API keys.

## Architecture

```
frontend/ (Next.js, App Router)
  components/chat-interface.tsx   ChatGPT-style UI, posts to POST /chat
backend/ (FastAPI)
  main.py           app entrypoint, /chat /history/{session_id} /health
  rag_service.py    retrieval + generation pipeline (get_sugarcane_answer)
  vector_db.py      Qdrant client + BGE-M3 embedding model singleton;
                     also the seeding script (run directly to build/populate
                     the Qdrant collection from sugarcanemerged3.json)
  chat_db.py        Motor (async MongoDB) chat history persistence
  requirements.txt
```

### Request flow (`rag_service.get_sugarcane_answer`)

1. **Router** — LLM classifies the query into a category (`price`, `disease`, `pest`, `fertilizer`, `general`) and produces an English gloss of the query for web search.
2. **Hybrid retrieval** — BGE-M3 (`vector_db.embed_model`) encodes the query into a dense vector and sparse (lexical) weights. Both are queried against the Qdrant collection (`sugarcane_knowledge`, named vectors `dense` + `sparse`) and fused (`execute_weighted_search`) into a top-5 chunk list.
3. **Query expansion fallback** — if the best dense score is weak, an LLM generates Kannada synonyms and retrieval is retried with the expanded query.
4. **Relevance filtering** — each candidate chunk is checked for relevance before being used as context.
5. **Web fallback** — for `price` queries or when local retrieval has low confidence, falls back to Tavily web search instead of the local KB.
6. **Generation** — an LLM answers strictly from the retrieved context, in Kannada, with a fixed refusal string when the answer isn't in context. Answer is scored for faithfulness against the context by a second LLM call.
7. **Persistence** — user + assistant turns are saved to MongoDB (`chat_db.py`); the last 3 turns are replayed into the prompt as conversational memory.

`ChatResponse` (`main.py`) returns `answer`, `search_score` (top fused retrieval score), and `accuracy_score` (faithfulness judge score).

### Storage

- **Qdrant**: local, opened via `QdrantClient(path=...)` — file-based, single-process. Only one process (either `main.py` or a seeding/eval script) can hold the lock at a time; `unlock_db.py` exists to kill zombie holders.
- **MongoDB**: `mongodb://localhost:27017`, db `sugarcane_chat`, collection `messages`. No auth assumed for local dev.

### Models

- **Embeddings**: BAAI/bge-m3 via `FlagEmbedding.BGEM3FlagModel`, used for both dense and sparse (lexical) vectors.
- **LLM (generation, routing, judging)**: local Ollama, model `gemma4:e4b`, via the Ollama REST API / `ollama` Python package. No hosted API keys.

## Conventions

- All user-facing chatbot text is Kannada; code comments/identifiers are English.
- Backend loads `.env` from `backend/.env` (gitignored) via `python-dotenv`.
- Don't touch the eval harness (`evaluate_accuracy.py`, `test_bleu.py`, `test_rag.py`) — it needs domain-accurate agricultural answers, handled separately.
