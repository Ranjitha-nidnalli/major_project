from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import os

from rag_service import get_sugarcane_answer
from chat_db import connect_db, close_db, get_chat_history

app = FastAPI(title="Krishi Mitra API")

# CORS_ALLOW_ORIGINS: comma-separated list of allowed origins, e.g.
#   CORS_ALLOW_ORIGINS=http://localhost:3000,https://your-deployed-frontend.example
# Defaults to the local Next.js dev server if unset.
_cors_origins_env = os.getenv("CORS_ALLOW_ORIGINS", "http://localhost:3000")
CORS_ALLOW_ORIGINS = [origin.strip() for origin in _cors_origins_env.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    query: str
    session_id: str


class ChatResponse(BaseModel):
    answer: str
    search_score: Optional[float] = None
    accuracy_score: Optional[float] = None
    sources: Optional[List[str]] = None


@app.on_event("startup")
async def startup():
    connect_db()


@app.on_event("shutdown")
async def shutdown():
    close_db()


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    result = await get_sugarcane_answer(
        req.query, req.session_id, return_context=True, interactive=True
    )

    # source_chunks is the un-joined per-chunk list from rag_service.py --
    # splitting the joined "context" string on "\n\n" was unreliable, since
    # a single chunk's flattened text can itself contain an internal blank
    # line (see rag_service.py's return_context block).
    sources = result.get("source_chunks") or None

    return ChatResponse(
        answer=result["answer"],
        search_score=result.get("search_score"),
        accuracy_score=result.get("accuracy_score"),
        sources=sources,
    )


@app.get("/history/{session_id}")
async def history(session_id: str):
    msgs = await get_chat_history(session_id, limit=50)
    return {"session_id": session_id, "messages": msgs}


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
