import asyncio
import json

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import os

from rag_service import get_sugarcane_answer
from chat_db import connect_db, close_db, get_chat_history
from channels import telegram as tg, whatsapp as wa
from services.chat_core import handle_feedback, handle_message
from services.fact_answer import Knowledge
from services.store import Store

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
    # v2 (ANSWER_MODE=facts) only: answered | refused_* and the confirmed records shown
    facts_status: Optional[str] = None
    record_ids: Optional[List[str]] = None
    tier: Optional[str] = None
    tier_reason: Optional[str] = None


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
        facts_status=result.get("facts_status"),
        record_ids=result.get("record_ids"),
        tier=result.get("tier"),
        tier_reason=result.get("tier_reason"),
    )


# ---- channel-agnostic API (K-036): every channel goes through handle_message ----------

_store = None


def get_store():
    global _store
    if _store is None:
        _store = Store()
    return _store


async def _answer_fn(text, tier_b_ok):
    # tier_b_ok False = daily Tier B ceiling reached: Tier A or refuse. True keeps the
    # TIER_B_ENABLED setting. Channels keep their own log, so no Mongo chat history.
    return await get_sugarcane_answer(text, "channel", return_context=True, interactive=True,
                                      tier_b_enabled=None if tier_b_ok else False, save_history=False)


class AskRequest(BaseModel):
    channel: str = "web"
    user_id: str
    text: str


class Button(BaseModel):
    label: str
    data: str


class AskResponse(BaseModel):
    kind: str                      # answer | consent_request | consent_granted | consent_declined |
    messages: List[str]            # deleted | rate_limited | empty | feedback
    answer_id: Optional[str] = None
    tier: Optional[str] = None
    tier_reason: Optional[str] = None
    source_line: str = ""
    buttons: List[Button] = []


class FeedbackRequest(BaseModel):
    channel: str = "web"
    user_id: str
    data: str


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    return await handle_message(get_store(), req.channel, req.user_id, req.text, _answer_fn,
                                knowledge=Knowledge.load())


@app.post("/feedback", response_model=AskResponse)
async def feedback(req: FeedbackRequest):
    reply = handle_feedback(get_store(), req.channel, req.user_id, req.data)
    if reply is None:
        raise HTTPException(status_code=400, detail="Unknown feedback button")
    return reply


@app.post("/telegram/webhook")
async def telegram_webhook(request: Request):
    if not tg.secret_ok(request.headers.get("X-Telegram-Bot-Api-Secret-Token")):
        raise HTTPException(status_code=403, detail="bad secret")
    event = tg.parse_update(await request.json())
    if event is None:
        return {"ok": True}
    if event["kind"] == "message":
        reply = await handle_message(get_store(), "telegram", event["user_id"], event["text"], _answer_fn,
                                     knowledge=Knowledge.load())
    else:
        reply = handle_feedback(get_store(), "telegram", event["user_id"], event["data"])
        if reply is not None:
            await asyncio.to_thread(_safe_send, tg.send, "answerCallbackQuery",
                                    {"callback_query_id": event["callback_id"], "text": reply["messages"][0]})
            return {"ok": True}
    if reply is not None:
        for method, payload in tg.to_api_calls(reply, event["chat_id"]):
            await asyncio.to_thread(_safe_send, tg.send, method, payload)
    return {"ok": True}      # always 200: Telegram would otherwise retry the same update


@app.get("/whatsapp/webhook")
async def whatsapp_verify(request: Request):
    challenge = wa.verify_challenge(dict(request.query_params))
    if challenge is None:
        raise HTTPException(status_code=403, detail="verification failed")
    return Response(content=challenge, media_type="text/plain")


@app.post("/whatsapp/webhook")
async def whatsapp_webhook(request: Request):
    raw = await request.body()
    if not wa.signature_ok(raw, request.headers.get("X-Hub-Signature-256")):
        raise HTTPException(status_code=403, detail="bad signature")
    for event in wa.parse_payload(json.loads(raw)):
        if event["kind"] == "message":
            reply = await handle_message(get_store(), "whatsapp", event["user_id"], event["text"], _answer_fn,
                                         knowledge=Knowledge.load())
        else:
            reply = handle_feedback(get_store(), "whatsapp", event["user_id"], event["data"])
        if reply is not None:
            for body in wa.to_api_calls(reply, event["user_id"]):
                await asyncio.to_thread(_safe_send, wa.send, body)
    return {"ok": True}


def _safe_send(fn, *args):
    try:
        fn(*args)
    except Exception as exc:      # a failed send must not make the webhook fail and retry
        print(f"⚠️ channel send failed: {type(exc).__name__}")


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
