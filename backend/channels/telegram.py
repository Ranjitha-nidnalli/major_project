"""
Telegram adapter (K-037). Pure translation plus a small sender; the logic is in
services/chat_core.py. Set TELEGRAM_BOT_TOKEN and TELEGRAM_WEBHOOK_SECRET in backend/.env
(never committed). Free to run: Telegram's Bot API has no per-message charge.
"""
import hmac
import json
import os
import urllib.request


def secret_ok(header_value, expected=None):
    """Telegram sends our secret back in X-Telegram-Bot-Api-Secret-Token on every update."""
    expected = expected if expected is not None else os.getenv("TELEGRAM_WEBHOOK_SECRET", "")
    return bool(expected) and hmac.compare_digest(header_value or "", expected)


def parse_update(update):
    """-> {'kind': 'message'|'callback', 'chat_id', 'user_id', 'text'|'data', ...} or None."""
    msg = update.get("message")
    if msg and msg.get("text") is not None and msg.get("from") and not msg["from"].get("is_bot"):
        return {"kind": "message", "chat_id": msg["chat"]["id"], "user_id": msg["from"]["id"],
                "text": msg["text"]}
    cb = update.get("callback_query")
    if cb and cb.get("data") and cb.get("message"):
        return {"kind": "callback", "chat_id": cb["message"]["chat"]["id"], "user_id": cb["from"]["id"],
                "data": cb["data"], "callback_id": cb["id"]}
    return None


def to_api_calls(reply, chat_id):
    """Reply dict (chat_core) -> [(method, payload)]. Buttons ride on the last message."""
    calls = []
    for i, text in enumerate(reply["messages"]):
        payload = {"chat_id": chat_id, "text": text}
        if reply["buttons"] and i == len(reply["messages"]) - 1:
            payload["reply_markup"] = {"inline_keyboard": [[
                {"text": b["label"], "callback_data": b["data"]} for b in reply["buttons"]]]}
        calls.append(("sendMessage", payload))
    return calls


def send(method, payload, token=None):
    token = token or os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")
    req = urllib.request.Request(
        f"https://api.telegram.org/bot{token}/{method}", data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read())
