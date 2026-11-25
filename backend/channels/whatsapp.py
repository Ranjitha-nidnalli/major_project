"""
WhatsApp adapter on the Meta WhatsApp Cloud API (K-039), for the free test number. Real
numbers need business verification and have per-conversation fees: that is K-040 (check
Meta's current pricing page; this project has not verified it). Env: WHATSAPP_VERIFY_TOKEN,
WHATSAPP_APP_SECRET, WHATSAPP_ACCESS_TOKEN, WHATSAPP_PHONE_NUMBER_ID (never committed).
"""
import hashlib
import hmac
import json
import os
import urllib.request

GRAPH_VERSION = os.getenv("WHATSAPP_GRAPH_VERSION", "v21.0")


def verify_challenge(params, expected_token=None):
    """GET webhook handshake. Returns the challenge string to echo, or None."""
    expected_token = expected_token if expected_token is not None else os.getenv("WHATSAPP_VERIFY_TOKEN", "")
    ok = (params.get("hub.mode") == "subscribe" and expected_token
          and hmac.compare_digest(params.get("hub.verify_token") or "", expected_token))
    return params.get("hub.challenge") if ok else None


def signature_ok(raw_body, header_value, app_secret=None):
    """POST bodies carry X-Hub-Signature-256: sha256=<HMAC of the raw body with the app secret>."""
    app_secret = app_secret if app_secret is not None else os.getenv("WHATSAPP_APP_SECRET", "")
    if not app_secret or not header_value or not header_value.startswith("sha256="):
        return False
    good = hmac.new(app_secret.encode("utf-8"), raw_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(header_value[len("sha256="):], good)


def parse_payload(payload):
    """-> list of {'kind': 'message'|'callback', 'user_id' (phone number), 'text'|'data'}."""
    out = []
    for entry in payload.get("entry", []):
        for change in entry.get("changes", []):
            for m in change.get("value", {}).get("messages", []):
                if m.get("type") == "text":
                    out.append({"kind": "message", "user_id": m["from"], "text": m["text"]["body"]})
                elif m.get("type") == "interactive" and m["interactive"].get("type") == "button_reply":
                    out.append({"kind": "callback", "user_id": m["from"],
                                "data": m["interactive"]["button_reply"]["id"]})
    return out


def to_api_calls(reply, to):
    """Reply dict -> Graph API message bodies. Buttons ride on the last message (max 3)."""
    calls = []
    for i, text in enumerate(reply["messages"]):
        base = {"messaging_product": "whatsapp", "to": to}
        if reply["buttons"] and i == len(reply["messages"]) - 1:
            calls.append({**base, "type": "interactive", "interactive": {
                "type": "button", "body": {"text": text},
                "action": {"buttons": [{"type": "reply", "reply": {"id": b["data"], "title": b["label"][:20]}}
                                       for b in reply["buttons"][:3]]}}})
        else:
            calls.append({**base, "type": "text", "text": {"body": text}})
    return calls


def send(body, token=None, phone_number_id=None):
    token = token or os.getenv("WHATSAPP_ACCESS_TOKEN")
    phone_number_id = phone_number_id or os.getenv("WHATSAPP_PHONE_NUMBER_ID")
    if not token or not phone_number_id:
        raise RuntimeError("WHATSAPP_ACCESS_TOKEN and WHATSAPP_PHONE_NUMBER_ID must be set")
    req = urllib.request.Request(
        f"https://graph.facebook.com/{GRAPH_VERSION}/{phone_number_id}/messages",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read())
