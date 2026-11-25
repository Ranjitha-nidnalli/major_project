"""
Channel-agnostic message handling (K-036, K-041 to K-045, K-048). Telegram, WhatsApp and
the web API all call handle_message(); adapters only translate their wire format.

Flow for one incoming message:
  1. hash the sender (raw ID is not stored)
  2. no consent yet -> show the consent text; only an explicit yes starts logging
  3. commands: delete my data
  4. per-user daily limit -> refuse with the KCC line
  5. daily Tier B ceiling reached -> Tier A only
  6. answer through the tiered policy, log it (record IDs, source, edition, date), return
     short Kannada messages plus three feedback buttons

User-facing Kannada here was written by Claude and needs a Kannada speaker's review (TODO.md).
"""
import re
import secrets
from datetime import datetime, timezone

from services.fact_answer import KCC_LINE
from services.privacy import RETENTION_DAYS, hash_user

CONSENT_VERSION = "v1"
CONSENT_TEXT_KN = (
    "ನಮಸ್ಕಾರ! ನಾನು ಕೃಷಿ ಮಿತ್ರ. ನಿಮ್ಮ ಪ್ರಶ್ನೆಗಳಿಗೆ ಪರಿಶೀಲಿತ ಮಾಹಿತಿಯಿಂದ ಮಾತ್ರ ಉತ್ತರಿಸುತ್ತೇನೆ; "
    "ಖಚಿತವಿಲ್ಲದಿದ್ದರೆ ಉತ್ತರಿಸುವುದಿಲ್ಲ.\n\n"
    f"ಗಮನಿಸಿ: ನಿಮ್ಮ ಪ್ರಶ್ನೆ ಮತ್ತು ನನ್ನ ಉತ್ತರವನ್ನು ಗುಣಮಟ್ಟ ಪರಿಶೀಲನೆಗಾಗಿ {RETENTION_DAYS} ದಿನ ಉಳಿಸಿಕೊಳ್ಳುತ್ತೇವೆ. "
    "ನಿಮ್ಮ ಫೋನ್ ನಂಬರ್ ಉಳಿಸುವುದಿಲ್ಲ. ಯಾವಾಗ ಬೇಕಾದರೂ 'ಅಳಿಸಿ' ಎಂದು ಕಳುಹಿಸಿದರೆ ನಿಮ್ಮ ಎಲ್ಲಾ ಮಾಹಿತಿ ಅಳಿಸುತ್ತೇವೆ.\n\n"
    "ಒಪ್ಪಿದರೆ 'ಹೌದು' ಎಂದು ಕಳುಹಿಸಿ."
)
WELCOME_KN = "ಧನ್ಯವಾದ. ಈಗ ನಿಮ್ಮ ಪ್ರಶ್ನೆ ಕೇಳಿ."
DECLINED_KN = "ಸರಿ, ಯಾವುದೇ ಮಾಹಿತಿ ಉಳಿಸಿಲ್ಲ. ಮನಸ್ಸು ಬದಲಾದರೆ 'ಹೌದು' ಎಂದು ಕಳುಹಿಸಿ."
DELETED_KN = "ನಿಮ್ಮ ಎಲ್ಲಾ ಮಾಹಿತಿಯನ್ನು ಅಳಿಸಲಾಗಿದೆ."
LIMIT_KN = "ಇಂದಿನ ಪ್ರಶ್ನೆಗಳ ಮಿತಿ ಮುಟ್ಟಿದೆ. ನಾಳೆ ಮತ್ತೆ ಕೇಳಿ."
FEEDBACK_THANKS_KN = {"useful": "ಧನ್ಯವಾದ!", "not_useful": "ಧನ್ಯವಾದ, ಸುಧಾರಿಸುತ್ತೇವೆ.",
                      "report": "ವರದಿಗೆ ಧನ್ಯವಾದ. ತಜ್ಞರು ಪರಿಶೀಲಿಸುತ್ತಾರೆ."}
SOURCE_PREFIX = "ಮೂಲ: "

YES = {"ಹೌದು", "yes", "y", "ok", "okay", "ಒಪ್ಪುತ್ತೇನೆ", "1"}
NO = {"ಇಲ್ಲ", "no", "n", "ಬೇಡ", "2"}
DELETE = {"/delete", "delete", "ಅಳಿಸಿ", "/ಅಳಿಸಿ"}

# Feedback buttons: id, Kannada label (WhatsApp allows 20 characters), stored kind.
BUTTONS = (("u", "ಉಪಯುಕ್ತ", "useful"), ("n", "ಉಪಯುಕ್ತವಲ್ಲ", "not_useful"), ("r", "ತಪ್ಪು ವರದಿ", "report"))
KIND_BY_CODE = {code: kind for code, _l, kind in BUTTONS}

USER_DAILY_LIMIT = 30         # questions per person per day
TIER_B_DAILY_CEILING = 200    # Tier B attempts (the only paid path) per day, all users
MAX_CHARS = 900               # per chat message; fits Telegram and WhatsApp

_BRACKET_ID = re.compile(r"\s*\[[^\]]*\]\s*$")


def source_line(citations):
    """'ಮೂಲ: ...' from the citation list; record IDs in brackets are not for farmers."""
    clean = []
    for c in citations or []:
        c = _BRACKET_ID.sub("", c).strip()
        if c and c not in clean:
            clean.append(c)
    return SOURCE_PREFIX + "; ".join(clean) if clean else ""


def split_messages(text, limit=MAX_CHARS):
    """Split on blank lines, then lines, then characters; no message exceeds `limit`."""
    out, cur = [], ""
    pieces = []
    for para in text.split("\n\n"):
        if len(para) <= limit:
            pieces.append(para)
            continue
        for line in para.split("\n"):
            while len(line) > limit:
                pieces.append(line[:limit])
                line = line[limit:]
            pieces.append(line)
    for p in pieces:
        if cur and len(cur) + 2 + len(p) > limit:
            out.append(cur)
            cur = p
        else:
            cur = f"{cur}\n\n{p}" if cur else p
    if cur:
        out.append(cur)
    return out


def provenance_for(knowledge, record_ids):
    """Which document, region, edition and page each served record came from."""
    seen = []
    for rid in record_ids:
        rec = knowledge.records.get(rid)
        if not rec:
            continue
        src = knowledge.sources[rec["_source"]]
        seen.append({"record_id": rid, "document": src.get("source_title"), "region": src.get("region"),
                     "crop": src.get("crop"), "page": rec.get("printed_page")})
    return seen


def _buttons(answer_id):
    return [{"label": label, "data": f"fb:{answer_id}:{code}"} for code, label, _k in BUTTONS]


def _reply(kind, messages, **extra):
    return {"kind": kind, "messages": messages, "answer_id": None, "tier": None, "tier_reason": None,
            "source_line": "", "buttons": [], **extra}


async def handle_message(store, channel, raw_user_id, text, answer_fn, knowledge=None, now=None,
                         user_limit=USER_DAILY_LIMIT, tier_b_ceiling=TIER_B_DAILY_CEILING):
    """
    `answer_fn(text, tier_b_enabled)` -> dict from rag_service.get_sugarcane_answer (tier,
    tier_reason, answer, source_chunks, record_ids). Returns the reply dict (see _reply).
    """
    now = now or datetime.now(timezone.utc)
    user = hash_user(channel, raw_user_id)
    cleaned = (text or "").strip()
    word = cleaned.lower()

    if word in DELETE:
        store.delete_user(user)
        return _reply("deleted", [DELETED_KN])

    if not store.has_consent(user):
        # Nothing about this person or this question is stored until an explicit yes.
        if word in YES:
            store.grant_consent(user, now, CONSENT_VERSION)
            return _reply("consent_granted", [WELCOME_KN])
        if word in NO:
            return _reply("consent_declined", [DECLINED_KN])
        return _reply("consent_request", [CONSENT_TEXT_KN])

    if not cleaned:
        return _reply("empty", [WELCOME_KN])

    if store.count(now, f"user:{user}") >= user_limit:
        return _reply("rate_limited", [f"{LIMIT_KN}\n\n{KCC_LINE}"])
    store.incr(now, f"user:{user}")

    tier_b_ok = store.count(now, "tier_b") < tier_b_ceiling       # at the ceiling: Tier A or refuse
    result = await answer_fn(cleaned, tier_b_ok)
    tier, reason = result.get("tier") or "C", result.get("tier_reason") or ""
    if tier == "B" or reason.startswith(("tier_b_error", "tier_b_refused", "tier_b_rejected")):
        store.incr(now, "tier_b")

    record_ids = result.get("record_ids") or []
    prov = provenance_for(knowledge, record_ids) if knowledge else []
    citations = result.get("source_chunks") or []
    if tier == "B":
        prov = prov or [{"document": c} for c in citations]
    answer_id = secrets.token_hex(6)
    store.log_answer(answer_id, now, user, channel, cleaned, result["answer"], tier, reason,
                     record_ids, prov)
    src = source_line(citations)
    body = result["answer"]
    if src and src not in body and tier == "B":
        body = f"{body}\n\n{src}"
    return _reply("answer", split_messages(body), answer_id=answer_id, tier=tier, tier_reason=reason,
                  source_line=src, buttons=_buttons(answer_id))


def handle_feedback(store, channel, raw_user_id, data, now=None):
    """`data` is 'fb:<answer_id>:<u|n|r>' from a button. Only the person who got the answer counts."""
    now = now or datetime.now(timezone.utc)
    parts = (data or "").split(":")
    if len(parts) != 3 or parts[0] != "fb" or parts[2] not in KIND_BY_CODE:
        return None
    user = hash_user(channel, raw_user_id)
    answer = store.get_answer(parts[1])
    if not answer or answer["user_hash"] != user:
        return None
    kind = KIND_BY_CODE[parts[2]]
    store.add_feedback(parts[1], user, kind, now)
    return _reply("feedback", [FEEDBACK_THANKS_KN[kind]], answer_id=parts[1])
