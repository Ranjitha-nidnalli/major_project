"""
Channel layer (K-036 to K-048): consent, delete, feedback, limits, answer log, adapters, audit.
Pure: in-memory SQLite and fake answer functions; no network, LLM or Telegram/WhatsApp.
"""
import asyncio
import hashlib
import hmac
import json
from datetime import datetime, timedelta, timezone

import pytest

from channels import telegram as tg, whatsapp as wa
from eval import weekly_audit
from services import chat_core as cc
from services.chat_core import handle_feedback, handle_message, split_messages, source_line
from services.fact_answer import KCC_LINE, Knowledge
from services.privacy import hash_user
from services.store import Store

NOW = datetime(2026, 11, 25, 10, 0, tzinfo=timezone.utc)
K = Knowledge.load(reviewed_sheet="__no_such_file__")


@pytest.fixture(autouse=True)
def salt(monkeypatch):
    monkeypatch.setenv("PRIVACY_SALT", "test-salt")


class Fake:
    """Fake answer_fn; records every call so tests can see whether it ran."""
    def __init__(self, result=None):
        self.calls = []
        self.result = result or {"tier": "A", "tier_reason": "confirmed_records", "answer": "ಉತ್ತರ\n\n" + KCC_LINE,
                                 "record_ids": ["uasb-variety-co62175"],
                                 "source_chunks": ["UAS Bengaluru POP 2025, p. 165 [uasb-variety-co62175]"]}

    async def __call__(self, text, tier_b_ok):
        self.calls.append((text, tier_b_ok))
        return self.result


def send(store, text, fake, user="u1", channel="telegram", now=NOW, **kw):
    return asyncio.run(handle_message(store, channel, user, text, fake, knowledge=K, now=now, **kw))


def consented(user="u1", channel="telegram"):
    store, fake = Store(":memory:"), Fake()
    send(store, "ಹೌದು", fake, user=user, channel=channel)
    return store, fake


# ---- K-043 consent -------------------------------------------------------------------

def test_first_contact_shows_consent_and_stores_nothing():
    store, fake = Store(":memory:"), Fake()
    r = send(store, "ಕಬ್ಬಿನ ತಳಿಗಳು", fake)
    assert r["kind"] == "consent_request" and r["messages"] == [cc.CONSENT_TEXT_KN]
    assert not fake.calls
    for table in ("consent", "answers", "feedback", "review_queue", "usage"):
        assert store.db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0, table


def test_consent_text_states_retention_and_deletion():
    assert str(cc.RETENTION_DAYS) in cc.CONSENT_TEXT_KN and "ಅಳಿಸಿ" in cc.CONSENT_TEXT_KN


def test_yes_starts_service_and_no_stores_nothing():
    store, fake = Store(":memory:"), Fake()
    assert send(store, "ಇಲ್ಲ", fake)["kind"] == "consent_declined"
    assert store.db.execute("SELECT COUNT(*) FROM consent").fetchone()[0] == 0
    assert send(store, "ಹೌದು", fake)["kind"] == "consent_granted"
    assert send(store, "ಕಬ್ಬಿನ ತಳಿಗಳು", fake)["kind"] == "answer"


# ---- K-044 privacy -------------------------------------------------------------------

def test_no_raw_identifier_is_stored_anywhere():
    store, fake = consented(user="919876543210")
    send(store, "ಕಬ್ಬಿನ ತಳಿಗಳು", fake, user="919876543210")
    dump = "\n".join(str(row) for t in ("consent", "answers", "usage", "feedback", "review_queue")
                     for row in store.db.execute(f"SELECT * FROM {t}"))
    assert "919876543210" not in dump


def test_hash_is_stable_salted_and_channel_scoped(monkeypatch):
    assert hash_user("telegram", "1") == hash_user("telegram", "1")
    assert hash_user("telegram", "1") != hash_user("whatsapp", "1")
    assert hash_user("telegram", "1", salt="a") != hash_user("telegram", "1", salt="b")
    monkeypatch.delenv("PRIVACY_SALT")
    with pytest.raises(RuntimeError):
        hash_user("telegram", "1")


def test_delete_my_data_removes_everything_about_the_user_only():
    store, fake = consented("a")
    send(store, "ಹೌದು", fake, user="b")
    send(store, "q1", fake, user="a")
    send(store, "q2", fake, user="b")
    aid = send(store, "q3", fake, user="a")["answer_id"]
    handle_feedback(store, "telegram", "a", f"fb:{aid}:r", NOW)
    assert send(store, "ಅಳಿಸಿ", fake, user="a")["kind"] == "deleted"
    assert store.user_rows(hash_user("telegram", "a")) == {"consent": 0, "answers": 0, "feedback": 0}
    assert store.open_reviews() == []
    assert store.user_rows(hash_user("telegram", "b"))["answers"] == 1
    assert send(store, "q", fake, user="a")["kind"] == "consent_request"      # asked afresh


def test_retention_purges_old_answers():
    store, fake = consented()
    send(store, "old", fake, now=NOW - timedelta(days=100))
    send(store, "new", fake, now=NOW)
    assert store.purge_expired(NOW, 90) == 1
    assert [a["question"] for a in store.answers_since(NOW - timedelta(days=365))] == ["new"]


# ---- K-045 answer log ----------------------------------------------------------------

def test_every_answer_is_traceable_to_record_document_and_date():
    store, fake = consented()
    r = send(store, "ಕಬ್ಬಿನ ತಳಿಗಳು", fake)
    row = store.get_answer(r["answer_id"])
    assert row["record_ids"] == ["uasb-variety-co62175"] and row["ts"].startswith("2026-11-25")
    assert row["provenance"][0]["document"] == K.sources["uasb2025"]["source_title"]
    assert row["provenance"][0]["page"] and row["tier"] == "A"


# ---- K-041 feedback ------------------------------------------------------------------

def test_three_buttons_and_report_goes_to_review_queue():
    store, fake = consented()
    r = send(store, "q", fake)
    assert [b["label"] for b in r["buttons"]] == ["ಉಪಯುಕ್ತ", "ಉಪಯುಕ್ತವಲ್ಲ", "ತಪ್ಪು ವರದಿ"]
    assert all(len(b["label"]) <= 20 for b in r["buttons"])
    handle_feedback(store, "telegram", "u1", r["buttons"][0]["data"], NOW)
    assert store.open_reviews() == []
    handle_feedback(store, "telegram", "u1", r["buttons"][2]["data"], NOW)
    assert store.open_reviews() == [r["answer_id"]]


def test_feedback_from_someone_else_or_garbage_is_ignored():
    store, fake = consented("a")
    aid = send(store, "q", fake, user="a")["answer_id"]
    assert handle_feedback(store, "telegram", "intruder", f"fb:{aid}:r", NOW) is None
    assert handle_feedback(store, "telegram", "a", f"fb:{aid}:x", NOW) is None
    assert handle_feedback(store, "telegram", "a", "hello", NOW) is None
    assert handle_feedback(store, "telegram", "a", "fb:nosuch:r", NOW) is None


# ---- K-048 limits --------------------------------------------------------------------

def test_user_daily_limit_refuses_with_kcc_without_calling_the_answerer():
    store, fake = consented()
    for _ in range(2):
        assert send(store, "q", fake, user_limit=2)["kind"] == "answer"
    r = send(store, "q", fake, user_limit=2)
    assert r["kind"] == "rate_limited" and KCC_LINE in r["messages"][0] and len(fake.calls) == 2
    assert send(store, "q", fake, user_limit=2, now=NOW + timedelta(days=1))["kind"] == "answer"


def test_tier_b_ceiling_turns_tier_b_off_for_everyone():
    store = Store(":memory:")
    fake = Fake({"tier": "B", "tier_reason": "retrieval_gate_passed", "answer": "ಸಲಹೆ", "record_ids": [],
                 "source_chunks": ["UAS-B 2025, p. 12"]})
    send(store, "ಹೌದು", fake)
    send(store, "q1", fake, tier_b_ceiling=2)
    send(store, "q2", fake, tier_b_ceiling=2)
    send(store, "q3", fake, tier_b_ceiling=2)
    assert [c[1] for c in fake.calls] == [True, True, False]       # third call: Tier B switched off


def test_tier_a_answers_do_not_use_up_the_tier_b_budget():
    store, fake = consented()
    for _ in range(5):
        send(store, "q", fake, tier_b_ceiling=1)
    assert all(ok for _t, ok in fake.calls)


# ---- K-042 message format ------------------------------------------------------------

def test_messages_are_short_and_keep_all_text():
    text = "\n\n".join(f"ಪ್ಯಾರಾ {i} " + "ಕ" * 300 for i in range(8))
    parts = split_messages(text)
    assert all(len(p) <= cc.MAX_CHARS for p in parts) and len(parts) > 1
    assert "".join(parts).replace("\n", "") == text.replace("\n", "")
    assert split_messages("short") == ["short"]
    assert all(len(p) <= 100 for p in split_messages("x" * 450, limit=100))


def test_source_line_hides_record_ids_and_dedupes():
    assert source_line(["UAS 2025, p. 1 [rec-1]", "UAS 2025, p. 1 [rec-2]"]) == "ಮೂಲ: UAS 2025, p. 1"
    assert source_line([]) == ""


def test_kcc_line_is_on_a_refusal():
    store = Store(":memory:")
    fake = Fake({"tier": "C", "tier_reason": "x", "answer": "ಕ್ಷಮಿಸಿ\n\n" + KCC_LINE, "record_ids": [],
                 "source_chunks": []})
    send(store, "ಹೌದು", fake)
    r = send(store, "q", fake)
    assert r["tier"] == "C" and "1800-180-1551" in "".join(r["messages"])


# ---- K-037 Telegram ------------------------------------------------------------------

def test_telegram_parse_and_reply_shape():
    ev = tg.parse_update({"message": {"chat": {"id": 7}, "from": {"id": 9, "is_bot": False}, "text": "ನಮಸ್ಕಾರ"}})
    assert ev == {"kind": "message", "chat_id": 7, "user_id": 9, "text": "ನಮಸ್ಕಾರ"}
    assert tg.parse_update({"message": {"chat": {"id": 7}, "from": {"id": 9, "is_bot": True}, "text": "x"}}) is None
    assert tg.parse_update({"edited_message": {}}) is None
    cb = tg.parse_update({"callback_query": {"id": "c", "data": "fb:a:u", "from": {"id": 9},
                                             "message": {"chat": {"id": 7}}}})
    assert cb["kind"] == "callback" and cb["data"] == "fb:a:u"
    store, fake = consented()
    calls = tg.to_api_calls(send(store, "q", fake), 7)
    method, payload = calls[-1]
    assert method == "sendMessage" and len(payload["reply_markup"]["inline_keyboard"][0]) == 3


def test_telegram_secret_check(monkeypatch):
    monkeypatch.setenv("TELEGRAM_WEBHOOK_SECRET", "s3cret")
    assert tg.secret_ok("s3cret") and not tg.secret_ok("wrong") and not tg.secret_ok(None)
    monkeypatch.delenv("TELEGRAM_WEBHOOK_SECRET")
    assert not tg.secret_ok("") and not tg.secret_ok("anything")      # unset secret never matches


# ---- K-039 WhatsApp ------------------------------------------------------------------

def test_whatsapp_verify_signature_parse_and_buttons():
    assert wa.verify_challenge({"hub.mode": "subscribe", "hub.verify_token": "t", "hub.challenge": "42"}, "t") == "42"
    assert wa.verify_challenge({"hub.mode": "subscribe", "hub.verify_token": "bad", "hub.challenge": "42"}, "t") is None
    assert wa.verify_challenge({"hub.mode": "subscribe", "hub.challenge": "42"}, "") is None
    body = json.dumps({"x": 1}).encode()
    sig = "sha256=" + hmac.new(b"app", body, hashlib.sha256).hexdigest()
    assert wa.signature_ok(body, sig, "app") and not wa.signature_ok(body + b" ", sig, "app")
    assert not wa.signature_ok(body, None, "app") and not wa.signature_ok(body, sig, "")
    payload = {"entry": [{"changes": [{"value": {"messages": [
        {"from": "9199", "type": "text", "text": {"body": "ಹಲೋ"}},
        {"from": "9199", "type": "interactive", "interactive": {"type": "button_reply", "button_reply": {"id": "fb:a:r"}}},
        {"from": "9199", "type": "image"}]}}]}]}
    events = wa.parse_payload(payload)
    assert [e["kind"] for e in events] == ["message", "callback"] and events[1]["data"] == "fb:a:r"
    store, fake = consented(channel="whatsapp")
    bodies = wa.to_api_calls(send(store, "q", fake, channel="whatsapp"), "9199")
    assert bodies[-1]["type"] == "interactive" and len(bodies[-1]["interactive"]["action"]["buttons"]) == 3
    assert all(len(b["reply"]["title"]) <= 20 for b in bodies[-1]["interactive"]["action"]["buttons"])


# ---- K-046 weekly audit --------------------------------------------------------------

def test_weekly_audit_sample_and_marks_roundtrip(tmp_path):
    store, fake = consented()
    for i in range(6):
        send(store, f"q{i}", fake, now=NOW - timedelta(days=i))
    rows = weekly_audit.sample(store, NOW + timedelta(hours=1), days=3, n=10, seed=1)
    assert len(rows) == 3                                           # only the last 3 days, no more than exist
    sheet = tmp_path / "audit.csv"
    weekly_audit.write_sheet(rows, str(sheet))
    import csv
    with open(sheet, encoding="utf-8-sig") as f:
        table = list(csv.reader(f))
    assert weekly_audit.VERDICT_COL in table[0] and len(table) == 4      # header + 3 answers
    with open(sheet, encoding="utf-8-sig") as f:
        marked = list(csv.DictReader(f))
    marked[0][weekly_audit.VERDICT_COL], marked[1][weekly_audit.VERDICT_COL] = "OK", "wrong"
    with open(sheet, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(marked[0]))
        w.writeheader()
        w.writerows(marked)
    assert weekly_audit.import_marks(store, str(sheet), "2026-W48") == {"OK": 1, "WRONG": 1}
    marked[2][weekly_audit.VERDICT_COL] = "fine"
    with open(sheet, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(marked[0]))
        w.writeheader()
        w.writerows(marked)
    with pytest.raises(ValueError):
        weekly_audit.import_marks(store, str(sheet), "2026-W48")
