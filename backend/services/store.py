"""
SQLite store for the chat channels (K-043 to K-048): consent, answer log, feedback, review
queue, usage counters and audit marks. Standard library only, so it needs no service and
the free tier runs it as one file (KM_DB_PATH). Nothing here holds a raw phone number.
"""
import json
import os
import sqlite3
from datetime import datetime, timedelta, timezone

SCHEMA = """
CREATE TABLE IF NOT EXISTS consent (user_hash TEXT PRIMARY KEY, granted_at TEXT, text_version TEXT);
CREATE TABLE IF NOT EXISTS answers (
  answer_id TEXT PRIMARY KEY, ts TEXT, user_hash TEXT, channel TEXT, question TEXT, answer_text TEXT,
  tier TEXT, tier_reason TEXT, record_ids TEXT, provenance TEXT);
CREATE TABLE IF NOT EXISTS feedback (
  id INTEGER PRIMARY KEY AUTOINCREMENT, answer_id TEXT, user_hash TEXT, kind TEXT, ts TEXT);
CREATE TABLE IF NOT EXISTS review_queue (
  id INTEGER PRIMARY KEY AUTOINCREMENT, answer_id TEXT, reason TEXT, ts TEXT, status TEXT DEFAULT 'open');
CREATE TABLE IF NOT EXISTS usage (day TEXT, key TEXT, n INTEGER, PRIMARY KEY (day, key));
CREATE TABLE IF NOT EXISTS audit_marks (
  answer_id TEXT PRIMARY KEY, verdict TEXT, reviewer TEXT, note TEXT, week TEXT);
"""


def _iso(dt):
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


def _day(dt):
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d")


class Store:
    def __init__(self, path=None):
        path = path or os.getenv("KM_DB_PATH", os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "data", "krishi.db"))
        if path != ":memory:":
            os.makedirs(os.path.dirname(path), exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.executescript(SCHEMA)

    # consent -------------------------------------------------------------------------
    def has_consent(self, user_hash):
        return self.db.execute("SELECT 1 FROM consent WHERE user_hash=?", (user_hash,)).fetchone() is not None

    def grant_consent(self, user_hash, now, text_version):
        self.db.execute("INSERT OR REPLACE INTO consent VALUES (?,?,?)", (user_hash, _iso(now), text_version))
        self.db.commit()

    # answer log ----------------------------------------------------------------------
    def log_answer(self, answer_id, now, user_hash, channel, question, answer_text, tier, tier_reason,
                   record_ids, provenance):
        self.db.execute("INSERT INTO answers VALUES (?,?,?,?,?,?,?,?,?,?)", (
            answer_id, _iso(now), user_hash, channel, question, answer_text, tier, tier_reason,
            json.dumps(record_ids), json.dumps(provenance, ensure_ascii=False)))
        self.db.commit()

    def get_answer(self, answer_id):
        row = self.db.execute("SELECT answer_id, ts, user_hash, channel, question, answer_text, tier, "
                              "tier_reason, record_ids, provenance FROM answers WHERE answer_id=?",
                              (answer_id,)).fetchone()
        if not row:
            return None
        keys = ("answer_id", "ts", "user_hash", "channel", "question", "answer_text", "tier",
                "tier_reason", "record_ids", "provenance")
        d = dict(zip(keys, row))
        d["record_ids"], d["provenance"] = json.loads(d["record_ids"]), json.loads(d["provenance"])
        return d

    def answers_since(self, since):
        rows = self.db.execute("SELECT answer_id FROM answers WHERE ts >= ? ORDER BY ts", (_iso(since),)).fetchall()
        return [self.get_answer(r[0]) for r in rows]

    # feedback and review queue -------------------------------------------------------
    def add_feedback(self, answer_id, user_hash, kind, now):
        self.db.execute("INSERT INTO feedback (answer_id, user_hash, kind, ts) VALUES (?,?,?,?)",
                        (answer_id, user_hash, kind, _iso(now)))
        if kind == "report":
            self.db.execute("INSERT INTO review_queue (answer_id, reason, ts) VALUES (?,?,?)",
                            (answer_id, "reported_bad_answer", _iso(now)))
        self.db.commit()

    def open_reviews(self):
        return [r[0] for r in self.db.execute("SELECT answer_id FROM review_queue WHERE status='open' ORDER BY id")]

    # usage counters (rate limit and cost ceiling) ------------------------------------
    def count(self, now, key):
        row = self.db.execute("SELECT n FROM usage WHERE day=? AND key=?", (_day(now), key)).fetchone()
        return row[0] if row else 0

    def incr(self, now, key):
        self.db.execute("INSERT INTO usage VALUES (?,?,1) ON CONFLICT(day,key) DO UPDATE SET n=n+1",
                        (_day(now), key))
        self.db.commit()

    # audit marks ---------------------------------------------------------------------
    def save_mark(self, answer_id, verdict, reviewer, note, week):
        self.db.execute("INSERT OR REPLACE INTO audit_marks VALUES (?,?,?,?,?)",
                        (answer_id, verdict, reviewer, note, week))
        self.db.commit()

    def mark_counts(self):
        return dict(self.db.execute("SELECT verdict, COUNT(*) FROM audit_marks GROUP BY verdict").fetchall())

    # delete and retention ------------------------------------------------------------
    def delete_user(self, user_hash):
        """Remove everything stored about one person. Returns the number of answers removed."""
        ids = [r[0] for r in self.db.execute("SELECT answer_id FROM answers WHERE user_hash=?", (user_hash,))]
        self._delete_answers(ids)
        self.db.execute("DELETE FROM feedback WHERE user_hash=?", (user_hash,))
        self.db.execute("DELETE FROM consent WHERE user_hash=?", (user_hash,))
        self.db.execute("DELETE FROM usage WHERE key=?", (f"user:{user_hash}",))
        self.db.commit()
        return len(ids)

    def purge_expired(self, now, days):
        cutoff = _iso(now - timedelta(days=days))
        ids = [r[0] for r in self.db.execute("SELECT answer_id FROM answers WHERE ts < ?", (cutoff,))]
        self._delete_answers(ids)
        self.db.commit()
        return len(ids)

    def _delete_answers(self, ids):
        for i in ids:
            for table in ("answers", "feedback", "review_queue", "audit_marks"):
                self.db.execute(f"DELETE FROM {table} WHERE answer_id=?", (i,))

    def user_rows(self, user_hash):
        """Row counts held for a user (used by tests and the delete confirmation)."""
        q = lambda sql: self.db.execute(sql, (user_hash,)).fetchone()[0]
        return {"consent": q("SELECT COUNT(*) FROM consent WHERE user_hash=?"),
                "answers": q("SELECT COUNT(*) FROM answers WHERE user_hash=?"),
                "feedback": q("SELECT COUNT(*) FROM feedback WHERE user_hash=?")}
