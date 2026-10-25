"""
Live probe of the optional query canonicaliser (services/canonicalizer.py).

Runs eval/canonicalizer_probe.json through the real LLM (one call per probe,
LLM_BACKEND / GENERATION_MODEL from backend/.env) and then through the normal
v2 lookup with the model's topics as the fallback. Writes its own run file in
eval/runs/canonicalizer/ with the probe-set hash, model id and prompt hash.

This is a developer probe, not a farmer evaluation: see the note in the
probe file. Makes paid/free-tier API calls: run only when asked to.

Usage: cd backend && python eval/run_canonicalizer_probe.py [--limit N]
"""
import argparse
import asyncio
import datetime
import hashlib
import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from llm_client import call_llm, LLM_BACKEND, _DEFAULT_MODEL  # noqa: E402
from services import canonicalizer  # noqa: E402
from services.fact_answer import KNOWLEDGE_DIR, Knowledge, answer, match, topic_labels  # noqa: E402

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
PROBES = os.path.join(EVAL_DIR, "canonicalizer_probe.json")
OUT_DIR = os.path.join(EVAL_DIR, "runs", "canonicalizer")
OLD_INTENTS_COMMIT = "551c205^"   # keyword lists before the pilot-1 misses were fixed


def _knowledge_with_old_intents():
    """Copy knowledge/ to a temp dir with intents.json from before the keyword fix."""
    tmp = tempfile.mkdtemp()
    for name in os.listdir(KNOWLEDGE_DIR):
        src = os.path.join(KNOWLEDGE_DIR, name)
        if os.path.isfile(src):
            with open(src, "rb") as f, open(os.path.join(tmp, name), "wb") as g:
                g.write(f.read())
    old = subprocess.run(["git", "show", f"{OLD_INTENTS_COMMIT}:backend/knowledge/intents.json"],
                         capture_output=True, check=True, cwd=KNOWLEDGE_DIR).stdout
    with open(os.path.join(tmp, "intents.json"), "wb") as g:
        g.write(old)
    return Knowledge.load(knowledge_dir=tmp)


async def main(limit):
    raw = open(PROBES, "rb").read()
    probes = json.loads(raw)["probes"][:limit]
    current, old = Knowledge.load(), _knowledge_with_old_intents()

    async def call(messages, json_schema):
        return await call_llm(messages, max_tokens=1024, temperature=0.0, json_schema=json_schema)

    ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    prompt_hash = hashlib.sha256(canonicalizer.SYSTEM_PROMPT.encode()).hexdigest()[:12]
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"{_DEFAULT_MODEL.replace('/', '-')}__{hashlib.sha256(raw).hexdigest()[:12]}__{ts}.jsonl")
    results = []
    with open(out, "w", encoding="utf-8") as f:
        f.write(json.dumps({"meta": {"backend": LLM_BACKEND, "model": _DEFAULT_MODEL, "prompt_sha256_12": prompt_hash,
                                     "probes_sha256_12": hashlib.sha256(raw).hexdigest()[:12], "run": ts}}) + "\n")
        for p in probes:
            k = old if p.get("old_intents") else current
            m = match(k, p["q"])
            consulted = not (m["subjects"] or m["intents"] or m["organism_word"] or m["other_crop"])
            labels = topic_labels(k)
            raw_reply = {}

            async def recording_call(messages, json_schema):
                raw_reply["text"] = await call(messages, json_schema)
                return raw_reply["text"]

            topics = await canonicalizer.canonicalize(k, labels, p["q"], recording_call) if consulted else None
            r = answer(k, p["q"], topic_fallback=(lambda _q, _t=topics: _t) if topics is not None else None)
            refused = r["status"] != "answered"
            if p["expect"] == "refuse":
                ok, safety_issue = refused, not refused
            else:
                ok, safety_issue = (topics is not None and p["expect"] in topics), False
            rec = {"id": p["id"], "q": p["q"], "expect": p["expect"], "model_consulted": consulted,
                   "raw_reply": raw_reply.get("text"), "topics": topics, "status": r["status"],
                   "record_ids": r["record_ids"], "correct": ok, "answered_a_should_refuse": safety_issue}
            results.append(rec)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"{p['id']} expect={p['expect']:15s} topics={topics} -> {r['status']} {'OK' if ok else 'MISS'}"
                  f"{'  <-- ANSWERED A SHOULD-REFUSE' if safety_issue else ''}")

    ans = [x for x in results if x["expect"] != "refuse"]
    ref = [x for x in results if x["expect"] == "refuse"]
    print(f"\ntopic picked correctly: {sum(x['correct'] for x in ans)}/{len(ans)}"
          f" | should-refuse refused: {sum(x['correct'] for x in ref)}/{len(ref)}"
          f" | empty/invalid replies: {sum(1 for x in results if x['model_consulted'] and not x['topics'] and x['expect'] != 'refuse')}")
    print(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    asyncio.run(main(ap.parse_args().limit))
