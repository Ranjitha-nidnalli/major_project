"""
Optional LLM query canonicaliser for the v2 fact path (off by default).

Mentor's "hybrid middle ground": a model is used ONLY to understand the
question, never to write the answer. It maps a free-form Kannada question
that the keyword matcher did not recognise to zero or more TOPIC ids from
knowledge/intents.json (seed_rate, irrigation, ...). The deterministic
lookup in services/fact_answer.py stays the only source of text and doses.

Safety design (tests/invariants/test_canonicalizer.py):
  - Closed output: the label set is fact_answer.topic_labels(), which
    excludes pest/disease subjects, so the model can never select a pest
    dose. An unknown pest is already refused by the matcher before the
    model is ever asked (fact_answer.answer only consults the fallback when
    the matcher found nothing at all).
  - Constrained decoding where the backend supports it: json_schema() gives
    an enum over the labels, for providers with structured output.
  - Strict parsing regardless: parse_topics() accepts only a JSON object
    whose "topics" are all known labels. Anything else -> [] -> refusal.
  - Fail closed: any exception or empty reply is a refusal, not a guess.

Not yet evaluated against a live model. Enable with
ENABLE_LLM_CANONICALIZER=true in backend/.env only for experiments.
"""
import json

NONE_LABEL = "none"

SYSTEM_PROMPT = (
    "You classify a sugarcane farmer's question (usually in Kannada) into topic ids. "
    "Reply with JSON only: {\"topics\": [...]}. Use only ids from the list given. "
    "If no id clearly fits, or the question is about a pest, disease, insect, another "
    "crop, prices, weather or schemes, reply {\"topics\": [\"none\"]}. "
    "Never answer the question itself."
)


def describe_labels(knowledge, labels):
    """One line per label: id plus a few of its keywords, so the model knows what it means."""
    table = knowledge.intents["intents"]
    lines = []
    for label in labels:
        words = [w.lstrip("=") for w in table[label]["keywords"][:4]]
        lines.append(f"- {label}: {', '.join(words)}")
    return "\n".join(lines)


def build_messages(knowledge, labels, query):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"Topic ids:\n{describe_labels(knowledge, labels)}\n- {NONE_LABEL}: none of these\n\n"
            f"Question: {query}"
        )},
    ]


def json_schema(labels):
    """Schema for constrained decoding: the output can only be known ids or 'none'."""
    return {
        "type": "object",
        "properties": {
            "topics": {
                "type": "array",
                "items": {"type": "string", "enum": list(labels) + [NONE_LABEL]},
                "minItems": 1,
                "maxItems": 3,
            }
        },
        "required": ["topics"],
        "additionalProperties": False,
    }


def parse_topics(text, labels):
    """
    Strict: a JSON object with a 'topics' list of known ids. Any unknown id,
    extra text, or 'none' anywhere means no topics (refuse). Never repairs.
    """
    if not text:
        return []
    try:
        obj = json.loads(text.strip())
    except (ValueError, TypeError):
        return []
    if not isinstance(obj, dict) or set(obj) != {"topics"}:
        return []
    topics = obj["topics"]
    if not isinstance(topics, list) or not topics or len(topics) > 3:
        return []
    allowed = set(labels)
    if any(not isinstance(t, str) or t not in allowed for t in topics):
        return []          # includes "none" and anything invented
    return list(dict.fromkeys(topics))


async def canonicalize(knowledge, labels, query, call_llm):
    """
    Ask the model once. call_llm(messages, json_schema=...) -> str | None
    (llm_client.call_llm). Returns known topic ids or [] (refuse).
    """
    try:
        reply = await call_llm(build_messages(knowledge, labels, query),
                               json_schema=json_schema(labels))
    except Exception as e:
        print(f"⚠️ canonicalizer failed, refusing: {e}")
        return []
    topics = parse_topics(reply, labels)
    print(f"🧭 canonicalizer: {topics or 'none'} (raw={reply!r:.120})")
    return topics
