"""
services/judge_parse.py

Parses the faithfulness judge's reply into a score (TODO #48). Pure
function, no I/O, so it can be tested without loading the pipeline.

Returns None -- "the judge produced no usable score" -- rather than 0.0
for empty, unparseable, or out-of-range replies, so the caller can tell
a judge failure apart from a genuine "unfaithful" verdict and retry.
Verified 2026-09-25: gpt-oss-20b (a reasoning model) sometimes spends the
whole token budget on hidden reasoning and returns empty content
(finish_reason=length, 498/500 tokens reasoning), which the old parser
silently scored 0.0.
"""
import re
from typing import Optional

_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


def parse_judge_score(content: Optional[str]) -> Optional[float]:
    if not content or not content.strip():
        return None
    text = content.strip()
    try:
        score = float(text)
    except ValueError:
        # Tolerate a short wrapper like "Score: 0.8", but only when the
        # reply holds exactly one number -- otherwise there is no telling
        # which number is the score.
        numbers = _NUMBER_RE.findall(text)
        if len(numbers) != 1:
            return None
        score = float(numbers[0])
    if not 0.0 <= score <= 1.0:
        return None
    return score
