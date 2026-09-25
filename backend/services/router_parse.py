"""
services/router_parse.py

Parses the LLM router's reply into a category. Pure function, no I/O.

Returns None when the reply holds no valid category, so the caller can
retry or fall back explicitly. Verified 2026-09-25: with max_tokens=100,
gpt-oss-20b (a reasoning model) spent the budget on hidden reasoning and
returned empty content for 14 of 18 eval questions, which the pipeline
silently treated as "general".
"""
from typing import Optional

ROUTER_CATEGORIES = frozenset({"price", "disease", "pest", "fertilizer", "general"})


def parse_router_category(content: Optional[str]) -> Optional[str]:
    if not content or not content.strip():
        return None
    # Tolerate the old "CATEGORY | keywords" format and stray quotes/punctuation.
    first = content.strip().split("|")[0].strip().strip("'\"`.*").strip().lower()
    return first if first in ROUTER_CATEGORIES else None
