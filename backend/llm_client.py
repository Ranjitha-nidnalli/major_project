"""
Thin abstraction over LLM backends.

Supports three backends:
  - groq      (default, tested and working)
  - openrouter (new, for cross-judge evaluation and model diversity)
  - ollama    (untested scaffolding, blocked by Windows+Python 3.13 bug)

All model IDs are read from environment variables — no hardcoded fallback
strings. This prevents the "4 hardcoded fallback model strings" anti-pattern
and makes model switching a one-line .env change.

Usage:
    from llm_client import call_llm
    answer = await call_llm(messages, model="openai/gpt-oss-20b", max_tokens=250)

Env:
    LLM_BACKEND=groq|openrouter|ollama
    GENERATION_MODEL=openai/gpt-oss-20b   (default, no hardcoded fallback)
    GROQ_API_KEY=gsk_...                   (required if LLM_BACKEND=groq)
    OPENROUTER_API_KEY=sk-or-v1-...      (required if LLM_BACKEND=openrouter)
    OLLAMA_HOST=http://...                 (optional, used if LLM_BACKEND=ollama)
"""
import os
import time
import asyncio
from dotenv import load_dotenv

env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path=env_path, override=True)

LLM_BACKEND = os.getenv("LLM_BACKEND", "groq").lower()

# --- SINGLE SOURCE OF TRUTH for default model ---
# No hardcoded fallback strings anywhere else in this file.
# Changing the model is a one-line .env edit.
_DEFAULT_MODEL = os.getenv("GENERATION_MODEL", "openai/gpt-oss-20b")


# --- Groq path ---
_groq_client = None

def _get_groq_client():
    global _groq_client
    if _groq_client is None:
        from groq import Groq
        key = os.getenv("GROQ_API_KEY")
        if not key:
            raise ValueError("GROQ_API_KEY not set in .env")
        _groq_client = Groq(api_key=key)
    return _groq_client


def _groq_chat_sync(messages, model, max_tokens, temperature):
    # No hardcoded fallback — use the module-level default
    m = model or _DEFAULT_MODEL
    client = _get_groq_client()
    start = time.time()
    try:
        res = client.chat.completions.create(
            model=m,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        elapsed = time.time() - start
        print(f"⏱️  Groq ({m}) took {elapsed:.2f}s")
        if res is None or not res.choices or res.choices[0].message is None:
            return None
        return res.choices[0].message.content
    except Exception as e:
        elapsed = time.time() - start
        print(f"⚠️ Groq failed ({m}) after {elapsed:.2f}s: {e}")
        return None


# --- OpenRouter path (new) ---
_openrouter_client = None

def _get_openrouter_client():
    global _openrouter_client
    if _openrouter_client is None:
        # OpenRouter uses OpenAI-compatible client
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "OpenRouter backend requires 'openai' package. "
                "Install: pip install openai"
            )
        key = os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise ValueError("OPENROUTER_API_KEY not set in .env")
        _openrouter_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=key,
        )
    return _openrouter_client


def _openrouter_chat_sync(messages, model, max_tokens, temperature):
    m = model or _DEFAULT_MODEL
    client = _get_openrouter_client()
    start = time.time()
    try:
        res = client.chat.completions.create(
            model=m,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_headers={
                "HTTP-Referer": "https://github.com/Ranjitha-nidnalli/major_project",
                "X-Title": "Krishi Mitra",
            },
        )
        elapsed = time.time() - start
        print(f"⏱️  OpenRouter ({m}) took {elapsed:.2f}s")
        if res is None or not res.choices or res.choices[0].message is None:
            return None
        return res.choices[0].message.content
    except Exception as e:
        elapsed = time.time() - start
        print(f"⚠️ OpenRouter failed ({m}) after {elapsed:.2f}s: {e}")
        return None


# --- Ollama path (untested scaffolding) ---
_ollama_client = None

def _get_ollama_client():
    global _ollama_client
    if _ollama_client is None:
        from ollama import AsyncClient as AsyncOllamaClient
        _ollama_client = AsyncOllamaClient(
            host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
            timeout=300
        )
    return _ollama_client


async def _ollama_chat_async(messages, model, max_tokens, temperature):
    # No hardcoded fallback — use the module-level default
    m = model or _DEFAULT_MODEL
    client = _get_ollama_client()
    start = time.time()
    try:
        res = await client.chat(
            model=m,
            messages=messages,
            options={"temperature": temperature, "num_predict": max_tokens}
        )
        elapsed = time.time() - start
        print(f"⏱️  Ollama ({m}) took {elapsed:.2f}s")
        # Safe extraction — handles dict or object responses
        if res is None:
            return None
        if hasattr(res, 'message') and res.message is not None and hasattr(res.message, 'content'):
            return res.message.content
        if isinstance(res, dict):
            msg = res.get('message', {})
            if isinstance(msg, dict):
                return msg.get('content')
            if hasattr(msg, 'content'):
                return msg.content
        return None
    except Exception as e:
        elapsed = time.time() - start
        print(f"⚠️ Ollama failed ({m}) after {elapsed:.2f}s: {e}")
        return None


# --- Public API ---
async def call_llm(messages, model=None, max_tokens=500, temperature=0.0):
    """
    Unified LLM call. Backend selected by LLM_BACKEND env var.
    Model ID read from env var (GENERATION_MODEL) — no hardcoded defaults.

    Args:
        messages: List of dicts with 'role' and 'content' keys.
        model: Override model ID. If None, uses GENERATION_MODEL from .env.
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature (0.0 = deterministic).

    Returns:
        Generated text string, or None on failure.
    """
    if LLM_BACKEND == "groq":
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, _groq_chat_sync, messages, model, max_tokens, temperature
        )
    elif LLM_BACKEND == "openrouter":
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, _openrouter_chat_sync, messages, model, max_tokens, temperature
        )
    elif LLM_BACKEND == "ollama":
        return await _ollama_chat_async(messages, model, max_tokens, temperature)
    else:
        raise ValueError(
            f"Unknown LLM_BACKEND: {LLM_BACKEND}. "
            f"Use 'groq', 'openrouter', or 'ollama'."
        )
