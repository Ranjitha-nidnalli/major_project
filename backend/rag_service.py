import os
import asyncio
import traceback
from dotenv import load_dotenv

# --- Environment Setup ---
env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path=env_path, override=True)

import torch
from groq import Groq
from tavily import TavilyClient

from qdrant_client import models
from vector_db import db_client, COLLECTION_NAME, embed_model, reranker_model
from chat_db import save_chat_message, get_chat_history

# --- Configuration ---
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "llama-3.1-8b-instant")
INTERACTIVE_MAX_PREDICT = int(os.getenv("INTERACTIVE_MAX_PREDICT", 150))
EVAL_MAX_PREDICT = int(os.getenv("EVAL_MAX_PREDICT", 500))
INTERACTIVE_TIMEOUT = int(os.getenv("INTERACTIVE_TIMEOUT", 120))

HARD_REFUSAL_THRESHOLD = 0.35
CONFIDENT_SEARCH_THRESHOLD = 0.5
FAITHFULNESS_GATE_THRESHOLD = 0.50

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not set in .env")

groq_client = Groq(api_key=GROQ_API_KEY)

print(f"[Krishi Mitra] Loaded with GENERATION_MODEL={GENERATION_MODEL} (Groq)")

ENABLE_WEB_FALLBACK = os.getenv("ENABLE_WEB_FALLBACK", "false").lower() == "true"
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY")) if ENABLE_WEB_FALLBACK else None

ENABLE_RERANKER = os.getenv("ENABLE_RERANKER", "false").lower() == "true"
RERANK_THRESHOLD = 0.5

# --- System Prompts & Messages ---
SYSTEM_INSTRUCTION = (
    "You are a strict data extractor. Do not repeat the question. "
    "Answer using only the provided context in clean, short Kannada bullet points. "
    "DO NOT add any outside knowledge, facts, or scientific explanations that are not explicitly written in the context. "
    "If the context lists 3 soil types, list exactly those 3. "
    "Use the exact Kannada terminology found in the context. "
    "If the answer is not there, say: \"ಕ್ಷಮಿಸಿ, ಈ ಮಾಹಿತಿ ನಮ್ಮ ಡೇಟಾಬೇಸ್ನಲ್ಲಿ ಲಭ್ಯವಿಲ್ಲ.\""
)

HARD_REFUSAL_MESSAGE = (
    "ಕ್ಷಮಿಸಿ, ನಿಮ್ಮ ಪ್ರಶ್ನೆಗೆ ಸಂಬಂಧಿಸಿದ ಮಾಹಿತಿ ಲಭ್ಯವಿಲ್ಲ. "
    "ದಯವಿಟ್ಟು ಬೇರೆ ರೀತಿಯಲ್ಲಿ ಕೇಳಿ ಪ್ರಯತ್ನಿಸಿ. "
    "ಹೆಚ್ಚಿನ ಸಹಾಯಕ್ಕಾಗಿ, ಕಿಸಾನ್ ಕಾಲ್ ಸೆಂಟರ್ಗೆ ಕರೆ ಮಾಡಿ: 1800-180-1551."
)

TIMEOUT_MESSAGE = (
    "ಕ್ಷಮಿಸಿ, ಉತ್ತರವನ್ನು ನೀಡಲು ಹೆಚ್ಚು ಸಮಯ ತೆಗೆದುಕೊಳ್ಳುತ್ತಿದೆ. "
    "ದಯವಿಟ್ಟು ಸ್ವಲ್ಪ ಸಮಯದ ನಂತರ ಮತ್ತೆ ಪ್ರಯತ್ನಿಸಿ."
)

ESCALATION_MESSAGE = (
    "\n\nಹೆಚ್ಚಿನ ಸಹಾಯಕ್ಕಾಗಿ, ಕಿಸಾನ್ ಕಾಲ್ ಸೆಂಟರ್ಗೆ ಕರೆ ಮಾಡಿ: 1800-180-1551."
)

EMPTY_ANSWER_FALLBACK_MESSAGE = (
    "ಕ್ಷಮಿಸಿ, ಅಗತ್ಯಮಾಧ್ಯಮ ಮಾಹಿತಿ ಇಲ್ಲದ ಕಾರಣ ಸೂಕ್ತ ಉತ್ತರವನ್ನು ನೀಡಲಾಗುತ್ತಿಲ್ಲ. "
    "ಹೆಚ್ಚಿನ ಸಹಾಯಕ್ಕೆ ಕರೆಮಾಡಿ: ರೈತ ಸಹಾಯ ಕೇಂದ್ರ — 1800-180-1551"
)

SAFETY_CRITICAL_CATEGORIES = {"pest", "disease", "fertilizer"}


import json
import time


def _groq_chat(messages, model=None, max_tokens=500, temperature=0.0):
    """
    Synchronous Groq call.
    """
    m = model or GENERATION_MODEL
    kwargs = {
        "model": m,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    start = time.time()
    try:
        res = groq_client.chat.completions.create(**kwargs)
        elapsed = time.time() - start
        print(f"⏱️  Groq ({m}) took {elapsed:.2f}s")
        if res is None:
            print("⚠️ Groq returned None response object")
            return None
        if not res.choices:
            print("⚠️ Groq returned empty choices")
            return None
        msg = res.choices[0].message
        if msg is None:
            print("⚠️ Groq returned None message")
            return None
        return msg.content
    except Exception as e:
        elapsed = time.time() - start
        print(f"⚠️ Groq failed ({m}) after {elapsed:.2f}s: {e}")
        traceback.print_exc()
        return None


async def safe_groq_chat(messages, model=None, max_tokens=500, temperature=0.0):
    """Async wrapper around sync Groq client."""
    try:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            _groq_chat,
            messages,
            model,
            max_tokens,
            temperature
        )
    except Exception as e:
        print(f"⚠️ safe_groq_chat wrapper failed: {e}")
        traceback.print_exc()
        return None


async def calculate_faithfulness(context: str, answer: str, judge_model: str = None) -> float:
    if not answer or not context:
        return 0.0
    prompt = (
        "You are a professional grader. Rate the faithfulness of the Answer based ONLY on the provided Context. "
        "If the answer contains info not in the context, penalize it. "
        "Return ONLY a number between 0.0 and 1.0. Nothing else."
    )
    user_msg = f"Context:\n{context}\n\nAnswer:\n{answer}"
    content = await safe_groq_chat(
        messages=[{"role": "system", "content": prompt}, {"role": "user", "content": user_msg}],
        model=judge_model or GENERATION_MODEL,
        max_tokens=50,
        temperature=0.0
    )
    if content is None:
        return 0.0
    try:
        # Extract first float-like thing from the response
        content = content.strip()
        # Try to parse as float directly
        return float(content)
    except ValueError:
        # Try to extract a number from the text
        import re
        match = re.search(r"(\d+\.\d+|\d+)", content)
        if match:
            return float(match.group(1))
        print(f"⚠️ Judge returned non-numeric: {content}")
        return 0.0
    except Exception as e:
        print(f"⚠️ Judge parse failed: {e}")
        return 0.0


async def generate_from_context(
    user_query: str,
    context_text: str,
    session_id: str,
    generation_model: str = None,
    judge_model: str = None,
    save_to_db: bool = True,
    run_judge: bool = True,
    max_predict_tokens: int = None
):
    try:
        model = generation_model or GENERATION_MODEL
        max_tokens = max_predict_tokens or EVAL_MAX_PREDICT

        history = await get_chat_history(session_id, limit=3) if save_to_db else []
        history_text = "\n".join([f"{m['role']}: {m['content']}" for m in history])

        final_prompt = f"History:\n{history_text}\n\nContext:\n{context_text}\n\nQuestion: {user_query}"

        print("🔄 Calling Groq for generation...")
        content = await safe_groq_chat(
            messages=[
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "user", "content": final_prompt}
            ],
            model=model,
            max_tokens=max_tokens,
            temperature=0.0
        )
        print(f"🔄 Groq generation result: {content[:50] if content else 'None'}...")

        if content is None:
            print("⚠️ Groq returned None. Using fallback refusal.")
            ans = EMPTY_ANSWER_FALLBACK_MESSAGE
        elif not content.strip():
            print("⚠️ Model returned empty answer. Using fallback.")
            ans = EMPTY_ANSWER_FALLBACK_MESSAGE
        else:
            ans = content.strip()

        if save_to_db:
            await save_chat_message(session_id, "user", user_query)
            await save_chat_message(session_id, "assistant", ans)

        accuracy_score = None
        if run_judge:
            print("🔄 Calling Groq for faithfulness judge...")
            accuracy_score = await calculate_faithfulness(context_text, ans, judge_model=judge_model)
            print(f"🔄 Judge score: {accuracy_score}")

        return {"answer": ans, "accuracy_score": accuracy_score}
    except Exception as e:
        print(f"❌ ERROR inside generate_from_context: {e}")
        traceback.print_exc()
        raise


async def get_sugarcane_answer(user_query: str, session_id: str, return_context: bool = False, interactive: bool = True):
    # ==========================================
    # 1. THE SWITCHBOARD (Router)
    # ==========================================
    router_prompt = (
        "Classify the user query into: 'price', 'disease', 'pest', 'fertilizer', or 'general'.\n"
        "Also translate the query to English keywords for web search purposes.\n"
        "Format: CATEGORY | ENGLISH_KEYWORDS"
    )

    route_content = await safe_groq_chat(
        messages=[{"role": "system", "content": router_prompt}, {"role": "user", "content": user_query}],
        max_tokens=100,
        temperature=0.0
    )

    if route_content:
        route_parts = route_content.strip().split('|')
        category = route_parts[0].strip().lower()
        english_search_query = route_parts[1].strip() if len(route_parts) > 1 else user_query
    else:
        print("⚠️ Router failed, defaulting to general")
        category, english_search_query = "general", user_query

    # ==========================================
    # 2. RETRIEVAL WITH BGE-M3 HYBRID
    # ==========================================
    print(f"🔍 Searching DB for: {user_query}")

    async def get_vectors(q_text):
        output = await asyncio.to_thread(embed_model.encode, [q_text], return_dense=True, return_sparse=True)
        dense_vec = output['dense_vecs'][0].tolist()
        lex_weights = output['lexical_weights'][0]
        sp_indices = [int(k) for k in lex_weights.keys()]
        sp_values = [float(v) for v in lex_weights.values()]
        return dense_vec, sp_indices, sp_values

    dense_vec, sparse_indices, sparse_values = await get_vectors(user_query)

    async def execute_weighted_search(d_vec, s_idx, s_val):
        response = await asyncio.to_thread(
            db_client.query_points,
            collection_name=COLLECTION_NAME,
            prefetch=[
                models.Prefetch(query=d_vec, using="dense", limit=15),
                models.Prefetch(query=models.SparseVector(indices=s_idx, values=s_val), using="sparse", limit=15),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=5,
        )
        hits = response.points
        fused = [{"score": h.score, "payload": h.payload} for h in hits]
        return fused, hits[0].score if hits else 0.0

    top_chunks, best_fused_score = await execute_weighted_search(dense_vec, sparse_indices, sparse_values)

    # ==========================================
    # 3. SAFETY & CONFIDENCE CHECKS
    # ==========================================
    docs = [p["payload"]["text"] for p in top_chunks]
    search_score = top_chunks[0]['score'] if top_chunks else 0.0

    is_low_confidence = not docs or search_score < HARD_REFUSAL_THRESHOLD
    is_medium_confidence = search_score < CONFIDENT_SEARCH_THRESHOLD
    is_safety_critical = category in SAFETY_CRITICAL_CATEGORIES

    # Hard Refusal: low retrieval confidence
    if is_low_confidence:
        print(f"🟥 Hard Refusal: Low confidence score ({search_score:.2f}).")
        return {
            "answer": HARD_REFUSAL_MESSAGE,
            "search_score": search_score,
            "accuracy_score": 0.0,
            "context": "\n\n".join(docs) if return_context else None,
        }

    # For safety-critical categories, require higher confidence
    if is_safety_critical and is_medium_confidence:
        print(f"🟧 Safety Refusal: {category} query with medium confidence ({search_score:.2f}).")
        return {
            "answer": HARD_REFUSAL_MESSAGE,
            "search_score": search_score,
            "accuracy_score": 0.0,
            "context": "\n\n".join(docs) if return_context else None,
        }

    print(f"🟢 DB Hit! Best Score: {search_score:.2f}")
    context_text = "\n\n".join([f"{doc}" for doc in docs])

    # ==========================================
    # 4. MEMORY & GENERATION
    # ==========================================
    ans = ""
    accuracy_score = 0.0
    try:
        timeout = INTERACTIVE_TIMEOUT if interactive else 900
        max_tokens = INTERACTIVE_MAX_PREDICT if interactive else EVAL_MAX_PREDICT

        print("🔄 Entering generate_from_context...")
        gen_result = await asyncio.wait_for(
            generate_from_context(
                user_query, context_text, session_id,
                run_judge=False, max_predict_tokens=max_tokens
            ),
            timeout=timeout
        )
        print(f"🔄 generate_from_context returned: {type(gen_result)}")
        ans = gen_result["answer"]
        accuracy_score = gen_result.get("accuracy_score", 0.0)

        # Faithfulness gate — reject hallucinated answers
        if accuracy_score is not None and accuracy_score < FAITHFULNESS_GATE_THRESHOLD:
            print(f"🟨 Faithfulness gate triggered ({accuracy_score:.2f} < {FAITHFULNESS_GATE_THRESHOLD}). Returning refusal.")
            ans = HARD_REFUSAL_MESSAGE
            accuracy_score = 0.0
        else:
            # Append escalation line if confidence is not high
            if is_medium_confidence and ans:
                ans += ESCALATION_MESSAGE

    except asyncio.TimeoutError:
        print(f"⏰ Timeout error after {timeout}s!")
        ans = TIMEOUT_MESSAGE
        accuracy_score = 0.0
    except Exception as e:
        print(f"❌ Unexpected pipeline error: {e}")
        traceback.print_exc()
        ans = HARD_REFUSAL_MESSAGE
        accuracy_score = 0.0

    response_data = {
        "answer": ans,
        "search_score": search_score,
        "accuracy_score": accuracy_score,
    }

    if return_context:
        response_data["context"] = context_text
    return response_data
