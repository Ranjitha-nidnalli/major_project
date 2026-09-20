import os
import asyncio
from dotenv import load_dotenv

# --- Environment Setup ---
env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path=env_path, override=True)

import torch

from qdrant_client import models
from vector_db import db_client, COLLECTION_NAME, embed_model, reranker_model
from chat_db import save_chat_message, get_chat_history
from llm_client import call_llm  # NEW: unified LLM abstraction
from eval.numeric_faithfulness import check_numeric_faithfulness  # #5
from services.gating import decide as gating_decide, GatingConfig

# --- Configuration ---
GENERATION_MODEL = os.getenv("GENERATION_MODEL")  # No fallback — must be set in .env
INTERACTIVE_MAX_PREDICT = int(os.getenv("INTERACTIVE_MAX_PREDICT", 250))
EVAL_MAX_PREDICT = int(os.getenv("EVAL_MAX_PREDICT", 500))
INTERACTIVE_TIMEOUT = int(os.getenv("INTERACTIVE_TIMEOUT", 120))

# HARD_REFUSAL_THRESHOLD removed: the abstention gate now lives in
# services/gating.py and is driven by dense cosine relevance, not the RRF
# fusion score. See GATING_CONFIG below and ARCHITECTURE.md Section 21.
CONFIDENT_SEARCH_THRESHOLD = 0.5  # UNCALIBRATED: only gates the escalation-line UX, not refusal
FAITHFULNESS_GATE_THRESHOLD = 0.50

SAFETY_CRITICAL_CATEGORIES = {"pest", "disease", "fertilizer"}

# Abstention gate config (Task 2). Thresholds are UNCALIBRATED placeholders;
# Phase 2 replaces them via backend/eval/threshold_sweep.py against a real
# labelled answerable/unanswerable set. Do not report these as validated.
GATING_CONFIG = GatingConfig(safety_critical_categories=frozenset(SAFETY_CRITICAL_CATEGORIES))

print(f"[Krishi Mitra] Loaded with GENERATION_MODEL={GENERATION_MODEL}")

# #37: Tavily web fallback removed — was dead code (instantiated but never called).
# If web fallback is needed in future, re-implement with a real search call,
# not an orphaned client instantiation.
ENABLE_WEB_FALLBACK = False

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


import json
import time


async def calculate_faithfulness(context: str, answer: str, judge_model: str = None) -> float:
    if not answer or not context:
        return 0.0
    prompt = (
        "You are a professional grader. Rate the faithfulness of the Answer based ONLY on the provided Context. "
        "If the answer contains info not in the context, penalize it. "
        "Return ONLY a number between 0.0 and 1.0. Nothing else."
    )
    user_msg = f"Context:\n{context}\n\nAnswer:\n{answer}"
    content = await call_llm(
        messages=[{"role": "system", "content": prompt}, {"role": "user", "content": user_msg}],
        model=judge_model or GENERATION_MODEL,
        max_tokens=50,
        temperature=0.0
    )
    if content is None:
        return 0.0
    try:
        content = content.strip()
        return float(content)
    except ValueError:
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

        print("🔄 Calling LLM for generation...")
        content = await call_llm(
            messages=[
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "user", "content": final_prompt}
            ],
            model=model,
            max_tokens=max_tokens,
            temperature=0.0
        )
        print(f"🔄 LLM generation result: {content[:50] if content else 'None'}...")

        if content is None:
            print("⚠️ LLM returned None. Using fallback refusal.")
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
            print("🔄 Calling LLM for faithfulness judge...")
            accuracy_score = await calculate_faithfulness(context_text, ans, judge_model=judge_model)
            print(f"🔄 Judge score: {accuracy_score}")

        return {"answer": ans, "accuracy_score": accuracy_score}
    except Exception as e:
        print(f"❌ ERROR inside generate_from_context: {e}")
        import traceback
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

    route_content = await call_llm(
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

    # Use embed_query wrapper for Indic NLP normalization before embedding
    from vector_db import embed_query
    dense_vec, sparse_indices, sparse_values = await asyncio.to_thread(embed_query, user_query)

    # Category is intentionally NOT used as a retrieval filter.
    #
    # Verified defect (ARCHITECTURE.md Section 12): a Qdrant Filter(should=[...])
    # with no other clauses is NOT a soft boost — it hard-restricts the result set
    # to matching points only. Because the LLM router can misclassify a query,
    # this made gold chunks unreachable whenever the router got the category wrong
    # (e.g. a seed-treatment question routed as "general" while its chunk lives in
    # "disease" was completely unretrievable).
    #
    # Fix applied (Option A, Section 12): remove the filter entirely and rely on
    # pure hybrid (dense + sparse, RRF-fused) retrieval. `category` is still
    # produced by the router and used below as a signal for the safety-critical
    # confidence gate (SAFETY_CRITICAL_CATEGORIES) — an LLM classification must
    # never be allowed to make a document unreachable, but it may still inform
    # how cautious the gate is.
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

    async def get_dense_relevance(d_vec):
        """
        Max dense cosine similarity over the retrieved cards, for the
        abstention gate (services/gating.py). This is a plain dense-only
        query -- no fusion, no filter -- against the same collection.
        The dense vector space is configured with Distance.COSINE
        (vector_db.py), so the returned score is a genuine cosine
        similarity, unlike the rank-derived RRF fusion score.
        """
        response = await asyncio.to_thread(
            db_client.query_points,
            collection_name=COLLECTION_NAME,
            query=d_vec,
            using="dense",
            limit=1,
        )
        hits = response.points
        return hits[0].score if hits else None

    top_chunks, best_fused_score = await execute_weighted_search(dense_vec, sparse_indices, sparse_values)

    docs = [p["payload"]["text"] for p in top_chunks]
    # search_score (RRF fusion score) is kept only for logging/telemetry and
    # for the ChatResponse payload the frontend already expects. It must
    # NEVER be used as the relevance signal for the abstention gate -- RRF
    # scores are rank-derived, so a top-ranked-but-irrelevant hit can score
    # 1.0 (verified defect, ARCHITECTURE.md Section 21).
    search_score = top_chunks[0]['score'] if top_chunks else 0.0

    # ==========================================
    # 3. ABSTENTION GATE (Task 2 / ARCHITECTURE.md Section 21)
    # ==========================================
    # Relevance = max dense cosine similarity over the retrieved cards,
    # computed via a plain dense-only query (no fusion, no filter). The
    # dense vector space is configured with COSINE distance in vector_db.py,
    # so this score is a genuine cosine similarity, unlike the RRF score.
    dense_relevance = await get_dense_relevance(dense_vec) if docs else None

    gate_decision = gating_decide(relevance=dense_relevance, category=category, config=GATING_CONFIG)

    if not gate_decision.should_answer:
        print(f"🟥 Gate refusal ({gate_decision.reason}).")
        return {
            "answer": HARD_REFUSAL_MESSAGE,
            "search_score": search_score,
            "relevance": dense_relevance,
            "accuracy_score": 0.0,
            "context": "\n\n".join(docs) if return_context else None,
        }

    # Only used for the escalation-line UX below (not a refusal decision).
    is_medium_confidence = dense_relevance is not None and dense_relevance < CONFIDENT_SEARCH_THRESHOLD

    print(f"🟢 DB Hit! Relevance (dense cosine): {dense_relevance:.2f} | RRF score (telemetry only): {search_score:.2f}")
    context_text = "\n\n".join([f"{doc}" for doc in docs])

    # ==========================================
    # 4. MEMORY & GENERATION
    # ==========================================
    ans = ""
    accuracy_score = 0.0
    numeric_score = 1.0
    numeric_violations = []
    try:
        timeout = INTERACTIVE_TIMEOUT if interactive else 900
        max_tokens = INTERACTIVE_MAX_PREDICT if interactive else EVAL_MAX_PREDICT

        print("🔄 Entering generate_from_context...")
        gen_result = await asyncio.wait_for(
            generate_from_context(
                user_query, context_text, session_id,
                run_judge=interactive, max_predict_tokens=max_tokens
            ),
            timeout=timeout
        )
        print(f"🔄 generate_from_context returned: {type(gen_result)}")
        ans = gen_result["answer"]
        accuracy_score = gen_result.get("accuracy_score", 0.0)

        # #5: Numeric faithfulness check — cross-reference numbers in answer against context
        if ans and ans != EMPTY_ANSWER_FALLBACK_MESSAGE and ans != HARD_REFUSAL_MESSAGE:
            print("🔄 Running numeric faithfulness check...")
            numeric_score, numeric_violations = check_numeric_faithfulness(context_text, ans, strict=True)
            print(f"🔄 Numeric score: {numeric_score}, violations: {len(numeric_violations)}")
            if numeric_violations:
                for v in numeric_violations:
                    print(f"   ⚠️  Numeric violation: {v['answer_number']} {v['unit']} (severity: {v['severity']})")

        # Combined gate: semantic OR numeric failure triggers refusal
        semantic_fail = accuracy_score is not None and accuracy_score < FAITHFULNESS_GATE_THRESHOLD
        numeric_fail = numeric_score < 1.0

        if semantic_fail or numeric_fail:
            reason = []
            if semantic_fail:
                reason.append(f"semantic ({accuracy_score:.2f} < {FAITHFULNESS_GATE_THRESHOLD})")
            if numeric_fail:
                reason.append(f"numeric ({len(numeric_violations)} violation(s))")
            print(f"🟨 Faithfulness gate triggered ({', '.join(reason)}). Returning refusal.")
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
        import traceback
        traceback.print_exc()
        ans = HARD_REFUSAL_MESSAGE
        accuracy_score = 0.0

    response_data = {
        "answer": ans,
        "search_score": search_score,
        "accuracy_score": accuracy_score,
    }

    # Include numeric check details in debug/return mode
    if return_context:
        response_data["context"] = context_text
        response_data["numeric_faithfulness"] = {
            "score": numeric_score,
            "violations": numeric_violations,
        }

    return response_data
