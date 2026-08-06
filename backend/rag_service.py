import os
import asyncio
from dotenv import load_dotenv

env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path=env_path, override=True)

import torch
from ollama import AsyncClient as AsyncOllamaClient
from tavily import TavilyClient

from qdrant_client import models
from vector_db import db_client, COLLECTION_NAME, embed_model, reranker_model
from chat_db import save_chat_message, get_chat_history

# Initialize Clients
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "gemma4:e4b")
# timeout guards against a model that never emits a stop token (e.g. a base,
# non-instruction-tuned checkpoint) generating until the context limit -
# observed taking 10+ minutes on a 2.5B model on CPU. num_predict below is
# the primary guard; this is a hard backstop.
ollama_client = AsyncOllamaClient(host=os.getenv("OLLAMA_HOST", "http://localhost:11434"), timeout=300)

# Caps generation length so a runaway/non-stopping model fails fast instead of
# rambling to its context limit (same root cause as the ollama_client timeout
# above).
MAX_PREDICT_TOKENS = 500

ENABLE_WEB_FALLBACK = os.getenv("ENABLE_WEB_FALLBACK", "false").lower() == "true"
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY")) if ENABLE_WEB_FALLBACK else None

# Default off: on this single-crop corpus, first-stage hybrid retrieval already
# reaches recall@10=1.0 (see backend/eval/PROJECT_PLAN.md P1.2), so the reranker's
# ~30-40x latency cost per query buys ordering gains (MRR/nDCG), not recall. Worth
# re-benchmarking once the corpus covers multiple crops and recall is no longer
# saturated.
ENABLE_RERANKER = os.getenv("ENABLE_RERANKER", "false").lower() == "true"

import json

async def calculate_faithfulness(context: str, answer: str, judge_model: str = None) -> float:
    prompt = (
        "You are a professional grader. Rate the faithfulness of the Answer based ONLY on the provided Context. "
        "If the answer contains info not in the context, penalize it. "
        "Return ONLY a valid JSON object in the exact following format:\n"
        '{"score": 0.8}'
    )
    user_msg = f"Context:\n{context}\n\nAnswer:\n{answer}"
    try:
        res = await ollama_client.chat(
            model=judge_model or GENERATION_MODEL,
            messages=[{"role": "system", "content": prompt}, {"role": "user", "content": user_msg}],
            format="json",
            options={"temperature": 0.0, "num_predict": MAX_PREDICT_TOKENS}
        )
        data = json.loads(res.message.content.strip())
        return float(data.get("score", 0.0))
    except Exception as e:
        print(f"⚠️ Judge failed: {e}")
        return 0.0


SYSTEM_INSTRUCTION = (
    "You are a strict data extractor. Do not repeat the question. "
    "Answer using only the provided context in clean, short Kannada bullet points. "
    "DO NOT add any outside knowledge, facts, or scientific explanations that are not explicitly written in the context. "
    "If the context lists 3 soil types, list exactly those 3. "
    "Use the exact Kannada terminology found in the context. "
    "If the answer is not there, say: \"ಕ್ಷಮಿಸಿ, ಈ ಮಾಹಿತಿ ನಮ್ಮ ಡೇಟಾಬೇಸ್ನಲ್ಲಿ ಲಭ್ಯವಿಲ್ಲ.\""
)


async def generate_from_context(
    user_query: str,
    context_text: str,
    session_id: str,
    generation_model: str = None,
    judge_model: str = None,
    save_to_db: bool = True,
    run_judge: bool = True,
):
    """
    The generation + faithfulness-judge phase of the pipeline, factored out so
    it can be replayed against a frozen retrieval context (see
    backend/eval/build_contexts.py + run_eval.py's P2 generation comparison)
    with a specific generation_model/judge_model, independent of whatever
    GENERATION_MODEL the live retrieval pipeline is currently configured with.

    run_judge=False skips the faithfulness call entirely (accuracy_score=None)
    so a batch of generations can run back-to-back on one model without
    Ollama swapping in the judge model between every question - see
    eval/run_eval.py, which batches all generations, then all judging,
    to avoid repeated multi-GB model reloads.
    """
    model = generation_model or GENERATION_MODEL

    history = await get_chat_history(session_id, limit=3) if save_to_db else []
    history_text = "\n".join([f"{m['role']}: {m['content']}" for m in history])

    final_prompt = f"History:\n{history_text}\n\nContext:\n{context_text}\n\nQuestion: {user_query}"

    try:
        final_response = await ollama_client.chat(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "user", "content": final_prompt}
            ],
            options={"temperature": 0.0, "num_predict": MAX_PREDICT_TOKENS}
        )
        ans = final_response.message.content
    except Exception as e:
        ans = f"ದೋಷ ಎದುರಾಗಿದೆ: {e}"

    if save_to_db:
        await save_chat_message(session_id, "user", user_query)
        await save_chat_message(session_id, "assistant", ans)

    accuracy_score = await calculate_faithfulness(context_text, ans, judge_model=judge_model) if run_judge else None
    return {"answer": ans, "accuracy_score": accuracy_score}


async def get_sugarcane_answer(user_query: str, session_id: str, return_context: bool = False):
    # ==========================================
    # 1. THE SWITCHBOARD (Router)
    # ==========================================
    # We keep the router ONLY to detect "price" or to get English for Web Search
    router_prompt = (
        "Classify the user query into: 'price', 'disease', 'pest', 'fertilizer', or 'general'.\n"
        "Also translate the query to English keywords for web search purposes.\n"
        "Format: CATEGORY | ENGLISH_KEYWORDS"
    )
    
    try:
        routing_response = await ollama_client.chat(
            model=GENERATION_MODEL,
            messages=[{"role": "system", "content": router_prompt}, {"role": "user", "content": user_query}],
            options={"temperature": 0.0, "num_predict": MAX_PREDICT_TOKENS}
        )
        route_parts = routing_response.message.content.strip().split('|')
        category = route_parts[0].strip().lower()
        english_search_query = route_parts[1].strip() if len(route_parts) > 1 else user_query
    except Exception as e:
        category, english_search_query = "general", user_query

    # ==========================================
    # 2. RETRIEVAL WITH BGE-M3 HYBRID
    # ==========================================
    print(f"🔍 Searching DB for: {user_query}")
    
    # Generate Dense and Sparse vectors
    async def get_vectors(q_text):
        output = await asyncio.to_thread(embed_model.encode, [q_text], return_dense=True, return_sparse=True)
        dense_vec = output['dense_vecs'][0].tolist()
        lex_weights = output['lexical_weights'][0]
        sp_indices = [int(k) for k in lex_weights.keys()]
        sp_values = [float(v) for v in lex_weights.values()]
        return dense_vec, sp_indices, sp_values

    dense_vec, sparse_indices, sparse_values = await get_vectors(user_query)

    async def execute_weighted_search(d_vec, s_idx, s_val):
        # Native Qdrant Reciprocal Rank Fusion: both prefetches (dense + sparse, top 15
        # each) run server-side in a single call, and Qdrant fuses ranks internally
        # (RRF, default k=60) - no manual score math or normalization needed.
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

    # Query Expansion Fallback if the fused match is weak.
    # Qdrant's RRF returns rank-based scores (max ~0.033 for a rank-1-in-both hit with
    # the default k=60), not a 0-1 cosine similarity, so the old 0.4 threshold doesn't
    # translate directly. This threshold roughly means "best hit wasn't in the top 10
    # of either ranker".
    FUSION_SCORE_THRESHOLD = 1.0 / 70  # ~= rank 10 with k=60
    if best_fused_score < FUSION_SCORE_THRESHOLD and category != "general":
        print("💡 Semantic score < 0.4. Executing Query Expansion...")
        try:
            expansion_prompt = f"Provide 3 related Kannada synonyms or technical keywords for this query: '{user_query}'. Return ONLY a comma separated list."
            exp_res = await ollama_client.chat(
                model=GENERATION_MODEL, messages=[{"role": "user", "content": expansion_prompt}],
                options={"temperature": 0.3, "num_predict": MAX_PREDICT_TOKENS}
            )
            synonyms = exp_res.message.content.strip()
            expanded_query = f"{user_query} {synonyms}"
            
            e_dense, e_s_idx, e_s_val = await get_vectors(expanded_query)
            top_chunks, _ = await execute_weighted_search(e_dense, e_s_idx, e_s_val)
        except Exception as e:
            print(f"⚠️ Query Expansion failed: {e}")

    # Cross-Encoder Reranker (Closer) - off by default, see ENABLE_RERANKER above.
    RERANK_THRESHOLD = 0.5  # sigmoid probability cutoff for relevance

    if ENABLE_RERANKER and top_chunks:
        print(f"🕵️‍♂️ Reranking {len(top_chunks)} chunks...")
        pairs = [[user_query, c["payload"]["text"]] for c in top_chunks]
        rerank_scores = await asyncio.to_thread(reranker_model.predict, pairs, activation_fn=torch.nn.Sigmoid())
        ranked = sorted(zip(top_chunks, rerank_scores), key=lambda pair: pair[1], reverse=True)
        top_chunks = [chunk for chunk, score in ranked if score >= RERANK_THRESHOLD]

    docs = [p["payload"]["text"] for p in top_chunks]
    
    # If no docs found, fallback to web
    is_low_confidence = not docs

    # ==========================================
    # 3. WEB FALLBACK (Tavily)
    # ==========================================
    used_web_search = False
    if ENABLE_WEB_FALLBACK and (category == "price" or is_low_confidence):
        print("🌐 Falling back to Web Search...")
        used_web_search = True
        try:
            # Use the ENGLISH query for web search (Tavily works better with English)
            web_results = await asyncio.to_thread(
                tavily_client.search, query=f"sugarcane {english_search_query} Karnataka", search_depth="advanced"
            )
            context_text = "\n\n".join([f"<doc>{r['content']}</doc>" for r in web_results.get("results", [])[:5]])
        except Exception as e:
            print(f"❌ Web Search Error: {e}")
            context_text = "\n\n".join([f"<doc>{doc}</doc>" for doc in docs]) if docs else "No information available."
    else:
        print(f"🟢 DB Hit! Best Score: {top_chunks[0]['score'] if top_chunks else 'N/A'}")
        context_text = "\n\n".join([f"<doc>{doc}</doc>" for doc in docs])

    # ==========================================
    # 4. MEMORY & GENERATION
    # ==========================================
    gen_result = await generate_from_context(user_query, context_text, session_id)
    ans = gen_result["answer"]

    search_score = top_chunks[0]["score"] if top_chunks else 0.0
    accuracy_score = gen_result["accuracy_score"]

    response_data = {
        "answer": ans,
        "search_score": search_score,
        "accuracy_score": accuracy_score
    }

    if return_context:
        response_data["context"] = context_text
        return response_data
    return response_data