import os
import asyncio
import json
from dotenv import load_dotenv

env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path=env_path, override=True)

from rag_service import get_sugarcane_answer, groq_client
from chat_db import connect_db, close_db

# Test Set
TEST_SET = [
    {
        "question": "ಕಬ್ಬಿನಲ್ಲಿ ಕೆಂಪು ಕೊಳೆ ರೋಗದ ಲಕ್ಷಣಗಳೇನು?",
        "expected": "ಎಲೆಗಳ ಮೇಲೆ ಕೆಂಪು ಕಲೆಗಳು, ಕಾಂಡದ ಒಳಗೆ ಕೆಂಪು ಬಣ್ಣ ಮತ್ತು ಹುಳಿ ವಾಸನೆ."
    },
    {
        "question": "ಕಬ್ಬು ಬೆಳೆಯಲು ಯಾವ ರೀತಿಯ ಮಣ್ಣು ಸೂಕ್ತ?",
        "expected": "ನೀರು ಚೆನ್ನಾಗಿ ಬಸಿದು ಹೋಗುವ ಗೋಡು ಮಣ್ಣು, ಕಪ್ಪು ಮಣ್ಣು ಅಥವಾ ಮರಳು ಮಿಶ್ರಿತ ಮಣ್ಣು."
    },
    {
        "question": "ಕಾಂಡ ಕೊರಕ ಕೀಟವನ್ನು ನಿಯಂತ್ರಿಸುವುದು ಹೇಗೆ?",
        "expected": "ರಾಸಾಯನಿಕ ಕೀಟನಾಶಕಗಳ ಬಳಕೆ ಅಥವಾ ಟ್ರೈಕೋಗ್ರಾಮಾ ಪರಾವಲಂಬಿ ಕೀಟಗಳ ಬಿಡುಗಡೆ."
    },
    {
        "question": "ಒಂದು ಎಕರೆಗೆ ಎಷ್ಟು ಪ್ರಮಾಣದ ರಸಗೊಬ್ಬರ ನೀಡಬೇಕು?",
        "expected": "ಮಣ್ಣು ಪರೀಕ್ಷೆ ಆಧಾರದ ಮೇಲೆ ಸಾರಜನಕ, ರಂಜಕ ಮತ್ತು ಯೂರಿಯಾವನ್ನು ಸರಿಯಾದ ಪ್ರಮಾಣದಲ್ಲಿ ನೀಡಬೇಕು."
    },
    {
        "question": "ಕಬ್ಬು ನಾಟಿ ಮಾಡಲು ಉತ್ತಮ ತಿಂಗಳು ಯಾವುದು?",
        "expected": "ಅಕ್ಟೋಬರ್ ನಿಂದ ನವೆಂಬರ್, ಅಥವಾ ಜನವರಿ ನಿಂದ ಫೆಬ್ರವರಿ ತಿಂಗಳು ಉತ್ತಮ."
    }
]

JUDGE_PROMPT = """
You are an expert evaluator grading an AI assistant's answers.
You will be provided with:
1. QUESTION: The user's question (in Kannada).
2. CONTEXT: The retrieved knowledge.
3. GENERATED_ANSWER: The AI's generated answer (in Kannada).
4. EXPECTED_ANSWER: The golden standard expected answer.

Please evaluate the GENERATED_ANSWER based on two metrics:
1. Faithfulness (0.0 to 1.0): Did the AI stay 100% loyal to the CONTEXT? Did it hallucinate? (1.0 = completely faithful).
2. Relevance (0.0 to 1.0): Did it actually answer the QUESTION effectively considering the EXPECTED_ANSWER? (1.0 = fully relevant and correct).

Respond ONLY with a valid JSON in the exact following format:
{
  "faithfulness": 0.9,
  "relevance": 0.8,
  "reasoning": "Short explanation here"
}
"""

import uuid

async def evaluate_question(item, index):
    print(f"\n--- Testing Q{index + 1}: {item['question']} ---")
    session_id = str(uuid.uuid4())
    
    # Extract ans and context_text via the new parameter
    result = await get_sugarcane_answer(item["question"], session_id, return_context=True)
    ans = result["answer"]
    context_text = result["context"]
    
    prompt = (
        f"QUESTION: {item['question']}\n\n"
        f"CONTEXT: {context_text}\n\n"
        f"EXPECTED_ANSWER: {item['expected']}\n\n"
        f"GENERATED_ANSWER: {ans}"
    )

    try:
        response = await groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": JUDGE_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        
        result_json = response.choices[0].message.content
        metrics = json.loads(result_json)
        
        faithfulness = metrics.get('faithfulness', 0.0)
        relevance = metrics.get('relevance', 0.0)
        
        print(f"✅ Generated Answer: {ans[:100]}...")
        print(f"📊 Faithfulness: {faithfulness} | Relevance: {relevance}")
        print(f"🧠 Reasoning: {metrics.get('reasoning')}")
        
        return faithfulness, relevance
    except Exception as e:
        print(f"❌ Error during AI evaluation: {e}")
        return 0.0, 0.0

async def run_evaluation():
    # Connect to MongoDB before running evaluation
    connect_db()
    
    total_faithfulness = 0.0
    total_relevance = 0.0
    
    for i, item in enumerate(TEST_SET):
        f, r = await evaluate_question(item, i)
        total_faithfulness += f
        total_relevance += r
        
    avg_f = total_faithfulness / len(TEST_SET)
    avg_r = total_relevance / len(TEST_SET)
    mean_accuracy = (avg_f + avg_r) / 2
    
    print("\n" + "="*40)
    print("📈 EVALUATION RESULTS 📈")
    print("="*40)
    print(f"Average Faithfulness : {avg_f:.2f}")
    print(f"Average Relevance    : {avg_r:.2f}")
    print(f"Mean Accuracy Score  : {mean_accuracy:.2f}")
    print("="*40)
    
    # Close connection after execution
    close_db()

if __name__ == "__main__":
    asyncio.run(run_evaluation())
