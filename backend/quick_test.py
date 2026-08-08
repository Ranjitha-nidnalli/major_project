import asyncio
import sys
sys.path.insert(0, ".")

from rag_service import get_sugarcane_answer
from chat_db import connect_db, close_db

async def test():
    connect_db()

    questions = [
        "ಕಬ್ಬಿಗೆ ಯಾವ ಮಣ್ಣು ಉತ್ತಮ?",
        "ಕಬ್ಬಿನಲ್ಲಿ ಬೊರರ್ ರೋಗದ ಲಕ್ಷಣಗಳು ಏನು?",
        "ಕಬ್ಬಿನ ಬೆಳೆಗೆ ಯಾವ ರಸಗೊಬ್ಬರ ಬಳಸಬೇಕು?",
    ]

    for q in questions:
        print(f"\n{'='*60}")
        print(f"Q: {q}")
        print("="*60)
        r = await get_sugarcane_answer(q, f"test_{hash(q) & 0xFFFFFFFF}")
        print(f"Answer: {r['answer'][:300]}...")
        print(f"Search score: {r['search_score']:.2f}")
        print(f"Accuracy score: {r.get('accuracy_score', 'N/A')}")

    close_db()

if __name__ == "__main__":
    asyncio.run(test())
