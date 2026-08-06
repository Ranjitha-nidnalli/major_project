import asyncio
import os
from rag_service import get_sugarcane_answer
from chat_db import connect_db, close_db

async def test():
    connect_db()
    try:
        print("Testing get_sugarcane_answer...")
        # Testing with a simple query and a dummy session_id
        result = await get_sugarcane_answer("How to manage Red Rot?", "test_session_123")
        print(f"\nSUCCESS! Answer: {result['answer']}")
    except Exception as e:
        print(f"\nFAILED! Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        close_db()

if __name__ == "__main__":
    asyncio.run(test())
