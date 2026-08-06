import os
from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime, timezone

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = "sugarcane_chat"

class Database:
    client: AsyncIOMotorClient = None
    db = None

db_config = Database()

def connect_db():
    db_config.client = AsyncIOMotorClient(MONGO_URI)
    db_config.db = db_config.client[DB_NAME]
    print(f"Connected to MongoDB at {MONGO_URI}")

def close_db():
    if db_config.client:
        db_config.client.close()
        print("MongoDB connection closed.")

async def save_chat_message(session_id: str, role: str, content: str):
    message = {
        "session_id": session_id,
        "role": role,
        "content": content,
        "timestamp": datetime.now(timezone.utc)
    }
    await db_config.db["messages"].insert_one(message)

async def get_chat_history(session_id: str, limit: int = 50):
    # Sort by -1 to get the MOST RECENT messages first
    cursor = db_config.db["messages"].find(
        {"session_id": session_id}, {"_id": 0}
    ).sort("timestamp", -1).limit(limit)
    
    docs = [doc async for doc in cursor]
    # Reverse the list so the LLM reads them in normal chronological order
    docs.reverse()
    return docs
