import os
from dotenv import load_dotenv

# Load env variables
env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path=env_path, override=True)

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from chat_db import connect_db, close_db, get_chat_history
from rag_service import get_sugarcane_answer

# --- MODERN LIFESPAN MANAGER ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Connect to MongoDB
    print("🚀 Connecting to MongoDB...")
    connect_db() 
    yield
    # Shutdown: Close connections
    print("🛑 Closing MongoDB connection...")
    close_db()

app = FastAPI(title="Sugarcane Chat API", lifespan=lifespan)

# Setup CORS for your Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], # Your Next.js port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- MODELS ---
class ChatRequest(BaseModel):
    session_id: str
    query: str

class ChatResponse(BaseModel):
    answer: str
    search_score: float = 0.0
    accuracy_score: float = 0.0

# --- ENDPOINTS ---

@app.get("/")
async def root():
    return {"message": "Welcome to Sugarcane Chat API"}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        # Calls the logic that handles Retrieval + LLM + Saving to Mongo
        result = await get_sugarcane_answer(request.query, request.session_id)
        return ChatResponse(
            answer=result.get("answer", ""),
            search_score=result.get("search_score", 0.0),
            accuracy_score=result.get("accuracy_score", 0.0)
        )
    except Exception as e:
        print(f"❌ Server Error: {e}")
        raise HTTPException(status_code=500, detail="ನನ್ನ ಕ್ಷಮಿಸಿ, ಸರ್ವರ್‌ನಲ್ಲಿ ದೋಷ ಕಂಡುಬಂದಿದೆ.")

@app.get("/history/{session_id}")
async def get_history(session_id: str, limit: int = 50):
    try:
        history = await get_chat_history(session_id, limit)
        return {"history": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    # Run the server on port 8000
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)