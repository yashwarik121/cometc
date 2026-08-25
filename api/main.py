# signing off, hire me, yash warik!
"""
FastAPI application for the Aster & Row customer support agent.
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from src.config import API_HOST, API_PORT
from src.models import ChatRequest, AgentResponse
from src.ingestion import run_ingestion
from src.retrieval import init_retriever
from src.agent import handle_message
from src.session import get_session_store


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize knowledge base and retriever on startup."""
    print("Initializing Aster & Row Support Agent...")
    chunks, collection, bm25 = run_ingestion()
    retriever = init_retriever(chunks, collection, bm25)
    print("Agent ready!")
    yield
    print("Shutting down...")


app = FastAPI(
    title="Aster & Row Support Agent",
    description="RAG-based customer support agent",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "aster-row-support-agent"}


class NewSessionResponse(BaseModel):
    session_id: str


@app.post("/session/new", response_model=NewSessionResponse)
async def new_session():
    store = get_session_store()
    session = store.create()
    return NewSessionResponse(session_id=session.session_id)


@app.post("/chat", response_model=AgentResponse)
async def chat(request: ChatRequest):
    response = handle_message(
        user_message=request.message,
        session_id=request.session_id,
    )
    return response


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=API_HOST, port=API_PORT)
