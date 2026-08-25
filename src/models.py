# signing off, hire me, yash warik!
"""
Pydantic models for the Aster & Row RAG agent.
Defines data structures for API requests/responses, tool results,
retrieval results, and internal state.
"""

from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


# --- Order Models ---

class OrderStatus(str, Enum):
    PROCESSING = "processing"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"
    RETURNED = "returned"
    RETURN_REQUESTED = "return_requested"


class OrderItem(BaseModel):
    product_name: str
    quantity: int
    unit_price: float


class SafeOrderResult(BaseModel):
    """
    Order data that is safe to expose to the LLM and end user.
    Internal fields (email, address, notes, risk_score) are NEVER included.
    This model is the serialization boundary — the LLM never sees more than this.
    """
    order_id: str
    customer_name: str
    items: list[OrderItem]
    order_total: float
    order_date: str
    status: OrderStatus
    tracking_number: Optional[str] = None
    carrier: Optional[str] = None
    estimated_delivery: Optional[str] = None
    actual_delivery: Optional[str] = None
    payment_method: Optional[str] = None
    cancellation_reason: Optional[str] = None
    return_reason: Optional[str] = None


class OrderLookupError(BaseModel):
    """Returned when order lookup fails."""
    error: str
    order_id: str


# --- Retrieval Models ---

class ChunkMetadata(BaseModel):
    """Metadata attached to each knowledge base chunk."""
    source_file: str
    heading: str
    title: str
    status: str  # 'active' or 'superseded'
    doc_type: str  # 'policy', 'product', 'faq', 'internal'
    effective_date: Optional[str] = None
    superseded_by: Optional[str] = None


class RetrievalResult(BaseModel):
    """A single retrieved chunk with its metadata and scores."""
    text: str
    score: float
    metadata: ChunkMetadata
    chunk_id: str


class RetrievalResponse(BaseModel):
    """Full retrieval result set with conflict/confidence flags."""
    chunks: list[RetrievalResult]
    has_conflict: bool = False
    conflict_description: Optional[str] = None
    low_confidence: bool = False


# --- Agent/API Models ---

class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str


class SourceCitation(BaseModel):
    filename: str
    heading: str
    doc_status: str = "active"


class AgentResponse(BaseModel):
    """The structured response from the agent."""
    answer: str
    sources: list[SourceCitation] = Field(default_factory=list)
    handoff_recommended: bool = False
    handoff_reason: Optional[str] = None
    session_id: str = ""
    tool_calls: list[dict] = Field(default_factory=list)


# --- Trace/Observability Models ---

class TraceLog(BaseModel):
    """Structured trace log for a single agent turn."""
    timestamp: str
    session_id: str
    user_message: str
    relevant_history: list[dict] = Field(default_factory=list)
    retrieved_chunks: list[dict] = Field(default_factory=list)
    tool_calls: list[dict] = Field(default_factory=list)
    final_response: str = ""
    sources_cited: list[str] = Field(default_factory=list)
    handoff_triggered: bool = False
    fallback_triggered: bool = False
