"""
Session management for multi-turn conversations.
Tracks conversation history, current order ID, and topic context.
"""

import re
import uuid
from typing import Optional


class Session:
    """A single conversation session."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.conversation_history: list[dict] = []  # [{role, content}]
        self.current_order_id: Optional[str] = None
        self.current_topic: Optional[str] = None  # 'order', 'shipping', 'returns', 'product', 'general'
        self.max_history = 10

    def add_turn(self, role: str, content: str):
        """Add a conversation turn."""
        self.conversation_history.append({"role": role, "content": content})
        # Trim to max history
        if len(self.conversation_history) > self.max_history * 2:
            self.conversation_history = self.conversation_history[-(self.max_history * 2):]

    def get_history(self) -> list[dict]:
        """Get conversation history."""
        return list(self.conversation_history)

    def get_recent_history(self, n: int = 6) -> list[dict]:
        """Get the last n turns."""
        return self.conversation_history[-(n):]


class SessionStore:
    """In-memory session store with isolation between sessions."""

    def __init__(self):
        self._sessions: dict[str, Session] = {}

    def get_or_create(self, session_id: Optional[str] = None) -> Session:
        """Get existing session or create a new one."""
        if session_id and session_id in self._sessions:
            return self._sessions[session_id]
        sid = session_id or str(uuid.uuid4())
        session = Session(sid)
        self._sessions[sid] = session
        return session

    def get(self, session_id: str) -> Optional[Session]:
        """Get a session by ID, or None if not found."""
        return self._sessions.get(session_id)

    def create(self) -> Session:
        """Create a new session with a fresh ID."""
        sid = str(uuid.uuid4())
        session = Session(sid)
        self._sessions[sid] = session
        return session


# Order ID extraction
ORDER_ID_PATTERN = re.compile(
    r'(?:order\s*(?:id|#|number|num|no\.?)?\s*(?:is|:)?\s*)?'
    r'#?\s*(AR[\s\-]?\d{5}|\d{5})\b',
    re.IGNORECASE
)


def extract_order_id(text: str) -> Optional[str]:
    """
    Extract an order ID from user message text.
    Handles: AR-12345, ar-12345, AR 12345, #AR-12345, 12345
    """
    match = ORDER_ID_PATTERN.search(text)
    if match:
        raw = match.group(1).strip().upper()
        # Normalize to AR-XXXXX
        digits_match = re.search(r'\d{5}', raw)
        if digits_match:
            return f"AR-{digits_match.group(0)}"
    return None


def detect_topic(message: str) -> Optional[str]:
    """Simple keyword-based topic detection."""
    msg_lower = message.lower()

    if any(w in msg_lower for w in ["order", "track", "status", "delivery", "deliver", "ship"]):
        if extract_order_id(message):
            return "order"

    if any(w in msg_lower for w in ["return", "refund", "exchange", "restocking"]):
        return "returns"
    if any(w in msg_lower for w in ["shipping", "ship", "deliver", "canada", "international", "australia", "uk"]):
        return "shipping"
    if any(w in msg_lower for w in ["product", "sofa", "table", "lamp", "blanket", "pillow", "bed", "nightstand", "furniture", "decor", "color", "price"]):
        return "product"
    if any(w in msg_lower for w in ["warranty", "guarantee", "defect"]):
        return "warranty"
    if any(w in msg_lower for w in ["cancel", "cancelled"]):
        return "order"
    if any(w in msg_lower for w in ["order", "track", "status"]):
        return "order"

    return "general"


# Module-level singleton
_session_store: SessionStore | None = None


def get_session_store() -> SessionStore:
    global _session_store
    if _session_store is None:
        _session_store = SessionStore()
    return _session_store
