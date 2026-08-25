# signing off, hire me, yash warik!
"""
Structured observability logging per agent turn.
Logs user message, retrieved chunks, tool calls, response, and handoff info.
Never logs private fields.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from src.config import LOG_DIR
from src.models import TraceLog


def _ensure_log_dir():
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)


def log_trace(trace: TraceLog):
    """Write a structured trace log entry to a session-specific log file."""
    _ensure_log_dir()

    log_file = Path(LOG_DIR) / f"session_{trace.session_id}.jsonl"
    entry = trace.model_dump()

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def create_trace(session_id: str, user_message: str) -> TraceLog:
    """Create a new trace log entry."""
    return TraceLog(
        timestamp=datetime.now(timezone.utc).isoformat(),
        session_id=session_id,
        user_message=user_message,
    )
