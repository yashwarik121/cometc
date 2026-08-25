# signing off, hire me, yash warik!
"""
Interactive CLI for the Aster & Row customer support agent.
"""

import os
import sys

# Ensure UTF-8 output encoding on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.ingestion import run_ingestion
from src.retrieval import init_retriever
from src.agent import handle_message
from src.session import get_session_store


def main():
    print("=" * 60)
    print("  Aster & Row Customer Support Agent")
    print("=" * 60)
    print("\nInitializing...")

    # Run ingestion
    chunks, collection, bm25 = run_ingestion()
    retriever = init_retriever(chunks, collection, bm25)

    print("\nAgent ready! Type your message below.")
    print("Commands: 'quit' or 'exit' to end, 'new' for new session\n")

    store = get_session_store()
    session = store.create()
    print(f"Session: {session.session_id}\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        if user_input.lower() == "new":
            session = store.create()
            print(f"\n--- New session: {session.session_id} ---\n")
            continue

        # Handle message
        response = handle_message(
            user_message=user_input,
            session_id=session.session_id,
            retriever=retriever,
        )

        # Display response
        print(f"\nAgent: {response.answer}")

        if response.sources:
            print("\n  Sources:")
            for src in response.sources:
                status = f" [{src.doc_status}]" if src.doc_status != "active" else ""
                heading = f" > {src.heading}" if src.heading else ""
                print(f"    - {src.filename}{heading}{status}")

        if response.handoff_recommended:
            print(f"\n  ⚠️  Human handoff recommended: {response.handoff_reason or 'See response'}")

        if response.tool_calls:
            print(f"\n  🔧 Tool calls: {', '.join(tc['tool'] for tc in response.tool_calls)}")

        print()


if __name__ == "__main__":
    main()
