# signing off, hire me, yash warik!
"""
Evaluation suite for the Aster & Row customer support agent.
Runs all test cases from visible-cases.json and custom-cases.json.
Produces per-case pass/fail with category breakdown.

Usage:
    python evaluate.py [--verbose] [--case TC-001]
"""

import json
import os
import sys
import time
import argparse
from pathlib import Path
from collections import defaultdict

# Configure UTF-8 with error replacement for Windows terminals (CMD / PowerShell)
if sys.platform == "win32":
    try:
        os.system("chcp 65001 > nul 2>&1")
    except Exception:
        pass

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.ingestion import run_ingestion
from src.retrieval import init_retriever
from src.agent import handle_message
from src.session import get_session_store


def load_test_cases() -> list[dict]:
    """Load all test cases from both visible and custom case files."""
    cases = []
    eval_dir = Path(__file__).parent / "evaluation"

    for filename in ["visible-cases.json", "custom-cases.json"]:
        filepath = eval_dir / filename
        if filepath.exists():
            with open(filepath, "r") as f:
                file_cases = json.load(f)
                cases.extend(file_cases)
                print(f"  Loaded {len(file_cases)} cases from {filename}")

    return cases


def check_assertion(assertion: dict, response, all_tool_calls: list[dict]) -> tuple[bool, str]:
    """
    Check a single assertion against the agent response.
    Returns (passed, explanation).
    """
    atype = assertion["type"]
    value = assertion["value"]
    desc = assertion.get("description", "")

    answer_lower = response.answer.lower()
    answer_text = response.answer

    if atype == "contains":
        # Case-insensitive substring check
        passed = value.lower() in answer_lower
        if not passed:
            return False, f"FAIL: Expected '{value}' in response. {desc}"
        return True, f"PASS: Found '{value}' in response"

    elif atype == "not_contains":
        # Case-insensitive forbidden substring check
        passed = value.lower() not in answer_lower
        if not passed:
            return False, f"FAIL: Forbidden text '{value}' found in response. {desc}"
        return True, f"PASS: '{value}' correctly absent from response"

    elif atype == "tool_called":
        # Check if the specified tool was called
        tool_names = [tc.get("tool", "") for tc in all_tool_calls]
        passed = value in tool_names
        if not passed:
            return False, f"FAIL: Expected tool '{value}' to be called. Called: {tool_names}. {desc}"
        return True, f"PASS: Tool '{value}' was called"

    elif atype == "tool_not_called":
        # Check tool was NOT called
        tool_names = [tc.get("tool", "") for tc in all_tool_calls]
        passed = value not in tool_names
        if not passed:
            return False, f"FAIL: Tool '{value}' should NOT have been called. {desc}"
        return True, f"PASS: Tool '{value}' correctly not called"

    elif atype == "source_cited":
        # Check if a specific source file is cited
        source_files = [s.filename for s in response.sources]
        # Also check in the response text itself
        passed = (
            value in source_files
            or value.lower() in answer_lower
            or any(value in sf for sf in source_files)
        )
        if not passed:
            return False, f"FAIL: Expected source '{value}' to be cited. Sources: {source_files}. {desc}"
        return True, f"PASS: Source '{value}' cited"

    elif atype == "recommends_handoff":
        passed = response.handoff_recommended
        if not passed:
            # Also check text for handoff language
            handoff_keywords = ["contact", "support", "human", "reach out", "1-800", "support@"]
            text_handoff = any(kw in answer_lower for kw in handoff_keywords)
            passed = text_handoff
        if not passed:
            return False, f"FAIL: Expected handoff recommendation. {desc}"
        return True, f"PASS: Handoff recommended"

    else:
        return False, f"UNKNOWN assertion type: {atype}"


def run_single_case(case: dict, retriever, verbose: bool = False) -> dict:
    """Run a single test case. Returns result dict."""
    case_id = case["id"]
    category = case["category"]
    description = case["description"]
    turns = case["turns"]
    assertions = case["assertions"]

    # Create a fresh session for this case
    store = get_session_store()
    session = store.create()

    response = None
    all_tool_calls = []

    # Process each turn
    for turn in turns:
        if turn["role"] == "user":
            response = handle_message(
                user_message=turn["content"],
                session_id=session.session_id,
                retriever=retriever,
            )
            all_tool_calls.extend(response.tool_calls)
        elif turn["role"] == "assistant":
            # Simulate the assistant response in session history
            session.add_turn("assistant", turn["content"])

    if response is None:
        return {
            "id": case_id,
            "category": category,
            "description": description,
            "passed": False,
            "assertions": [{"passed": False, "detail": "No user turns in test case"}],
            "response": "",
        }

    # Check assertions
    assertion_results = []
    all_passed = True

    for assertion in assertions:
        passed, detail = check_assertion(assertion, response, all_tool_calls)
        assertion_results.append({
            "type": assertion["type"],
            "value": assertion["value"],
            "passed": passed,
            "detail": detail,
        })
        if not passed:
            all_passed = False

    return {
        "id": case_id,
        "category": category,
        "description": description,
        "passed": all_passed,
        "assertions": assertion_results,
        "response": response.answer[:200] + "..." if len(response.answer) > 200 else response.answer,
        "sources": [s.filename for s in response.sources],
        "tool_calls": [tc.get("tool") for tc in all_tool_calls],
        "handoff": response.handoff_recommended,
    }


def print_results(results: list[dict], verbose: bool = False):
    """Print formatted test results with category breakdown."""
    print("\n" + "=" * 70)
    print("  EVALUATION RESULTS")
    print("=" * 70)

    # Per-case results
    for r in results:
        status = "[PASS]" if r["passed"] else "[FAIL]"
        print(f"\n  {r['id']} [{r['category']}] {status}")
        print(f"    {r['description']}")

        if verbose or not r["passed"]:
            for a in r["assertions"]:
                marker = "  [+]" if a["passed"] else "  [-]"
                print(f"    {marker} {a['detail']}")
            if not r["passed"]:
                print(f"    Response: {r['response'][:150]}...")

    # Category breakdown
    print("\n" + "-" * 70)
    print("  CATEGORY BREAKDOWN")
    print("-" * 70)

    categories = defaultdict(lambda: {"total": 0, "passed": 0})
    for r in results:
        categories[r["category"]]["total"] += 1
        if r["passed"]:
            categories[r["category"]]["passed"] += 1

    total_pass = sum(c["passed"] for c in categories.values())
    total_cases = sum(c["total"] for c in categories.values())

    for cat, counts in sorted(categories.items()):
        pct = (counts["passed"] / counts["total"] * 100) if counts["total"] > 0 else 0
        bar = "#" * int(pct / 10) + "-" * (10 - int(pct / 10))
        print(f"  {cat:15s} [{bar}] {counts['passed']}/{counts['total']} ({pct:.0f}%)")

    print(f"\n  {'TOTAL':15s}        {total_pass}/{total_cases} ({total_pass/total_cases*100:.0f}%)")
    print("=" * 70)

    return total_pass, total_cases


def main():
    parser = argparse.ArgumentParser(description="Aster & Row Agent Evaluation Suite")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show all assertion details")
    parser.add_argument("--case", "-c", type=str, help="Run a specific test case by ID")
    parser.add_argument("--category", "-cat", type=str, help="Run only cases in a category")
    args = parser.parse_args()

    print("=" * 70)
    print("  Aster & Row Agent - Evaluation Suite")
    print("=" * 70)

    # Initialize
    print("\nInitializing agent...")
    chunks, collection, bm25 = run_ingestion()
    retriever = init_retriever(chunks, collection, bm25)

    # Load test cases
    print("\nLoading test cases...")
    cases = load_test_cases()

    # Filter if requested
    if args.case:
        cases = [c for c in cases if c["id"] == args.case]
        if not cases:
            print(f"  No case found with ID '{args.case}'")
            sys.exit(1)

    if args.category:
        cases = [c for c in cases if c["category"] == args.category]
        if not cases:
            print(f"  No cases found in category '{args.category}'")
            sys.exit(1)

    print(f"\nRunning {len(cases)} test cases...\n")

    # Run tests
    results = []
    for i, case in enumerate(cases):
        print(f"  [{i+1}/{len(cases)}] Running {case['id']}...", end="", flush=True)
        start = time.time()
        result = run_single_case(case, retriever, args.verbose)
        elapsed = time.time() - start
        status = "[PASS]" if result["passed"] else "[FAIL]"
        print(f" {status} ({elapsed:.1f}s)")
        results.append(result)

    # Print results
    total_pass, total_cases = print_results(results, args.verbose)

    # Save results to file
    results_file = Path(__file__).parent / "evaluation" / "results.json"
    with open(results_file, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "total_pass": total_pass,
            "total_cases": total_cases,
            "results": results,
        }, f, indent=2)
    print(f"\n  Results saved to {results_file}")

    # Exit with non-zero if any failures
    sys.exit(0 if total_pass == total_cases else 1)


if __name__ == "__main__":
    main()
