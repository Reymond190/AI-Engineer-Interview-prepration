#!/usr/bin/env python3
"""
tests/run_tests.py
------------------
Run the full agent scenario suite against the live local server
and print a summary of tool-call success rate, schema failures,
latency, and VRAM use.

Usage (inside WSL, conda env activated, from project root):
    python tests/run_tests.py [--base-url http://localhost:8000/v1] [--scenario <id>]

Each scenario creates a separate agent run with its own log file
under logs/<run_id>.ndjson.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import unittest.mock as mock
from pathlib import Path

# Ensure project root is on PYTHONPATH when run directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.harness  import AgentHarness
from agent.schema   import ParseResult
from tests.scenarios import SCENARIOS

GREEN  = "\033[0;32m"
RED    = "\033[0;31m"
YELLOW = "\033[1;33m"
CYAN   = "\033[0;36m"
NC     = "\033[0m"

PAD = 30


# ---------------------------------------------------------------------------
# VRAM snapshot helper
# ---------------------------------------------------------------------------

def vram_mib() -> tuple[int, int]:
    """Return (used_mib, total_mib) from nvidia-smi. Returns (0,0) on failure."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        used, total = [int(x.strip()) for x in out.split(",")]
        return used, total
    except Exception:  # noqa: BLE001
        return 0, 0


# ---------------------------------------------------------------------------
# Scenario runner
# ---------------------------------------------------------------------------

def run_scenario(scenario: dict, harness: AgentHarness) -> dict:
    """
    Run one scenario and validate the result against the scenario spec.
    Returns a result dict with pass/fail breakdown.
    """
    sid   = scenario["id"]
    task  = scenario["task"]

    expect_tools      = set(scenario.get("expect_tool", []))
    expect_in         = scenario.get("expect_in_answer", [])
    expect_not_in     = scenario.get("expect_not_in_answer", [])
    expect_schema_err = scenario.get("expect_schema_error", False)
    tool_error_str    = scenario.get("tool_error_contains", None)
    inject_malformed  = scenario.get("inject_malformed_first", False)
    max_turns         = scenario.get("max_turns", harness.max_turns)

    old_max = harness.max_turns
    harness.max_turns = max_turns

    # ---- Possibly inject a malformed first response ----------------------
    if inject_malformed:
        original_create = harness.client.chat.completions.create

        call_count = [0]
        def _mock_create(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                # Return a fake malformed response object
                m = mock.MagicMock()
                m.choices[0].message.content = "I will answer: <tool_call>NOTJSON</tool_call>"
                m.usage.prompt_tokens     = 10
                m.usage.completion_tokens = 10
                return m
            return original_create(**kwargs)

        harness.client.chat.completions.create = _mock_create  # type: ignore[method-assign]

    vram_before, total_vram = vram_mib()
    t0 = time.perf_counter()
    stats = harness.run(task=task)
    elapsed = time.perf_counter() - t0
    vram_after, _ = vram_mib()

    if inject_malformed:
        harness.client.chat.completions.create = original_create  # type: ignore[method-assign]
        harness.max_turns = old_max

    harness.max_turns = old_max

    answer  = (stats.get("final_answer") or "").lower()
    log_path = Path(stats.get("log_path", ""))

    # ---- Read full NDJSON log to check tool usage -------------------------
    called_tools: set[str] = set()
    tool_errors:  list[str] = []
    schema_errors: int = 0

    if log_path.exists():
        for line in log_path.read_text().splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("event") == "tool_call":
                called_tools.add(rec.get("action", ""))
            if rec.get("event") == "tool_result" and rec.get("error"):
                tool_errors.append(rec["error"])
            if rec.get("event") == "malformed":
                schema_errors += 1

    # ---- Assertions -------------------------------------------------------
    failures: list[str] = []

    for tool in expect_tools:
        if tool not in called_tools:
            failures.append(f"Expected tool '{tool}' to be called but was not.")

    for substr in expect_in:
        if substr.lower() not in answer:
            failures.append(f"Expected {substr!r} in final answer.")

    for substr in expect_not_in:
        if substr.lower() in answer:
            failures.append(f"Unexpected {substr!r} found in final answer.")

    if expect_schema_err and schema_errors == 0:
        failures.append("Expected at least one schema error but none occurred.")

    if tool_error_str:
        if not any(tool_error_str.lower() in e.lower() for e in tool_errors):
            failures.append(f"Expected tool error containing {tool_error_str!r}.")

    return {
        "id":            sid,
        "passed":        len(failures) == 0,
        "failures":      failures,
        "turns":         stats.get("turns", 0),
        "tool_calls":    stats.get("tool_calls", 0),
        "schema_errors": schema_errors,
        "called_tools":  sorted(called_tools),
        "latency_s":     round(elapsed, 1),
        "vram_delta_mib": vram_after - vram_before,
        "total_vram_mib": total_vram,
        "log_path":      str(log_path),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Run agent capability test suite")
    parser.add_argument("--base-url",  default=os.getenv("AGENT_BASE_URL", "http://localhost:8000/v1"))
    parser.add_argument("--scenario",  default=None, help="Run only this scenario id")
    parser.add_argument("--max-turns", type=int, default=10)
    args = parser.parse_args()

    harness = AgentHarness(
        base_url=args.base_url,
        max_turns=args.max_turns,
        max_tokens=1024,
        temperature=0.0,
    )

    scenarios = SCENARIOS
    if args.scenario:
        scenarios = [s for s in SCENARIOS if s["id"] == args.scenario]
        if not scenarios:
            print(f"No scenario with id {args.scenario!r}. Available: {[s['id'] for s in SCENARIOS]}")
            return 1

    results = []
    print(f"\n{CYAN}Running {len(scenarios)} scenario(s) against {args.base_url}{NC}\n")
    print(f"{'Scenario':<{PAD}} {'Status':8} {'Turns':6} {'Tools':5} {'Latency':8} {'VRAM Δ':8}")
    print("-" * 72)

    for scenario in scenarios:
        if scenario.get("skip_live") and not scenario.get("inject_malformed_first"):
            print(f"{scenario['id']:<{PAD}} {'SKIP':8}")
            continue

        res = run_scenario(scenario, harness)
        results.append(res)

        status = f"{GREEN}PASS{NC}" if res["passed"] else f"{RED}FAIL{NC}"
        vram_str = f"+{res['vram_delta_mib']}MiB" if res["vram_delta_mib"] > 0 else f"{res['vram_delta_mib']}MiB"
        print(
            f"{res['id']:<{PAD}} {status:8} {res['turns']:6} {res['tool_calls']:5}"
            f" {res['latency_s']:7.1f}s {vram_str:8}"
        )

        if not res["passed"]:
            for f in res["failures"]:
                print(f"  {RED}✗{NC} {f}")

    # ---- Summary ---------------------------------------------------------
    passed = sum(1 for r in results if r["passed"])
    total  = len(results)

    print("\n" + "="*72)
    print(f"  Results  :  {passed}/{total} passed")
    print(f"  Avg latency : {sum(r['latency_s'] for r in results)/max(total,1):.1f}s per scenario")
    tool_calls_total = sum(r["tool_calls"] for r in results)
    schema_err_total = sum(r["schema_errors"] for r in results)
    print(f"  Tool calls  : {tool_calls_total}   Schema errors: {schema_err_total}")
    vram_after, total_vram = vram_mib()
    if total_vram > 0:
        print(f"  VRAM now    : {vram_after} / {total_vram} MiB  ({100*vram_after/total_vram:.1f}% used)")
    print("="*72)

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
