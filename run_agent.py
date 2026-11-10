#!/usr/bin/env python3
"""
run_agent.py – convenience wrapper for the agent harness.

Usage (from project root, inside WSL with conda env activated):
    python run_agent.py "What is 1234 * 5678?"
    python run_agent.py "Read inventory.txt and tell me the total Gadget X value."
    python run_agent.py "Run the 'df -h' shell command and summarise disk usage."

Options:
    --base-url   Server URL (default: http://localhost:8000/v1 or $AGENT_BASE_URL)
    --max-turns  Maximum tool-call rounds before giving up (default: 10)
    --max-tokens Max tokens per model response (default: 1024)
    --json       Print full JSON result at the end
"""

from agent.harness import AgentHarness

if __name__ == "__main__":
    import argparse
    import json
    import os

    parser = argparse.ArgumentParser(description="Run the DeepSeek tool-use agent")
    parser.add_argument("task",         help="Task for the agent")
    parser.add_argument("--base-url",   default=os.getenv("AGENT_BASE_URL", "http://localhost:8000/v1"))
    parser.add_argument("--max-turns",  type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--json",       action="store_true", help="Print full JSON stats at the end")
    args = parser.parse_args()

    harness = AgentHarness(
        base_url=args.base_url,
        max_turns=args.max_turns,
        max_tokens=args.max_tokens,
    )

    result = harness.run(task=args.task)

    if args.json:
        print(json.dumps(result, indent=2))
