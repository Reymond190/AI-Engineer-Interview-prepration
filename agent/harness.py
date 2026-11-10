"""
agent/harness.py
----------------
Deterministic tool-use agent loop for DeepSeek-R1-Distill-Qwen-14B.

Architecture
------------
The model is served locally via vLLM's OpenAI-compatible API.
Because the DeepSeek-R1 distilled models are reasoning models (they
emit <think>...</think> traces) and do NOT natively implement
OpenAI function-calling, we use a prompt-level tool schema injected
into the system message and parse the model's text output.

Tool-call round-trip
--------------------
  1. System prompt defines strict output format with XML-like tags.
  2. Model emits:
       <think>...</think>           ← stripped by parser
       <tool_call>{"action":..., "args":...}</tool_call>
     OR
       <final_answer>...</final_answer>
  3. Harness validates the JSON, dispatches the tool, and appends
     the observation as a new user message.
  4. Loop repeats until <final_answer> is received or max_turns is hit.

All events are logged to logs/<run_id>.ndjson via AgentLogger.
"""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path
from typing import Any

from openai import OpenAI

from agent.schema import ParseResult, parse_model_output, ToolName, SHELL_ALLOWLIST
from agent.tools import dispatch_tool
from agent.logger import AgentLogger


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = f"""You are a helpful assistant with access to a small set of local tools.

## Available tools

1. calculator  – evaluate a safe arithmetic expression
   Arguments: {{ "expression": "<math expression using + - * / // % ** ()>" }}
   Example:   {{"action": "calculator", "args": {{"expression": "17 * 4 + 3"}}}}

2. read_file   – read a file from the fixtures directory (read-only)
   Arguments: {{ "filename": "<filename with extension, no path>" }}
   Example:   {{"action": "read_file", "args": {{"filename": "inventory.txt"}}}}

3. shell       – run a read-only shell command from the allowlist
   Allowlist:  {sorted(SHELL_ALLOWLIST)}
   Arguments: {{ "command": "<command and args>" }}
   Example:   {{"action": "shell", "args": {{"command": "df -h"}}}}

## Output format – STRICT

When you want to use a tool, output EXACTLY:
<tool_call>
{{"action": "<tool_name>", "args": {{<args object>}}}}
</tool_call>

When you have a final answer (or no tool is needed), output EXACTLY:
<final_answer>
<your complete answer here>
</final_answer>

## Rules
- Output ONLY ONE block per turn (either <tool_call> or <final_answer>).
- Never output both in the same turn.
- Never fabricate tool results. Always wait for the tool observation.
- If a tool returns an error, report it in your <final_answer>.
"""


def _build_observation_message(action: str, result: str, error: str | None) -> str:
    """Format a tool result as the next user message."""
    if error:
        return (
            f"[tool: {action}]\n"
            f"ERROR: {error}\n"
            "Please acknowledge the error and continue."
        )
    return (
        f"[tool: {action}]\n"
        f"RESULT:\n{result}"
    )


# ---------------------------------------------------------------------------
# Main harness
# ---------------------------------------------------------------------------

class AgentHarness:
    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        max_turns: int = 10,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        log_dir: Path | None = None,
    ):
        self.client      = OpenAI(base_url=base_url, api_key="not-needed")
        self.model       = model
        self.max_turns   = max_turns
        self.max_tokens  = max_tokens
        self.temperature = temperature
        self.log_dir     = log_dir

    def run(self, task: str, run_id: str | None = None) -> dict[str, Any]:
        """
        Run the agent on a task.

        Parameters
        ----------
        task   : The user's natural-language task.
        run_id : Optional ID for the run. Defaults to a new UUID.

        Returns
        -------
        dict with keys: run_id, final_answer, turns, total_latency_s,
                        tool_calls, schema_errors, log_path
        """
        if run_id is None:
            run_id = str(uuid.uuid4())

        logger = AgentLogger(run_id=run_id, log_dir=self.log_dir)
        logger.log_init(task=task, model=self.model, base_url=str(self.client.base_url))

        messages: list[dict[str, str]] = [
            {"role": "system",  "content": SYSTEM_PROMPT},
            {"role": "user",    "content": task},
        ]

        stats = {
            "run_id":         run_id,
            "final_answer":   None,
            "turns":          0,
            "total_latency_s": 0.0,
            "tool_calls":     0,
            "schema_errors":  0,
            "log_path":       str(logger.path()),
        }

        for turn in range(self.max_turns):
            stats["turns"] = turn + 1

            # ---- Call model ------------------------------------------------
            t0 = time.perf_counter()
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,  # type: ignore[arg-type]
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
            except Exception as exc:  # noqa: BLE001
                logger.log_error(turn, f"Model API error: {exc}")
                stats["error"] = str(exc)
                break

            elapsed = time.perf_counter() - t0
            stats["total_latency_s"] += elapsed

            raw     = resp.choices[0].message.content or ""
            usage   = resp.usage
            in_tok  = usage.prompt_tokens     if usage else 0
            out_tok = usage.completion_tokens if usage else 0

            logger.log_model_response(turn, raw, elapsed, in_tok, out_tok)

            # Append assistant turn to history
            messages.append({"role": "assistant", "content": raw})

            # ---- Parse output ----------------------------------------------
            result: ParseResult = parse_model_output(raw)

            # ---- Malformed -------------------------------------------------
            if result.kind == "malformed":
                stats["schema_errors"] += 1
                logger.log_malformed(turn, raw, result.error or "unknown")

                # Feed error back so the model can self-correct (one chance)
                messages.append({
                    "role":    "user",
                    "content": (
                        f"[PARSE ERROR] Your output could not be parsed: {result.error}\n"
                        "Please try again with a valid <tool_call> or <final_answer> block."
                    ),
                })
                continue

            # ---- Final answer ----------------------------------------------
            if result.kind == "final_answer":
                answer = result.final_answer or ""
                logger.log_final_answer(turn, answer, stats["turns"], stats["total_latency_s"])
                stats["final_answer"] = answer
                break

            # ---- Tool call -------------------------------------------------
            if result.kind == "tool_call" and result.tool_call is not None:
                tc     = result.tool_call
                action = tc.action.value
                args   = tc.args

                logger.log_tool_call(turn, action, args)
                stats["tool_calls"] += 1

                tool_result, tool_error = dispatch_tool(tc)

                logger.log_tool_result(turn, action, tool_result, tool_error)

                observation = _build_observation_message(action, tool_result, tool_error)
                messages.append({"role": "user", "content": observation})
                continue

            # ---- Empty / unexpected ----------------------------------------
            logger.log_error(turn, f"Unhandled parse result kind: {result.kind!r}")
            break

        else:
            # max_turns exhausted without a final answer
            logger.log_error(None, f"Max turns ({self.max_turns}) reached without a final answer.")

        logger.log_summary({
            "turns":          stats["turns"],
            "tool_calls":     stats["tool_calls"],
            "schema_errors":  stats["schema_errors"],
            "total_latency_s": round(stats["total_latency_s"], 3),
            "final_answer_chars": len(stats.get("final_answer") or ""),
        })
        logger.close()

        return stats


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, json  # noqa: E401

    parser = argparse.ArgumentParser(description="Run the DeepSeek tool-use agent on a task.")
    parser.add_argument("task",        help="Task for the agent (quoted string)")
    parser.add_argument("--base-url",  default=os.getenv("AGENT_BASE_URL", "http://localhost:8000/v1"))
    parser.add_argument("--max-turns", type=int,   default=10)
    parser.add_argument("--max-tokens",type=int,   default=1024)
    parser.add_argument("--run-id",    default=None)
    args = parser.parse_args()

    harness = AgentHarness(
        base_url=args.base_url,
        max_turns=args.max_turns,
        max_tokens=args.max_tokens,
    )

    result = harness.run(task=args.task, run_id=args.run_id)
    print("\n--- Agent result ---")
    print(json.dumps(result, indent=2, ensure_ascii=False))
