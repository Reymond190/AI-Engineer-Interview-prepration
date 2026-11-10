"""
core.py - event log, sandbox-bound tools, and the tool-using agent loop.

The agent loop is hand-written (instead of ToolNode) because it needs things a
long autonomous run needs: a wall-clock deadline, a step cap, context
compaction for the 9B model's small KV budget, and a terminal "submit" tool.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "langgraph_course"))
from common import get_llm  # noqa: E402

from fx_data import fetch_rates  # noqa: E402
from sandbox import Sandbox, clip  # noqa: E402

# =============================================================================
# Event log: live terminal view + events.jsonl
# =============================================================================

COLORS = ["\033[36m", "\033[33m", "\033[35m", "\033[32m", "\033[34m", "\033[91m"]
RESET, DIM, BOLD = "\033[0m", "\033[2m", "\033[1m"


class EventLog:
    def __init__(self, run_dir: Path):
        self.path = run_dir / "events.jsonl"
        self.t0 = time.time()
        self.lock = threading.Lock()
        self.colors: dict[str, str] = {}

    def __call__(self, agent: str, kind: str, text: str = "", **data: Any) -> None:
        with self.lock:
            color = self.colors.setdefault(agent, COLORS[len(self.colors) % len(COLORS)])
            elapsed = time.time() - self.t0
            line = text.replace("\n", " ")[:220]
            print(f"{DIM}{elapsed:6.0f}s{RESET} {color}{agent:<12}{RESET} {BOLD}{kind:<10}{RESET} {line}", flush=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"t": round(elapsed, 1), "agent": agent, "kind": kind, "text": text, **data}, default=str) + "\n")


# =============================================================================
# Tools bound to one sandbox
# =============================================================================

class Submission(BaseModel):
    answer: str = Field(description="The final answer: an exact number/value, or a one-line conclusion for research tasks")
    summary: str = Field(description="How you got it: method, key evidence, caveats (max ~200 words)")
    files: list[str] = Field(default_factory=list, description="Workspace files that support the answer (code, csv, png)")


def make_tools(sb: Sandbox, log: EventLog, agent: str, deadline: float) -> list[BaseTool]:
    counter = {"n": 0}

    def budget(requested: int) -> int:
        """A command may never eat the agent's last 45s (needed to submit)."""
        return max(5, min(requested, 600, int(deadline - time.time() - 45)))

    @tool
    def run_python(code: str, timeout_s: int = 120) -> str:
        """Run a Python script in your sandbox (cwd /workspace, numpy/pandas/scipy/sympy/numba/sklearn/statsmodels installed).
        Print only compact results; save large outputs to files. The script is saved as /workspace/snippets/NNN.py."""
        counter["n"] += 1
        rel = f"snippets/{counter['n']:03d}.py"
        p = sb.host_path(rel)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(code, encoding="utf-8")
        r = sb.exec(f"python {rel}", timeout_s=budget(timeout_s))                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              87
        log(agent, "python", f"{rel} exit={r['exit_code']} {r['seconds']}s | {r['output'][:120]}")
        return f"exit_code={r['exit_code']} ({r['seconds']}s)\n{r['output'] or '(no output)'}"

    @tool
    def run_bash(command: str, timeout_s: int = 120) -> str:
        """Run a bash command in your sandbox (Debian, internet access, pip install --user works)."""
        r = sb.exec(command, timeout_s=budget(timeout_s))
        log(agent, "bash", f"$ {command[:80]} -> exit={r['exit_code']}")
        return f"exit_code={r['exit_code']} ({r['seconds']}s)\n{r['output'] or '(no output)'}"

    @tool
    def write_file(path: str, content: str) -> str:
        """Create or overwrite a file in /workspace."""
        p = sb.host_path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        log(agent, "write", f"{path} ({len(content)} chars)")
        return f"wrote {path}"

    @tool
    def read_file(path: str, max_chars: int = 4000) -> str:
        """Read a text file from /workspace (or /candidates for the verifier)."""
        if path.startswith("/candidates"):
            r = sb.exec(f"head -c {max_chars} {json.dumps(path)}")
            return r["output"]
        p = sb.host_path(path)
        return clip(p.read_text(encoding="utf-8", errors="replace"), max_chars) if p.exists() else f"No such file: {path}"

    @tool
    def fetch_fx(symbol: str, timeframe: str, start: str, end: str = "") -> str:
        """Download OHLC candles from the MetaTrader 5 terminal into /workspace/data/ as CSV.
        symbol e.g. 'EURUSD'; timeframe one of M1 M5 M15 M30 H1 H4 D1 W1; start/end ISO dates 'YYYY-MM-DD' (end empty = now).
        Columns: time (broker server time), open, high, low, close, volume (tick volume), spread (points)."""
        try:
            meta = fetch_rates(symbol, timeframe, start, end or None, sb.workspace / "data")
        except Exception as e:  # noqa: BLE001  report to the model, it can retry
            return f"fetch_fx failed: {e}"
        log(agent, "fetch_fx", f"{meta['symbol']} {meta['timeframe']} {meta['rows']} rows {meta['first']}..{meta['last']}")
        return json.dumps(meta)

    @tool(args_schema=Submission)
    def submit_result(answer: str, summary: str, files: list[str] | None = None) -> str:
        """Submit your final result. Call this exactly once when done; it ends your turn."""
        return "submitted"

    return [run_python, run_bash, write_file, read_file, fetch_fx, submit_result]


# =============================================================================
# Context compaction
# =============================================================================

def compact(messages: list[BaseMessage], limit: int, notes_llm) -> list[BaseMessage]:
    """Keep the context under `limit` tokens.
    1) shrink old tool outputs (cheap), 2) if still too big, summarise the middle
    into progress notes and keep the recent turns verbatim."""
    keep_recent = 8
    for m in messages[:-keep_recent]:
        if isinstance(m, ToolMessage) and len(str(m.content)) > 400:
            m.content = str(m.content)[:400] + " ...[old output truncated]"
    if count_tokens_approximately(messages) <= limit:
        return messages

    head, body = messages[:2], messages[2:]  # system + task
    cut = max(0, len(body) - keep_recent)
    while cut < len(body) and not isinstance(body[cut], AIMessage):  # never start the tail on an orphan ToolMessage
        cut += 1
    old, tail = body[:cut], body[cut:]
    transcript = "\n".join(f"{type(m).__name__}: {str(m.content)[:600]} {getattr(m, 'tool_calls', '') or ''}" for m in old)
    notes = notes_llm.invoke(
        [HumanMessage("Summarise this agent work log as compact progress notes: what was tried, key numbers/results, "
                      f"files created, what failed and why, and what remains. Max 250 words.\n\n{transcript[-20000:]}")]
    ).content
    return head + [HumanMessage(f"[Progress notes from your earlier work]\n{notes}")] + tail


# =============================================================================
# The agent loop
# =============================================================================

def run_agent(
    *, name: str, system: str, task: str, sandbox: Sandbox, log: EventLog, deadline: float,
    extra_tools: list[BaseTool] | None = None, submit_name: str = "submit_result",
    max_steps: int = 40, context_limit: int = 12000, thinking: bool = False,
) -> dict | None:
    """Run one tool-using agent until it calls `submit_name`, runs out of steps, or hits the deadline.
    Returns the submit tool's args (or None)."""
    tools = make_tools(sandbox, log, name, deadline) + (extra_tools or [])
    if submit_name != "submit_result":
        tools = [t for t in tools if t.name != "submit_result"]
    by_name = {t.name: t for t in tools}
    model = get_llm(thinking=thinking, max_tokens=4096).bind_tools(tools)
    notes_llm = get_llm(max_tokens=600)
    messages: list[BaseMessage] = [SystemMessage(system), HumanMessage(task)]
    warned, nudges = False, 0

    def force_submit(reason: str) -> dict | None:
        """Out of time/steps: one last call that may ONLY call the submit tool (tool_choice),
        so the agent reports partial results instead of returning nothing."""
        log(name, reason, "forcing a final submission")
        forced = get_llm(max_tokens=2048).bind_tools([by_name[submit_name]], tool_choice=submit_name)
        msgs = compact(messages, context_limit, notes_llm) + [
            HumanMessage(f"Time is up. Call {submit_name} NOW with your best result so far, and state clearly what is unfinished or unverified.")]
        try:
            ai = forced.invoke(msgs)
            if ai.tool_calls:
                log(name, "submit", json.dumps(ai.tool_calls[0]["args"])[:220])
                return ai.tool_calls[0]["args"]
        except Exception as e:  # noqa: BLE001
            log(name, "llm-error", repr(e)[:200])
        return None

    for step in range(1, max_steps + 1):
        remaining = deadline - time.time()
        if remaining <= 40:  # keep ~40s for the forced submission
            return force_submit("timeout")
        if remaining < 90 and not warned:
            messages.append(HumanMessage(f"TIME IS ALMOST UP ({remaining:.0f}s left). Call {submit_name} now with your best result."))
            warned = True

        messages = compact(messages, context_limit, notes_llm)
        try:
            ai: AIMessage = model.invoke(messages)
        except Exception as e:  # noqa: BLE001  e.g. context overflow; compact harder and retry once
            log(name, "llm-error", repr(e)[:200])
            messages = compact(messages, context_limit // 2, notes_llm)
            continue
        messages.append(ai)

        if not ai.tool_calls:
            if ai.content:
                log(name, "says", str(ai.content))
            nudges += 1
            if nudges > 2:  # keeps answering in prose: accept it rather than loop forever
                return {"answer": str(ai.content)[:500], "summary": "(agent never called the submit tool)", "files": []}
            messages.append(HumanMessage(f"Continue working with the tools. When finished, call {submit_name}."))
            continue
        nudges = 0

        for call in ai.tool_calls:
            if call["name"] == submit_name:
                log(name, "submit", json.dumps(call["args"])[:220])
                return call["args"]
            fn = by_name.get(call["name"])
            try:
                result = fn.invoke(call["args"]) if fn else f"Unknown tool {call['name']}"
            except Exception as e:  # noqa: BLE001  tool errors go back to the model
                result = f"Tool error: {e!r}"
            messages.append(ToolMessage(str(result), tool_call_id=call["id"], name=call["name"]))
        log(name, "step", f"{step}/{max_steps}, ~{count_tokens_approximately(messages)} tokens, {remaining:.0f}s left")

    return force_submit("step-limit")
