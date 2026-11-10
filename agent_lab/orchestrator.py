"""
orchestrator.py - planner -> N parallel sandboxed solvers -> verifier -> decide -> report

    START -> plan -> [Send solve x N] -> verify -> decide --(retry)--> plan
                                                         `-> report -> END

Design choices (each is a known lever for hard problems):
  * strategy diversity: the planner gives every solver a DIFFERENT approach;
    identical parallel agents make correlated mistakes
  * isolation: one container + one workspace per solver, no shared context
  * independent verification: the verifier gets read-only access to every
    solver's workspace and must re-run / re-derive, not just read claims
  * consensus among VERIFIED answers, retry with feedback, hard time budget
"""

from __future__ import annotations

import json
import operator
import re
import time
from collections import Counter
from pathlib import Path
from typing import Annotated, Literal, TypedDict

from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field

from core import EventLog, run_agent
from sandbox import Sandbox
from common import get_llm, structured  # noqa: E402  (path set up by core)

# =============================================================================
# Prompts
# =============================================================================

SOLVER_SYSTEM = """You are an autonomous problem solver working inside a Linux sandbox (Docker).
Your working directory is /workspace. You have tools to run Python and bash, read/write files,
fetch forex candles from MetaTrader 5 (fetch_fx), and submit your result.

How to work:
- Solve by WRITING AND RUNNING CODE. Every number you report must come from code you ran.
- Keep prints small (summaries, head/tail); save big outputs and charts to files.
- Test on small cases first, then scale. Prefer fast algorithms (numpy, numba, sieves, DP).
- Check your own result a second way when possible (brute force on small inputs, a different method).
- When done, call submit_result with the exact answer, a short method summary, and supporting files.
- If you get stuck, simplify, or try another approach. Never invent results."""

FOREX_RULES = """Getting data: `fetch_fx` is a TOOL you call directly (like run_python), NOT a Python module; it cannot
be imported. Call the tool once per dataset; it saves a CSV under /workspace/data/ and returns the file name.
Then load it in your scripts with pandas: pd.read_csv('data/<file>.csv', parse_dates=['time']).

Forex/data-science rules (the verifier will check these):
- No look-ahead: features at time t may only use data <= t; split train/test BY TIME (walk-forward).
- Include transaction costs (use the 'spread' column or >= 1 pip) in any strategy backtest.
- Report out-of-sample metrics and compare with a naive baseline (e.g. 50% direction, random walk).
- Treat small edges skeptically: report t-stats / confidence intervals, number of trades, and multiple-testing caveats.
- Predictions are probabilistic research output, not trading advice."""

VERIFIER_SYSTEM = """You are a rigorous, skeptical VERIFIER in a Linux sandbox.
Each candidate solver's workspace is mounted read-only at /candidates/<solver-name>/ (code in snippets/, data in data/, outputs).
Reuse the candidates' data files (e.g. pd.read_csv('/candidates/solver-1/data/<file>.csv')) instead of re-downloading;
tools like fetch_fx are called directly as tools, never imported in Python.
Budget your time: check the most important claim of each candidate first, then submit_verdicts well before time runs out.
For EACH candidate, independently check the claimed answer:
- Re-run or re-derive it with your own code in /workspace (copy files if needed; candidates are read-only).
- For math: verify with an independent method (brute force on small cases, different algorithm).
- For data/forex: look for look-ahead bias, train/test leakage, missing costs, overfitting, unsupported claims.
Then call submit_verdicts with one verdict per candidate. 'pass' only if you reproduced or independently confirmed it."""


# =============================================================================
# Schemas
# =============================================================================

class Strategy(BaseModel):
    name: str = Field(description="Short label, e.g. 'sieve+numba' or 'hour-of-day seasonality'")
    approach: str = Field(description="2-4 sentences: the concrete method this solver should try")


class Plan(BaseModel):
    task_kind: Literal["math", "data", "forex", "other"]
    success_criteria: str = Field(description="How to tell a correct/complete result")
    strategies: list[Strategy] = Field(description="One DIFFERENT strategy per solver")


class Verdict(BaseModel):
    candidate: str = Field(description="Solver name, e.g. solver-1")
    verdict: Literal["pass", "fail", "uncertain"]
    evidence: str = Field(description="What you ran/checked and what you found (1-3 sentences)")
    verified_answer: str = Field(default="", description="The answer you confirmed, if any")


class Verdicts(BaseModel):
    verdicts: list[Verdict]


class Candidate(TypedDict):
    solver: str
    round: int
    strategy: str
    answer: str
    summary: str
    files: list[str]
    seconds: float


class State(TypedDict, total=False):
    task: str
    n_solvers: int
    deadline: float
    round: int
    max_rounds: int
    plan: Plan
    feedback: str
    candidates: Annotated[list[Candidate], operator.add]
    verdicts: Annotated[list[dict], operator.add]
    final: dict


class SolverInput(TypedDict):
    task: str
    solver: str
    strategy: Strategy
    feedback: str
    deadline: float
    round: int


# =============================================================================
# Graph factory (closes over run_dir + log so nodes stay simple)
# =============================================================================

def build_graph(run_dir: Path, log: EventLog, verify: bool = True):
    # Thinking mode plans better, but Qwen3.5 can reason past its token budget and
    # return nothing (LengthFinishReasonError). Fall back to fast mode if so.
    planner = structured(Plan, thinking=True, max_tokens=4096).with_fallbacks([structured(Plan)])

    def plan(state: State) -> dict:
        n, rnd = state["n_solvers"], state.get("round", 0) + 1
        prompt = (f"Task:\n{state['task']}\n\nPlan for {n} independent solvers working in parallel, each in its own "
                  f"Python sandbox. Give exactly {n} DIFFERENT strategies (different algorithms, data views, or models) "
                  "so their errors are independent.")
        if state.get("feedback"):
            prompt += f"\n\nThis is retry round {rnd}. Previous attempts and verifier feedback:\n{state['feedback']}\nAvoid what failed."
        p = planner.invoke(prompt)
        p.strategies = (p.strategies * n)[:n]  # pad / trim to exactly n
        log("planner", "plan", f"round {rnd}: {p.task_kind} | " + " / ".join(s.name for s in p.strategies))
        (run_dir / f"plan_round{rnd}.json").write_text(p.model_dump_json(indent=2))
        return {"plan": p, "round": rnd}

    def fan_out(state: State) -> list[Send]:
        rnd = state["round"]
        # Each round gets the time left, minus a reserve for verification + report:
        # 30% of the remaining time (at least 4 minutes), so the verifier isn't starved.
        remaining = state["deadline"] - time.time()
        reserve = max(240, 0.3 * remaining) if verify else 30
        solve_deadline = state["deadline"] - reserve
        return [Send("solve", {"task": state["task"], "solver": f"solver-{i + 1}" + (f"r{rnd}" if rnd > 1 else ""),
                               "strategy": s, "feedback": state.get("feedback", ""), "deadline": solve_deadline, "round": rnd})
                for i, s in enumerate(state["plan"].strategies)]

    def solve(inp: SolverInput) -> dict:
        name, t0 = inp["solver"], time.time()
        task = (f"TASK:\n{inp['task']}\n\nYOUR STRATEGY ({inp['strategy'].name}): {inp['strategy'].approach}\n"
                f"Time budget: {max(0, inp['deadline'] - t0) / 60:.0f} minutes.")
        if inp["feedback"]:
            task += f"\n\nEarlier attempts by other solvers (learn from them):\n{inp['feedback']}"
        system = SOLVER_SYSTEM + ("\n\n" + FOREX_RULES if re.search(r"forex|fx|eurusd|gbpusd|usdjpy|market", inp["task"], re.I) else "")
        log(name, "start", f"{inp['strategy'].name}: {inp['strategy'].approach}")
        with Sandbox(run_dir / name, name) as sb:
            sub = run_agent(name=name, system=system, task=task, sandbox=sb, log=log, deadline=inp["deadline"], max_steps=60)
        sub = sub or {"answer": "", "summary": "no submission (timeout / gave up)", "files": []}
        return {"candidates": [{"solver": name, "round": inp["round"], "strategy": inp["strategy"].name,
                                "answer": str(sub.get("answer", "")), "summary": str(sub.get("summary", "")),
                                "files": sub.get("files") or [], "seconds": round(time.time() - t0, 1)}]}

    def verify_node(state: State) -> dict:
        current = [c for c in state["candidates"] if c["round"] == state["round"] and c["answer"]]
        if not current:
            return {"verdicts": []}
        listing = "\n\n".join(f"### {c['solver']} (strategy: {c['strategy']})\nANSWER: {c['answer']}\nSUMMARY: {c['summary']}\nFILES: {c['files']}"
                              for c in current)
        result: dict = {}

        @tool(args_schema=Verdicts)
        def submit_verdicts(verdicts: list) -> str:
            """Submit one verdict per candidate. Ends your turn."""
            return "ok"

        mounts = {run_dir / c["solver"]: f"/candidates/{c['solver']}" for c in current}
        deadline = max(state["deadline"] - 30, time.time() + 120)
        with Sandbox(run_dir / f"verifier-r{state['round']}", "verifier", readonly_mounts=mounts) as sb:
            sub = run_agent(name="verifier", system=VERIFIER_SYSTEM, sandbox=sb, log=log, deadline=deadline,
                            task=f"TASK the solvers worked on:\n{state['task']}\n\nCANDIDATES:\n{listing}",
                            extra_tools=[submit_verdicts], submit_name="submit_verdicts", max_steps=30)
        for v in (sub or {}).get("verdicts", []):
            v = v if isinstance(v, dict) else v.model_dump()
            v["round"] = state["round"]
            result.setdefault("verdicts", []).append(v)
            log("verifier", v.get("verdict", "?"), f"{v.get('candidate')}: {v.get('evidence', '')}")
        return result or {"verdicts": [{"candidate": c["solver"], "verdict": "uncertain", "evidence": "verifier did not submit",
                                        "verified_answer": "", "round": state["round"]} for c in current]}

    def decide(state: State) -> dict:
        rnd = state["round"]
        cands = {c["solver"]: c for c in state["candidates"] if c["round"] == rnd}
        verdicts = {v["candidate"]: v for v in state.get("verdicts", []) if v.get("round") == rnd}
        passed = [cands[s]["answer"] for s, v in verdicts.items() if v["verdict"] == "pass" and s in cands]
        answers = [c["answer"] for c in cands.values() if c["answer"]]
        if passed:
            answer, n = Counter(norm(a) for a in passed).most_common(1)[0]
            final = {"answer": answer, "status": "verified", "support": f"{n} verified solver(s)", "round": rnd}
        elif answers and not verify:
            answer, n = Counter(norm(a) for a in answers).most_common(1)[0]
            final = {"answer": answer, "status": "unverified (no verifier)", "support": f"{n}/{len(answers)} solvers agree", "round": rnd}
        else:
            time_left = state["deadline"] - time.time()
            if rnd < state["max_rounds"] and time_left > 300:
                fb = "\n".join(f"- {s} [{cands[s]['strategy']}] answered {cands[s]['answer']!r}: "
                               f"{verdicts.get(s, {}).get('verdict', 'n/a')} - {verdicts.get(s, {}).get('evidence', cands[s]['summary'][:200])}"
                               for s in cands)
                log("orchestrator", "retry", f"no verified answer in round {rnd}; {time_left:.0f}s left")
                return {"feedback": fb}
            answer, n = (Counter(norm(a) for a in answers).most_common(1)[0] if answers else ("", 0))
            final = {"answer": answer, "status": "unverified", "support": f"{n}/{len(answers)} solvers agree", "round": rnd}
        log("orchestrator", "decision", json.dumps(final))
        return {"final": final}

    def route_after_decide(state: State) -> Literal["plan", "report"]:
        return "report" if state.get("final") else "plan"

    def report(state: State) -> dict:
        final = state["final"]
        lines = [f"# Result\n\n**Answer:** {final['answer'] or '(none)'}  \n**Status:** {final['status']} ({final['support']})\n",
                 f"## Task\n{state['task']}\n", "## Solvers"]
        for c in state["candidates"]:
            v = next((v for v in state.get("verdicts", []) if v["candidate"] == c["solver"]), {})
            lines.append(f"### {c['solver']} - {c['strategy']} ({c['seconds']:.0f}s)\n- answer: `{c['answer']}`\n"
                         f"- verifier: {v.get('verdict', 'n/a')} - {v.get('evidence', '')}\n- files: {c['files']}\n\n{c['summary']}\n")
        if state["plan"].task_kind in ("data", "forex", "other"):
            synthesis = get_llm(max_tokens=1500).invoke(
                "Write a concise research summary (findings, evidence strength, caveats) from these solver reports and "
                "verifier verdicts. Be explicit about what was NOT verified.\n\n" + "\n".join(lines)).content
            lines.insert(1, f"## Synthesis\n{synthesis}\n")
        (run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
        return {}

    g = StateGraph(State)
    g.add_node("plan", plan)
    g.add_node("solve", solve)
    g.add_node("verify", verify_node if verify else (lambda s: {"verdicts": []}))
    g.add_node("decide", decide)
    g.add_node("report", report)
    g.add_edge(START, "plan")
    g.add_conditional_edges("plan", fan_out, ["solve"])
    g.add_edge("solve", "verify")
    g.add_edge("verify", "decide")
    g.add_conditional_edges("decide", route_after_decide)
    g.add_edge("report", END)
    return g.compile()


def norm(answer: str) -> str:
    """Normalise answers for voting: '1,234' == '1234' == ' 1234. '"""
    a = answer.strip().rstrip(".")
    return a.replace(",", "") if re.fullmatch(r"[-\d,]+(\.\d+)?", a) else a
