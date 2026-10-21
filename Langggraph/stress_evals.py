"""
stress_evals.py - raw-capability evals WITHOUT the course's guardrails.

The lessons are tuned (prompts, code-level checks, thinking where needed) so
they pass reliably. That makes them a poor way to compare models. These evals
remove the scaffolding and replay the exact failure modes we hit while
building the course, so a stronger model shows up as a higher score.

    python langgraph_course/stress_evals.py --n 5 --save stress_qwen3-8b.json
"""

from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
import time
from pathlib import Path

from langchain_core.messages import HumanMessage

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import MODEL, ask, get_llm, make_react_agent, structured  # noqa: E402

sup = importlib.import_module("05_supervisor_multi_agent")
rag = importlib.import_module("17_agentic_rag")
refl = importlib.import_module("13_reflection_evaluator_optimizer")
pe = importlib.import_module("14_plan_and_execute")


def supervisor_first_step() -> bool:
    """Fast-mode supervisor, empty work log: should NOT jump to the writer."""
    llm = structured(sup.SupervisorDecision)  # fast mode, no code guardrails
    d = llm.invoke(
        "You are a supervisor coordinating workers: researcher, analyst, writer.\n"
        "- researcher: looks facts up in internal notes\n- analyst: does arithmetic\n"
        "- writer: has no tools or knowledge; writes only from findings in the work log\n"
        "Task: Briefly compare LangGraph and CrewAI, and compute 3*12*40 LLM calls per day.\n"
        "Work done so far:\n(nothing yet)\n\nPick the single next worker."
    )
    return d.next in ("researcher", "analyst")


ROUTING = [
    ("How many vacation days can I carry over?", "company_docs"),
    ("Can I get money back for buying a desk chair for working from home?", "company_docs"),
    ("What is the capital of Australia?", "direct"),
    ("Do I need approval for a 700 EUR flight?", "company_docs"),
    ("Who painted the Mona Lisa?", "direct"),
]


def router_zero_shot() -> float:
    """Router with only the field description, no few-shot examples."""
    hits = sum(rag.router.invoke(f"Route this question: {q}").datasource == want for q, want in ROUTING)
    return hits / len(ROUTING)


def codegen_one_shot() -> float:
    """Fast mode, single attempt, fraction of the 8 tests passed."""
    reply = ask(f"{refl.TASK}\nReturn only the code in a ```python block.")
    m = re.search(r"```(?:python)?\s*(.*?)```", reply, re.DOTALL)
    out = refl.run_tests({"code": (m.group(1) if m else reply), "iteration": 0})
    return out["passed"] / len(refl.TESTS)


def single_agent_long_task() -> bool:
    """The plan-and-execute task done by ONE ReAct agent, no planner/replanner."""
    agent = make_react_agent([pe.get_price, pe.get_tax_rate, pe.calculator], "Use tools for every number.")
    task = "Total cost including tax to equip 3 new hires in Germany (de), each with a laptop, a monitor and a dock."
    out = agent.invoke({"messages": [HumanMessage(task)]}, {"recursion_limit": 30})
    return bool(re.search(r"7,?068\.6", out["messages"][-1].content))


def judge_without_rubric_hint() -> bool:
    """LLM-judge must mark a WRONG answer as incorrect (hallucination check)."""
    from pydantic import BaseModel

    class G(BaseModel):
        reasoning: str
        correct: bool

    g = structured(G).invoke("Question: What is 19.99 * 35?\nExpected: 699.65\nAnswer to grade: The result is 689.65.\nIs the answer correct?")
    return g.correct is False


EVALS = {
    "supervisor picks a worker first (fast, no guardrail)": supervisor_first_step,
    "RAG router zero-shot accuracy": router_zero_shot,
    "one-shot codegen tests passed (fast)": codegen_one_shot,
    "single ReAct agent solves 7-step costing task": single_agent_long_task,
    "judge catches a wrong number": judge_without_rubric_hint,
}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--save")
    args = ap.parse_args()

    results = {}
    for name, fn in EVALS.items():
        scores, t0 = [], time.perf_counter()
        for _ in range(args.n):
            try:
                scores.append(float(fn()))
            except Exception as e:  # noqa: BLE001  a crash counts as a fail
                print(f"  ! {name}: {e!r}"[:200])
                scores.append(0.0)
        results[name] = {"score": round(sum(scores) / len(scores), 2), "seconds": round((time.perf_counter() - t0) / args.n, 1)}
        print(f"{name:<55} {results[name]['score']:.2f}  ({results[name]['seconds']}s/run)", flush=True)

    print(f"{'MEAN (' + MODEL + ')':<55} {sum(r['score'] for r in results.values()) / len(results):.2f}")
    if args.save:
        Path(args.save).write_text(json.dumps({"model": MODEL, "results": results}, indent=2))
