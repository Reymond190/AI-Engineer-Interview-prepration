"""
13 - Reflection / evaluator-optimizer loops
===========================================

Interview topics
----------------
- Evaluator-optimizer: a generator produces output, an evaluator grades it and
  gives feedback, loop until it passes or a budget runs out. Classic cycle,
  which is exactly what LangGraph adds over DAG-only frameworks.
- Self-critique by the same LLM gives modest gains. Grounded feedback is much
  stronger: run the code, run the tests, check the citation, validate the
  schema. Reflexion (Shinn et al. 2023) = verbal feedback from real signals
  stored and reused across attempts.
- Design rules:
    * hard cap on iterations (cost + infinite-loop protection)
    * structured grade (pass/fail + specific feedback), not free text
    * keep the best attempt, not just the last one
    * evaluator can be a stronger/thinking model; generator a fast one
- Related: LLM-as-judge for evals (lesson 21), "generate N + pick best"
  (best-of-n / self-consistency), and debate (lesson 17).

Run:  python langgraph_course/13_reflection_evaluator_optimizer.py
"""

import re
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from common import ask, banner, show, structured

TASK = (
    "Write a Python function `parse_duration(s: str) -> int` that converts strings like "
    "'1h30m', '45s', '2h', '1h2m3s', '90m' into total seconds. Raise ValueError on invalid input "
    "such as '' or '5x' or 'h'."
)

TESTS = [
    ("parse_duration('1h30m')", 5400),
    ("parse_duration('45s')", 45),
    ("parse_duration('2h')", 7200),
    ("parse_duration('1h2m3s')", 3723),
    ("parse_duration('90m')", 5400),
    ("parse_duration('')", ValueError),
    ("parse_duration('5x')", ValueError),
    ("parse_duration('h')", ValueError),
]

MAX_ITERS = 4


class Review(BaseModel):
    verdict: Literal["pass", "revise"]
    feedback: str = Field(description="Specific, actionable fixes. Empty if pass.")


# The tests already did the heavy lifting, so a fast-mode critic is enough here. With a
# hosted frontier model you might give the critic more reasoning budget than the generator.
critic = structured(Review)


class State(TypedDict, total=False):
    code: str
    feedback: str
    iteration: int
    passed: int
    best_code: str
    best_passed: int
    review: str


def generate(state: State) -> dict:
    prompt = f"{TASK}\nReturn only the code in a ```python block."
    if state.get("feedback"):
        prompt += f"\n\nYour previous attempt:\n```python\n{state['code']}\n```\nFeedback to address:\n{state['feedback']}"
    reply = ask(prompt, thinking=True)  # code generation is where reasoning pays off
    m = re.search(r"```(?:python)?\s*(.*?)```", reply, re.DOTALL)
    return {"code": (m.group(1) if m else reply).strip(), "iteration": state.get("iteration", 0) + 1}


def run_tests(state: State) -> dict:
    """Grounded evaluation: execute the code. (Sandbox this in real life!)"""
    ns: dict = {}
    failures = []
    try:
        exec(state["code"], ns)
    except Exception as e:  # noqa: BLE001
        failures.append(f"code does not run: {e!r}")
    else:
        for expr, expected in TESTS:
            try:
                got = eval(expr, ns)
                if expected is ValueError or got != expected:
                    failures.append(f"{expr} returned {got!r}, expected {expected.__name__ if expected is ValueError else expected}")
            except ValueError:
                if expected is not ValueError:
                    failures.append(f"{expr} raised ValueError, expected {expected}")
            except Exception as e:  # noqa: BLE001
                failures.append(f"{expr} raised {e!r}")
    passed = len(TESTS) - len(failures) if not failures or "does not run" not in failures[0] else 0
    show(f"iteration {state['iteration']}", f"{passed}/{len(TESTS)} tests pass")
    update: dict = {"passed": passed, "feedback": "\n".join(failures)}
    if passed > state.get("best_passed", -1):
        update |= {"best_code": state["code"], "best_passed": passed}
    return update


def review(state: State) -> dict:
    """LLM critic runs only once the tests pass: style / edge cases the tests miss."""
    r = critic.invoke(f"Review this solution to the task.\nTask: {TASK}\n```python\n{state['code']}\n```")
    show("critic", r)
    return {"review": r.verdict, "feedback": r.feedback if r.verdict == "revise" else ""}


def after_tests(state: State) -> Literal["generate", "review", "__end__"]:
    if state["passed"] == len(TESTS):
        return "review"
    return "generate" if state["iteration"] < MAX_ITERS else END


def after_review(state: State) -> Literal["generate", "__end__"]:
    return "generate" if state["review"] == "revise" and state["iteration"] < MAX_ITERS else END


builder = StateGraph(State)
builder.add_node("generate", generate)
builder.add_node("run_tests", run_tests)
builder.add_node("review", review)
builder.add_edge(START, "generate")
builder.add_edge("generate", "run_tests")
builder.add_conditional_edges("run_tests", after_tests)
builder.add_conditional_edges("review", after_review)
graph = builder.compile()


if __name__ == "__main__":
    banner("Generate -> run tests (grounded) -> LLM review -> revise")
    out = graph.invoke({}, {"recursion_limit": 30})
    show("iterations", out["iteration"])
    show("best score", f"{out['best_passed']}/{len(TESTS)}")
    print(out["best_code"])
