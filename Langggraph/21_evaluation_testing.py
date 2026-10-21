"""
21 - Testing and evaluating agents
==================================

Interview topics
----------------
- Three layers, like the test pyramid:
    1. Deterministic unit tests (fast, run in CI): test routing functions and
       nodes as plain functions; replace the LLM with a FAKE chat model that
       returns scripted AIMessages (including tool_calls). Tests graph wiring,
       state updates, error paths, with zero tokens spent.
    2. Trajectory evaluation: run the real agent on a dataset and compare the
       sequence of tool calls with a reference (strict order / unordered /
       subset / superset). Catches "right answer, wrong path" (and vice versa).
    3. Output evaluation: exact match / code checks where possible, otherwise
       LLM-as-judge with a rubric and structured scores. Calibrate the judge
       against human labels; watch for position/verbosity/self-preference bias.
- Tooling: LangSmith datasets + experiments, `openevals` / `agentevals`
  packages (trajectory match, LLM-as-judge), pytest. Track cost, latency,
  number of steps and tool error rate alongside quality.
- Online: tracing every run, sampling production traces into eval datasets,
  regression-testing prompt/model changes before shipping.

Run:  python langgraph_course/21_evaluation_testing.py
      python -m pytest langgraph_course/21_evaluation_testing.py -k unit -q   (no LLM needed)
"""

import importlib
from typing import Any, Literal

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field

from common import banner, show, structured

react = importlib.import_module("03_tool_calling_react")  # the agent under test


# =============================================================================
# 1. Deterministic unit tests with a fake model
# =============================================================================

class FakeToolModel(GenericFakeChatModel):
    """GenericFakeChatModel that tolerates bind_tools (returns itself)."""

    def bind_tools(self, tools: Any, **kwargs: Any) -> "FakeToolModel":
        return self


def scripted(*messages: AIMessage) -> FakeToolModel:
    return FakeToolModel(messages=iter(messages))


def test_unit_tools_condition_routes_to_tools() -> None:
    from langgraph.prebuilt import tools_condition

    with_call = AIMessage("", tool_calls=[{"name": "calculator", "args": {"expression": "1+1"}, "id": "1"}])
    assert tools_condition({"messages": [with_call]}) == "tools"
    assert tools_condition({"messages": [AIMessage("done")]}) == "__end__"


def test_unit_react_graph_with_fake_llm(monkeypatch=None) -> None:
    fake = scripted(
        AIMessage("", tool_calls=[{"name": "lookup_product", "args": {"name": "gadget"}, "id": "c1"}]),
        AIMessage("", tool_calls=[{"name": "calculator", "args": {"expression": "19.99 * 35"}, "id": "c2"}]),
        AIMessage("Total stock value of gadgets is $699.65."),
    )
    original = react.model
    react.model = fake  # the agent node reads the module-level model
    try:
        out = react.graph.invoke({"messages": [HumanMessage("value of gadgets?")]})
    finally:
        react.model = original
    tool_outputs = [m.content for m in out["messages"] if m.type == "tool"]
    assert tool_outputs == ["gadget: price=$19.99, qty=35", "699.65"], tool_outputs
    assert out["messages"][-1].content.endswith("$699.65.")


def test_unit_unknown_tool_args_produce_recoverable_error() -> None:
    fake = scripted(
        AIMessage("", tool_calls=[{"name": "calculator", "args": {"wrong_arg": "1"}, "id": "c1"}]),
        AIMessage("Sorry, let me retry."),
    )
    original = react.model
    react.model = fake
    try:
        out = react.graph.invoke({"messages": [HumanMessage("x")]})
    finally:
        react.model = original
    err = next(m for m in out["messages"] if m.type == "tool")
    assert err.status == "error", "ToolNode(handle_tool_errors=True) should return an error ToolMessage, not raise"


# =============================================================================
# 2. Trajectory evaluation on the real agent
# =============================================================================

def tool_trajectory(messages: list) -> list[str]:
    return [c["name"] for m in messages if isinstance(m, AIMessage) for c in m.tool_calls]


def trajectory_match(actual: list[str], reference: list[str], mode: Literal["strict", "unordered", "superset"]) -> bool:
    if mode == "strict":
        return actual == reference
    if mode == "unordered":
        return sorted(actual) == sorted(reference)
    return all(actual.count(t) >= reference.count(t) for t in set(reference))  # did at least these calls


DATASET = [
    {"input": "What is the total stock value of widgets?", "reference_tools": ["lookup_product", "calculator"], "expected": "510"},
    {"input": "Is the gizmo in stock?", "reference_tools": ["lookup_product"], "expected": "no"},
    {"input": "What is 12 squared?", "reference_tools": ["calculator"], "expected": "144"},
]


# =============================================================================
# 3. LLM-as-judge with a rubric
# =============================================================================

class Grade(BaseModel):
    # Field ORDER matters: the model generates left to right, so reason first, then score.
    reasoning: str = Field(description="Brief justification, written before the scores")
    correct: bool = Field(description="Does the answer contain the expected fact?")
    helpfulness: int = Field(ge=1, le=5, description="1 = useless, 3 = correct but padded/unclear, 5 = direct, complete and concise")


judge = structured(Grade, thinking=True)


def run_evals() -> None:
    rows = []
    for ex in DATASET:
        out = react.graph.invoke({"messages": [HumanMessage(ex["input"])]}, {"recursion_limit": 12})
        answer = out["messages"][-1].content
        traj = tool_trajectory(out["messages"])
        grade = judge.invoke(
            f"Question: {ex['input']}\nExpected key fact: {ex['expected']}\nAnswer to grade: {answer}\n"
            "Grade the answer. Be strict about correctness."
        )
        rows.append(
            {
                "input": ex["input"],
                "trajectory": traj,
                "traj_superset": trajectory_match(traj, ex["reference_tools"], "superset"),
                "traj_strict": trajectory_match(traj, ex["reference_tools"], "strict"),
                "steps": len(out["messages"]),
                "correct": grade.correct,
                "helpfulness": grade.helpfulness,
            }
        )
        show(ex["input"], rows[-1])
    n = len(rows)
    banner("Summary")
    show("accuracy", f"{sum(r['correct'] for r in rows)}/{n}")
    show("trajectory superset match", f"{sum(r['traj_superset'] for r in rows)}/{n}")
    show("trajectory strict match", f"{sum(r['traj_strict'] for r in rows)}/{n}")
    show("avg helpfulness", round(sum(r["helpfulness"] for r in rows) / n, 2))


if __name__ == "__main__":
    banner("1. Unit tests (fake LLM, no tokens)")
    for t in [test_unit_tools_condition_routes_to_tools, test_unit_react_graph_with_fake_llm, test_unit_unknown_tool_args_produce_recoverable_error]:
        t()
        print(f"  PASS {t.__name__}")

    banner("2+3. Trajectory + LLM-as-judge evals on the real agent")
    run_evals()
