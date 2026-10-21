"""
14 - Plan-and-execute (with replanning)
=======================================

Interview topics
----------------
- ReAct decides one step at a time; it can wander on long tasks and every
  step pays for the full reasoning. Plan-and-execute separates:
    planner   (strong / thinking model) writes an explicit multi-step plan
    executor  (fast model + tools) does one step at a time
    replanner looks at results so far and either finishes or revises the
              REMAINING plan
- Benefits: long-horizon coherence, cheaper executor calls, the plan is
  inspectable (and can be shown to / edited by a human, lesson 10).
  Costs: rigid if the replanner is weak; more total calls for easy tasks.
- Variants: ReWOO (plan with variable placeholders, execute all tools, solve
  once), LLMCompiler (plan as a DAG, run independent steps in parallel), and
  the todo-list tool that "deep agents" use (lesson 20).

Run:  python langgraph_course/14_plan_and_execute.py
"""

import operator
from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from common import banner, make_react_agent, show, structured

# --- Tools for the executor -----------------------------------------------------

PRICES = {"laptop": 1450.0, "monitor": 320.0, "dock": 210.0, "headset": 95.0}
TAX = {"de": 0.19, "fr": 0.20, "uk": 0.20, "us-ca": 0.0725}


@tool
def get_price(item: str) -> str:
    """Unit price in USD of an equipment item (laptop, monitor, dock, headset)."""
    p = PRICES.get(item.lower().rstrip("s"))
    return f"{item}: ${p}" if p else f"unknown item; known: {list(PRICES)}"


@tool
def get_tax_rate(region: str) -> str:
    """Sales tax rate for a region code: de, fr, uk, us-ca."""
    r = TAX.get(region.lower())
    return f"{region}: {r}" if r is not None else f"unknown region; known: {list(TAX)}"


@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression like '3 * (1450 + 320) * 1.19'."""
    return str(round(eval(expression, {"__builtins__": {}}, {}), 2))


executor = make_react_agent(
    [get_price, get_tax_rate, calculator],
    "Do ONLY the one step you are given, nothing beyond it. Use tools for every number. Report the result in one sentence.",
    name="executor",
)


# --- Planner / replanner ------------------------------------------------------

class Plan(BaseModel):
    steps: list[str] = Field(description="Ordered, concrete steps. Each step is doable with one or two tool calls.")


class Replan(BaseModel):
    done: bool = Field(description="True if the results so far fully answer the objective")
    final_answer: str = Field(description="The answer if done, else empty")
    remaining_steps: list[str] = Field(description="Updated remaining steps if not done, else empty")


planner = structured(Plan, thinking=True)  # spend reasoning budget once, on the plan
replanner = structured(Replan)  # frequent, simpler decision -> fast mode


class State(TypedDict, total=False):
    objective: str
    plan: list[str]
    past_steps: Annotated[list[tuple[str, str]], operator.add]
    answer: str


def plan_step(state: State) -> dict:
    plan = planner.invoke(f"Make a step-by-step plan for: {state['objective']}\nTools available: get_price, get_tax_rate, calculator.")
    show("plan", plan.steps)
    return {"plan": plan.steps}


def execute_step(state: State) -> dict:
    step = state["plan"][0]
    done = "\n".join(f"- {s}: {r}" for s, r in state.get("past_steps", []))
    out = executor.invoke({"messages": [HumanMessage(f"Results so far:\n{done or '(none)'}\n\nYour step: {step}")]}, {"recursion_limit": 12})
    result = out["messages"][-1].content
    show("executed", f"{step} -> {result[:200]}")
    return {"past_steps": [(step, result)]}


def replan_step(state: State) -> dict:
    done = "\n".join(f"- {s}: {r}" for s, r in state["past_steps"])
    r = replanner.invoke(
        f"Objective: {state['objective']}\nOriginal plan: {state['plan']}\nCompleted steps and results:\n{done}\n\n"
        "Set done=true only if a completed step's result already contains the final number; copy it verbatim. "
        "Never do arithmetic yourself: if a total is still missing, add a calculator step. Otherwise list only the remaining steps."
    )
    if r.done or not r.remaining_steps:
        return {"answer": r.final_answer or done, "plan": []}
    show("replanned", r.remaining_steps)
    return {"plan": r.remaining_steps}


def should_continue(state: State) -> str:
    return END if state.get("answer") or len(state["past_steps"]) >= 8 else "execute"


builder = StateGraph(State)
builder.add_node("planner", plan_step)
builder.add_node("execute", execute_step)
builder.add_node("replan", replan_step)
builder.add_edge(START, "planner")
builder.add_edge("planner", "execute")
builder.add_edge("execute", "replan")
builder.add_conditional_edges("replan", should_continue, ["execute", END])
graph = builder.compile()


if __name__ == "__main__":
    banner("Plan -> execute -> replan")
    objective = "Total cost including tax to equip 3 new hires in Germany (de), each with a laptop, a monitor and a dock."
    out = graph.invoke({"objective": objective}, {"recursion_limit": 40})
    banner("Answer")
    print(out["answer"])
    show("expected", f"{round(3 * (1450 + 320 + 210) * 1.19, 2)}")
