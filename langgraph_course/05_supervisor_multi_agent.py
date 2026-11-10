"""
05 - Supervisor (orchestrator-worker) multi-agent system
========================================================

Interview topics
----------------
- Supervisor pattern: one LLM "manager" decides which specialist runs next,
  workers report back to it, it decides when the job is done.

        START -> supervisor -> {researcher | analyst | writer} -> supervisor ... -> END

- Why multi-agent at all? Each worker gets a small, focused prompt and
  toolset (fewer tools = better tool selection), its own context window,
  and can be developed/evaluated independently. Cost: more LLM calls,
  more latency, and errors compound across hand-offs. Start with a single
  agent and split only when tool count or prompt complexity hurts.
- Context engineering: workers return a *summary*, not their whole
  scratchpad. The supervisor sees a compact log. This keeps every context
  small (critical here: 8k window).
- Supervisor output is structured (next worker + instruction), and there is
  a hard cap on rounds to prevent infinite delegation.
- Observed while building this with Qwen3-8B: in fast mode the supervisor kept
  sending everything straight to the writer (who then made facts up). Fixes, in
  order of strength: describe what each worker can and CANNOT do; enforce
  preconditions in code (no writer before findings, no FINISH before an answer);
  give the routing decision a reasoning budget (thinking=True).
- Variants to mention:
    * tool-calling supervisor: each worker wrapped as a tool the supervisor
      calls ("agents as tools"). Recommended default in LangChain docs now.
    * `langgraph-supervisor` package: create_supervisor(agents, model).
    * hierarchical: supervisors of supervisors (lesson 07).

Run:  python langgraph_course/05_supervisor_multi_agent.py
"""

import operator
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from pydantic import BaseModel, Field

from common import ask, banner, make_react_agent, show, structured

# --- Tools for the workers --------------------------------------------------------

FACTS = {
    "langgraph": "LangGraph is a low-level orchestration framework for stateful, long-running agents; v1.0 shipped Oct 2025.",
    "crewai": "CrewAI is a role-based multi-agent framework: you define agents with roles/goals and tasks for a crew.",
    "autogen": "AutoGen (Microsoft) models agents as conversational participants; merged into Microsoft Agent Framework in 2025.",
}


@tool
def search_notes(query: str) -> str:
    """Search internal notes about agent frameworks. Query with a framework name like 'crewai'."""
    hits = [v for k, v in FACTS.items() if k in query.lower().replace(" ", "")]
    return "\n".join(hits) or f"No notes found. Try one of: {', '.join(FACTS)}"


@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression like '3 * 12 + 4'."""
    return str(eval(expression, {"__builtins__": {}}, {}))


researcher = make_react_agent([search_notes], "You are a researcher. Use search_notes, then give a short factual summary.", name="researcher")
analyst = make_react_agent([calculator], "You are a quantitative analyst. Use the calculator for all arithmetic.", name="analyst")

WORKERS = ("researcher", "analyst", "writer")


# --- Shared state -----------------------------------------------------------------

class WorkLog(TypedDict):
    worker: str
    instruction: str
    result: str


class SupervisorDecision(BaseModel):
    next: Literal["researcher", "analyst", "writer", "FINISH"] = Field(
        description="researcher = look up facts, analyst = arithmetic, writer = write the final answer, FINISH = writer already produced the answer"
    )
    instruction: str = Field(description="One-sentence instruction for that worker")


class State(TypedDict, total=False):
    task: str
    log: Annotated[list[WorkLog], operator.add]  # append-only work history
    current: SupervisorDecision  # assignment being worked on (overwritten each round)
    rounds: int
    final: str


def format_log(log: list[WorkLog]) -> str:
    return "\n".join(f"- {e['worker']} ({e['instruction']}): {e['result']}" for e in log) or "(nothing yet)"


# --- Nodes ------------------------------------------------------------------------

MAX_ROUNDS = 5
supervisor_llm = structured(SupervisorDecision, thinking=True)  # routing = planning: worth the reasoning budget


def supervisor(state: State) -> Command[Literal["researcher", "analyst", "writer", "__end__"]]:
    rounds = state.get("rounds", 0)
    if rounds >= MAX_ROUNDS:
        return Command(goto=END)
    decision = supervisor_llm.invoke(
        # Tell the supervisor what each worker can and CANNOT do; otherwise it
        # assumes the writer can just "know" the facts and do the maths.
        "You are a supervisor coordinating workers:\n"
        "- researcher: looks facts up in internal notes (the only source of facts)\n"
        "- analyst: does arithmetic with a calculator\n"
        "- writer: has NO tools and NO knowledge; writes only from findings already in the work log\n"
        f"Task: {state['task']}\n\nWork done so far:\n{format_log(state.get('log', []))}\n\n"
        "Pick the single next worker. Use the writer last, once facts and numbers are ready. "
        "Choose FINISH only after the writer has answered."
    )
    show(f"supervisor round {rounds + 1}", f"{decision.next}: {decision.instruction}")
    if decision.next == "FINISH":
        if state.get("final"):
            return Command(goto=END)
        # Invariants belong in code, not in the prompt: never finish without an answer.
        show("guardrail", "FINISH rejected (writer has not run) -> writer")
        decision = SupervisorDecision(next="writer", instruction="Write the final answer from the findings.")
    if decision.next == "writer" and not any(e["worker"] != "writer" for e in state.get("log", [])):
        # Precondition, also in code: the writer has nothing to write from yet.
        show("guardrail", "writer rejected (no findings yet) -> researcher")
        decision = SupervisorDecision(next="researcher", instruction=f"Find the facts needed for: {state['task']}")
    return Command(goto=decision.next, update={"rounds": rounds + 1, "current": decision})


def run_subagent(agent, state: State) -> str:
    instruction = state["current"].instruction
    # Fresh, isolated context for the sub-agent: only its instruction (+ the task).
    out = agent.invoke({"messages": [HumanMessage(f"Overall task: {state['task']}\nYour job: {instruction}")]}, {"recursion_limit": 10})
    return out["messages"][-1].content


def make_worker(name: str):
    def node(state: State) -> Command[Literal["supervisor"]]:
        if name == "writer":
            result = ask(f"Write the final answer to: {state['task']}\nUse only these findings:\n{format_log(state.get('log', []))}\nMax 4 sentences.")
        else:
            result = run_subagent(researcher if name == "researcher" else analyst, state)
        show(name, result)
        # Workers report a compact result, not their whole scratchpad.
        entry = {"worker": name, "instruction": state["current"].instruction, "result": result[:600]}
        update: dict = {"log": [entry]}
        if name == "writer":
            update["final"] = result
        return Command(goto="supervisor", update=update)

    return node


builder = StateGraph(State)
builder.add_node("supervisor", supervisor)
for w in WORKERS:
    builder.add_node(w, make_worker(w))
builder.add_edge(START, "supervisor")
graph = builder.compile()


if __name__ == "__main__":
    banner("Supervisor multi-agent")
    task = (
        "Briefly compare LangGraph and CrewAI. Also: if a team runs 3 agents, each making 12 LLM calls "
        "per task, and does 40 tasks a day, how many LLM calls per day is that?"
    )
    out = graph.invoke({"task": task}, {"recursion_limit": 30})
    banner("Final answer")
    print(out.get("final", "(no final answer; hit MAX_ROUNDS)"))
