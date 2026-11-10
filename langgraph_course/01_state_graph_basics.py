"""
01 - StateGraph fundamentals
============================

Interview topics
----------------
- A LangGraph app is a *state machine*: State (a TypedDict / Pydantic model),
  Nodes (functions state -> partial update), Edges (what runs next).
- Nodes return PARTIAL updates. Each key is merged into state by its
  *reducer*. No reducer = overwrite. `Annotated[list, operator.add]` = append.
  `add_messages` = append, but de-duplicate/replace by message id.
- Execution model is Pregel / "super-steps": all nodes triggered in the same
  step run in parallel, their updates are applied together at the end of
  the step (that's why reducers matter for parallel branches).
- input_schema / output_schema: restrict what callers send in / get back.
- context_schema + Runtime: run-scoped, read-only dependencies (user id,
  model name, db handle). Not part of state, not checkpointed.
- compile() validates the graph; get_graph().draw_mermaid() visualises it.

Run:  python langgraph_course/01_state_graph_basics.py
"""

import operator
from dataclasses import dataclass
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.runtime import Runtime

from common import ask, banner, print_messages, show


# --- 1. State with reducers --------------------------------------------------

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]  # append / upsert by id
    steps: Annotated[list[str], operator.add]  # append
    topic: str  # overwrite (last write wins)
    summary: str


class InputState(TypedDict):  # what the caller is allowed to pass in
    topic: str


class OutputState(TypedDict):  # what the caller gets back
    summary: str
    steps: list[str]


@dataclass
class Context:  # run-scoped config, injected via Runtime
    audience: str = "a junior engineer"


# --- 2. Nodes: (state, [runtime]) -> partial update --------------------------

def prepare(state: State) -> dict:
    return {
        "messages": [HumanMessage(f"Explain {state['topic']}")],
        "steps": ["prepare"],
    }


def explain(state: State, runtime: Runtime[Context]) -> dict:
    answer = ask(
        f"Explain '{state['topic']}' to {runtime.context.audience} in at most 3 sentences.",
        max_tokens=900,
    )
    return {"summary": answer, "steps": ["explain"]}


def finish(state: State) -> dict:
    return {"steps": ["finish"]}


# --- 3. Wire it up ------------------------------------------------------------

builder = StateGraph(State, context_schema=Context, input_schema=InputState, output_schema=OutputState)
builder.add_node("prepare", prepare)
builder.add_node("explain", explain)
builder.add_node("finish", finish)
builder.add_edge(START, "prepare")
builder.add_edge("prepare", "explain")
builder.add_edge("explain", "finish")
builder.add_edge("finish", END)
graph = builder.compile()


def reducer_demo() -> None:
    """Reducers without any LLM: show how two parallel nodes merge."""

    class S(TypedDict):
        overwrite: str
        appended: Annotated[list[str], operator.add]

    def a(_: S) -> dict:
        return {"appended": ["from a"], "overwrite": "a"}

    def b(_: S) -> dict:
        return {"appended": ["from b"]}  # writing `overwrite` here too would raise InvalidUpdateError

    g = StateGraph(S)
    g.add_node("a", a)
    g.add_node("b", b)
    g.add_edge(START, "a")
    g.add_edge(START, "b")  # a and b run in the same super-step
    g.add_edge(["a", "b"], END)
    show("parallel merge", g.compile().invoke({"overwrite": "init", "appended": []}))

    # add_messages replaces a message that has the same id
    msgs = add_messages([HumanMessage("v1", id="1")], [HumanMessage("v2", id="1"), HumanMessage("new", id="2")])
    print("[add_messages upsert]")
    print_messages(msgs)


if __name__ == "__main__":
    banner("Graph structure (paste into mermaid.live)")
    print(graph.get_graph().draw_mermaid())

    banner("Reducers")
    reducer_demo()

    banner("Invoke with input/output schema + runtime context")
    result = graph.invoke({"topic": "reducers in LangGraph"}, context=Context(audience="a 10 year old"))
    show("output keys", list(result))
    show("steps", result["steps"])
    show("summary", result["summary"], width=800)
