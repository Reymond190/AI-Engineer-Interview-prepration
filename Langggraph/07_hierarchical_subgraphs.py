"""
07 - Subgraphs and hierarchical teams
=====================================

Interview topics
----------------
- A compiled graph is itself a node ("graphs all the way down"). That's how
  you build hierarchical multi-agent systems: a top-level supervisor whose
  workers are entire teams (each a subgraph with its own supervisor/workers).
- Two ways to embed a subgraph:
    1. add_node("team", compiled_subgraph): parent and child SHARE state keys;
       overlapping keys flow in and out automatically.
    2. Call subgraph.invoke(...) inside a node: child has a PRIVATE schema; you
       map parent state -> child input and child output -> parent update.
       Use this to isolate context (the child never sees the parent's messages).
- Persistence: pass a checkpointer only to the parent; subgraphs inherit it.
  `graph.get_state(config, subgraphs=True)` shows nested state, and an
  interrupt() inside a subgraph pauses the whole tree.
- Streaming with subgraphs=True yields (namespace, chunk) so you can see which
  team/node produced an event. Essential for UIs on top of nested agents.
- Command(graph=Command.PARENT, goto=...) lets a node inside a subgraph jump
  to a node in the parent (cross-team handoff).

Run:  python langgraph_course/07_hierarchical_subgraphs.py
"""

import operator
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from common import ask, banner, show


# --- Research team: PRIVATE schema, called from a wrapper node --------------------

class ResearchState(TypedDict, total=False):
    question: str
    notes: Annotated[list[str], operator.add]
    brief: str


def researcher(angle: str):
    def node(state: ResearchState) -> dict:
        note = ask(f"In 2 sentences, give the {angle} of: {state['question']}")
        return {"notes": [f"[{angle}] {note}"]}

    return node


def research_lead(state: ResearchState) -> dict:
    brief = ask("Merge these notes into a 3-bullet research brief:\n" + "\n".join(state["notes"]))
    return {"brief": brief}


research_builder = StateGraph(ResearchState)
research_builder.add_node("pros", researcher("main advantages"))
research_builder.add_node("cons", researcher("main risks or drawbacks"))
research_builder.add_node("lead", research_lead)
research_builder.add_edge(START, "pros")
research_builder.add_edge(START, "cons")  # pros + cons run in parallel
research_builder.add_edge(["pros", "cons"], "lead")  # join: wait for both
research_builder.add_edge("lead", END)
research_team = research_builder.compile(name="research_team")


# --- Writing team: SHARES keys with the parent, added directly as a node ----------

class WritingState(TypedDict, total=False):
    brief: str  # shared with parent (input)
    draft: str  # private to the team
    article: str  # shared with parent (output)


def writer(state: WritingState) -> dict:
    return {"draft": ask(f"Write a 4-sentence executive summary from this brief:\n{state['brief']}")}


def editor(state: WritingState) -> dict:
    return {"article": ask(f"Edit for clarity and cut any filler. Return only the edited text:\n{state['draft']}")}


writing_builder = StateGraph(WritingState)
writing_builder.add_node("writer", writer)
writing_builder.add_node("editor", editor)
writing_builder.add_edge(START, "writer")
writing_builder.add_edge("writer", "editor")
writing_builder.add_edge("editor", END)
writing_team = writing_builder.compile(name="writing_team")


# --- Top level ---------------------------------------------------------------------

class ParentState(TypedDict, total=False):
    topic: str
    brief: str
    article: str


def run_research(state: ParentState) -> dict:
    # Explicit mapping in and out: the team's notes never leak into parent state.
    out = research_team.invoke({"question": state["topic"]})
    return {"brief": out["brief"]}


def check_brief(state: ParentState) -> Command:
    # A top-level "supervisor" decision in code: only write if research produced something.
    if len(state.get("brief", "")) < 40:
        return Command(goto=END)
    return Command(goto="writing_team")


parent = StateGraph(ParentState)
parent.add_node("research_team", run_research)
parent.add_node("check_brief", check_brief, destinations=("writing_team", END))
parent.add_node("writing_team", writing_team)  # compiled graph as a node
parent.add_edge(START, "research_team")
parent.add_edge("research_team", "check_brief")
parent.add_edge("writing_team", END)
graph = parent.compile(checkpointer=InMemorySaver())


if __name__ == "__main__":
    banner("Graph (xray=1 expands the subgraph that was added as a node)")
    print(graph.get_graph(xray=1).draw_mermaid())

    banner("Streaming with subgraphs=True -> (namespace, update)")
    config = {"configurable": {"thread_id": "report-1"}}
    for namespace, update in graph.stream({"topic": "adopting multi-agent LLM systems in a bank"}, config, stream_mode="updates", subgraphs=True):
        where = " > ".join(namespace) or "parent"
        for node in update:
            print(f"  [{where}] node={node}")

    state = graph.get_state(config)
    show("final article", state.values.get("article", "(skipped)"), width=1000)
    show("parent keys (no 'draft', no 'notes': team-private)", list(state.values))
