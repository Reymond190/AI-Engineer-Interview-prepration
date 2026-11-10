"""
02 - Routing: conditional edges vs Command
==========================================

Interview topics
----------------
- Router pattern: an LLM classifies the input, the graph branches on it.
  The LLM decides *which path*, your code decides *what the paths are*.
  (Anthropic's "Building effective agents" calls this a *workflow*, not an
  agent: control flow is still predefined.)
- Two ways to branch:
    a) add_conditional_edges(source, fn, path_map)  - routing logic lives
       in a separate function; the node itself only updates state.
    b) return Command(update=..., goto="node")       - the node updates
       state AND routes in one step. Annotate the return type with
       Command[Literal[...]] so the graph can be drawn/validated.
- Use (a) for pure routing, (b) when routing depends on work done inside
  the node (very common in multi-agent handoffs; see lesson 06).
- Always constrain the router output (Literal / enum) and have a fallback.

Run:  python langgraph_course/02_routing_conditional_edges.py
"""

from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from pydantic import BaseModel, Field

from common import ask, banner, show, structured

Route = Literal["billing", "technical", "general"]


class State(TypedDict, total=False):
    question: str
    route: Route
    answer: str


class RouteDecision(BaseModel):
    route: Route = Field(description="billing = payments/charges/refunds, technical = bugs/errors/how-to, general = anything else")
    reason: str


# Structured output = constrained JSON decoding on vLLM; returns a RouteDecision.
router = structured(RouteDecision)


def classify(state: State) -> dict:
    decision = router.invoke(f"Classify this customer message:\n{state['question']}")
    show("router", decision.model_dump())
    return {"route": decision.route}


def make_specialist(persona: str):
    def node(state: State) -> dict:
        return {"answer": ask(f"You are a {persona}. Answer in 2 sentences:\n{state['question']}")}

    return node


# --- (a) conditional edges -------------------------------------------------------

def pick_route(state: State) -> Route:
    return state.get("route", "general")


def build_with_conditional_edges():
    g = StateGraph(State)
    g.add_node("classify", classify)
    g.add_node("billing", make_specialist("billing support agent"))
    g.add_node("technical", make_specialist("senior support engineer"))
    g.add_node("general", make_specialist("friendly concierge"))
    g.add_edge(START, "classify")
    # path_map is optional when the function returns node names directly,
    # but providing it lets get_graph() draw the possible branches.
    g.add_conditional_edges("classify", pick_route, {"billing": "billing", "technical": "technical", "general": "general"})
    for n in ("billing", "technical", "general"):
        g.add_edge(n, END)
    return g.compile()


# --- (b) Command: update + goto from inside the node -----------------------------

def classify_and_route(state: State) -> Command[Literal["billing", "technical", "general"]]:
    decision = router.invoke(f"Classify this customer message:\n{state['question']}")
    show("router", decision.model_dump())
    return Command(update={"route": decision.route}, goto=decision.route)


def build_with_command():
    g = StateGraph(State)
    g.add_node("classify", classify_and_route)  # no outgoing edges needed
    g.add_node("billing", make_specialist("billing support agent"))
    g.add_node("technical", make_specialist("senior support engineer"))
    g.add_node("general", make_specialist("friendly concierge"))
    g.add_edge(START, "classify")
    return g.compile()


if __name__ == "__main__":
    banner("(a) add_conditional_edges")
    out = build_with_conditional_edges().invoke({"question": "I was charged twice for my subscription this month."})
    show("answer", out["answer"])

    banner("(b) Command(goto=...)")
    graph = build_with_command()
    print(graph.get_graph().draw_mermaid())
    out = graph.invoke({"question": "The app crashes with error 0x80070005 when I click export."})
    show("answer", out["answer"])
