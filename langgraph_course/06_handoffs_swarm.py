"""
06 - Handoffs and the swarm (decentralised) pattern
===================================================

Interview topics
----------------
- Supervisor (lesson 05) = centralised: every turn goes through a manager.
  Swarm / handoffs = decentralised: agents transfer control *directly* to
  each other with "handoff tools" (transfer_to_<agent>). OpenAI Agents SDK
  handoffs and `langgraph-swarm` are this pattern.
- A handoff is just a tool call that the graph turns into
      Command(goto="<agent>", update={...})
  Inside a subgraph you'd use Command(graph=Command.PARENT, goto=...) to jump
  to a sibling in the parent graph.
- `active_agent` is stored in state + a checkpointer, so the next user turn
  resumes with whichever agent was talking last (no re-triage).
- Shared message history vs per-agent private history: shared is simpler
  and gives continuity; private keeps contexts small. Here: shared, but
  stripped of reasoning traces.
- Trade-offs vs supervisor: fewer LLM calls and lower latency (no manager hop),
  but harder to reason about, risk of ping-pong loops (cap the handoffs).

Run:  python langgraph_course/06_handoffs_swarm.py
"""

from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.types import Command

from common import banner, get_llm, print_messages, show

AgentName = Literal["triage", "flights", "hotels"]


# --- Domain tools -------------------------------------------------------------------

BOOKINGS = {"ABC123": {"flight": "LH 401 FRA->JFK", "date": "2026-11-03", "hotel": "Hilton Midtown, 3 nights"}}


@tool
def get_booking(booking_id: str) -> str:
    """Fetch a booking by its 6-character id."""
    b = BOOKINGS.get(booking_id.upper())
    return str(b) if b else "No such booking."


@tool
def change_flight_date(booking_id: str, new_date: str) -> str:
    """Move the flight on a booking to a new date (YYYY-MM-DD)."""
    BOOKINGS[booking_id.upper()]["date"] = new_date
    return f"Flight moved to {new_date}."


@tool
def extend_hotel(booking_id: str, nights: int) -> str:
    """Set the number of hotel nights on a booking."""
    b = BOOKINGS[booking_id.upper()]
    b["hotel"] = b["hotel"].split(",")[0] + f", {nights} nights"
    return f"Hotel now {nights} nights."


# --- Handoff tools: the body never runs; the graph intercepts the call ---------------

def make_handoff_tool(agent: str, description: str):
    @tool(f"transfer_to_{agent}", description=description)
    def handoff() -> str:
        return f"Transferred to {agent}"

    return handoff


to_triage = make_handoff_tool("triage", "Hand the customer back to triage for anything outside your area.")
to_flights = make_handoff_tool("flights", "Hand off to the flights specialist (flight changes, dates).")
to_hotels = make_handoff_tool("hotels", "Hand off to the hotels specialist (rooms, nights).")

AGENTS: dict[str, tuple[str, list]] = {
    "triage": ("You are the triage agent of a travel company. Do not solve requests yourself; transfer to the right specialist.", [to_flights, to_hotels]),
    "flights": ("You are the flights specialist. Use your tools to change flights. For hotel requests, transfer to hotels.", [get_booking, change_flight_date, to_hotels, to_triage]),
    "hotels": ("You are the hotels specialist. Use your tools to change hotel stays. For flight requests, transfer to flights.", [get_booking, extend_hotel, to_flights, to_triage]),
}


class State(MessagesState):
    active_agent: AgentName
    handoffs: int


MAX_HANDOFFS = 4
MAX_TOOL_ROUNDS = 4


def make_agent_node(name: str):
    instructions, tools = AGENTS[name]
    system = SystemMessage(
        f"{instructions}\nYou are '{name}'. Never claim a change was made unless a tool result in this conversation confirms it."
    )
    model = get_llm().bind_tools(tools)
    regular = {t.name: t for t in tools if not t.name.startswith("transfer_to_")}

    def node(state: State) -> Command[AgentName | Literal["__end__"]]:
        new_msgs: list = []
        for _ in range(MAX_TOOL_ROUNDS):
            ai: AIMessage = model.invoke([system, *state["messages"], *new_msgs])
            ai.name = name
            new_msgs.append(ai)
            if not ai.tool_calls:  # final reply to the user -> end this turn
                return Command(goto="__end__", update={"messages": new_msgs, "active_agent": name})

            # Every tool_call must get a ToolMessage back (OpenAI message-format rule),
            # so run the regular tools first, then act on a handoff if there is one.
            handoff = None
            for call in ai.tool_calls:
                if call["name"].startswith("transfer_to_"):
                    handoff = handoff or call
                    continue
                result = regular[call["name"]].invoke(call["args"]) if call["name"] in regular else f"Unknown tool {call['name']}"
                show(f"{name} tool", f"{call['name']}({call['args']}) -> {result}")
                new_msgs.append(ToolMessage(str(result), tool_call_id=call["id"], name=call["name"]))

            for call in ai.tool_calls:
                if not call["name"].startswith("transfer_to_"):
                    continue
                target = call["name"].removeprefix("transfer_to_")
                if call is not handoff or state.get("handoffs", 0) >= MAX_HANDOFFS:
                    new_msgs.append(ToolMessage("Handoff rejected; answer the user yourself.", tool_call_id=call["id"], name=call["name"]))
                    continue
                show("handoff", f"{name} -> {target}")
                # Tell the receiving agent exactly what it now owns, so it acts instead of
                # assuming the previous agent already handled it.
                pending = next(m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage))
                note = f"Transferred to {target}. {target}, the customer's open request is: {pending!r}. Handle it with your tools."
                new_msgs.append(ToolMessage(note, tool_call_id=call["id"], name=call["name"]))
            if handoff and any(m.content.startswith("Transferred") for m in new_msgs[-len(ai.tool_calls):]):
                target = handoff["name"].removeprefix("transfer_to_")
                return Command(goto=target, update={"messages": new_msgs, "active_agent": target, "handoffs": state.get("handoffs", 0) + 1})
        return Command(goto="__end__", update={"messages": new_msgs, "active_agent": name})

    return node


def route_to_active(state: State) -> AgentName:
    """Each new user turn starts with whoever was talking last."""
    return state.get("active_agent", "triage")


builder = StateGraph(State)
for agent_name in AGENTS:
    builder.add_node(agent_name, make_agent_node(agent_name))
builder.add_conditional_edges(START, route_to_active, list(AGENTS))
graph = builder.compile(checkpointer=InMemorySaver())


def user_turn(text: str, config: dict) -> None:
    banner(f"USER: {text}")
    # Reset handoff counter per user turn.
    out = graph.invoke({"messages": [HumanMessage(text)], "handoffs": 0}, config)
    print_messages(out["messages"][-1:])
    show("active_agent after turn", out["active_agent"])


if __name__ == "__main__":    
    config = {"configurable": {"thread_id": "customer-42"}, "recursion_limit": 20}
    user_turn("Hi, booking ABC123: please move my flight to 2026-11-05.", config)
    user_turn("Thanks. Also make the hotel 5 nights instead of 3.", config)
    show("booking now", BOOKINGS["ABC123"])
