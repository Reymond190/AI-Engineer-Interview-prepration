"""
10 - Human-in-the-loop: interrupt() and Command(resume=...)
===========================================================

Interview topics
----------------
- `interrupt(payload)` pauses the graph *inside a node*, saves state via the
  checkpointer, and surfaces `payload` to the caller (result["__interrupt__"]).
  The process can exit; hours later anyone with the thread_id resumes with
      graph.invoke(Command(resume=value), config)
  and `interrupt()` returns `value` (requires a checkpointer).
- IMPORTANT: on resume the node re-runs FROM THE TOP. Code before interrupt()
  executes again, so keep side effects after the interrupt (or idempotent).
  Multiple interrupts in one node are matched to resume values by order.
- Common patterns:
    approve / reject   a risky tool call (payments, emails, deletes)
    edit               let the human fix the tool arguments before execution
    review & feedback  human feedback goes back to the LLM as a message
    input              ask the human a clarifying question mid-run
- Static breakpoints (compile(interrupt_before=["tools"])) are mostly for
  debugging; interrupt() is the production API.
- Prebuilt: create_agent(..., middleware=[HumanInTheLoopMiddleware(
      interrupt_on={"send_refund": True})]) does the same for agents.

Run:  python langgraph_course/10_human_in_the_loop.py              (scripted decisions)
      python langgraph_course/10_human_in_the_loop.py --interactive (you decide)
"""

import sys
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt

from common import banner, get_llm, print_messages, show

SENSITIVE = {"send_refund"}


@tool
def lookup_order(order_id: str) -> str:
    """Look up an order's amount and status."""
    return f"Order {order_id}: amount $120.00, status delivered, item damaged per customer photo"


@tool
def send_refund(order_id: str, amount: float, reason: str) -> str:
    """Refund money to the customer. Requires human approval."""
    return f"REFUNDED ${amount:.2f} on {order_id} ({reason})"


TOOLS = [lookup_order, send_refund]
model = get_llm().bind_tools(TOOLS)
SYSTEM = SystemMessage("You are a support agent. Look up orders before refunding. Refund only what the order shows.")


def agent(state: MessagesState) -> dict:
    return {"messages": [model.invoke([SYSTEM, *state["messages"]])]}


def route(state: MessagesState) -> Literal["human_review", "tools", "__end__"]:
    last = state["messages"][-1]
    if not last.tool_calls:
        return "__end__"
    return "human_review" if any(c["name"] in SENSITIVE for c in last.tool_calls) else "tools"


def human_review(state: MessagesState) -> Command[Literal["tools", "agent"]]:
    last = state["messages"][-1]
    call = next(c for c in last.tool_calls if c["name"] in SENSITIVE)

    # Everything above this line re-runs on resume, so it must be side-effect free.
    decision = interrupt({"question": "Approve this tool call?", "tool": call["name"], "args": call["args"]})

    if decision["action"] == "approve":
        return Command(goto="tools")
    if decision["action"] == "edit":
        # Replace the AIMessage (same id -> add_messages overwrites it) with edited args.
        edited = last.model_copy(deep=True)
        for c in edited.tool_calls:
            if c["id"] == call["id"]:
                c["args"] = {**c["args"], **decision["args"]}
        return Command(goto="tools", update={"messages": [edited]})
    # reject: answer every tool call with a ToolMessage carrying the human's feedback.
    feedback = [ToolMessage(f"Rejected by human reviewer: {decision.get('feedback', '')}", tool_call_id=c["id"], name=c["name"]) for c in last.tool_calls]
    return Command(goto="agent", update={"messages": feedback})


builder = StateGraph(MessagesState)
builder.add_node("agent", agent)
builder.add_node("tools", ToolNode(TOOLS))
builder.add_node("human_review", human_review)
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", route)
builder.add_edge("tools", "agent")
graph = builder.compile(checkpointer=InMemorySaver())  # interrupts need a checkpointer


def ask_human(payload: dict) -> dict:
    print(f"\n  >>> {payload['question']} {payload['tool']}({payload['args']})")
    action = input("  approve / edit / reject: ").strip().lower() or "approve"
    if action == "edit":
        return {"action": "edit", "args": {"amount": float(input("  new amount: "))}}
    if action == "reject":
        return {"action": "reject", "feedback": input("  feedback for the agent: ")}
    return {"action": "approve"}


def run(thread: str, request: str, scripted: list[dict], interactive: bool) -> None:
    banner(f"thread={thread}: {request}")
    config = {"configurable": {"thread_id": thread}, "recursion_limit": 20}
    result = graph.invoke({"messages": [HumanMessage(request)]}, config)
    # Keep resuming while the graph is paused on an interrupt.
    while result.get("__interrupt__"):
        payload = result["__interrupt__"][0].value
        show("paused, next node", graph.get_state(config).next)
        decision = ask_human(payload) if interactive else scripted.pop(0)
        show("human decision", decision)
        result = graph.invoke(Command(resume=decision), config)
    print_messages(result["messages"][-3:])


if __name__ == "__main__":
    interactive = "--interactive" in sys.argv
    run("t-approve", "Order A-17 arrived damaged, please refund it.", [{"action": "approve"}], interactive)
    run("t-edit", "Order B-22 arrived damaged, please refund it.", [{"action": "edit", "args": {"amount": 60.0}}], interactive)
    run(
        "t-reject",
        "Order C-31 arrived damaged, refund me.",
        [{"action": "reject", "feedback": "Offer a replacement instead of a refund."}, {"action": "reject", "feedback": "No refunds."}],
        interactive,
    )
