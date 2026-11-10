"""
03 - Tool calling and the ReAct agent loop
==========================================

Interview topics
----------------
- ReAct = Reason + Act: the model alternates "think / call a tool / read the
  observation" until it can answer. In LangGraph this is a 2-node cycle:

        START -> agent --(has tool_calls?)--> tools -> agent ... -> END

- `@tool` turns a typed, docstring'd function into a schema the model sees.
  Good tool design (clear names, few params, actionable error strings) matters
  more than prompt tweaks.
- `ToolNode` executes every tool_call on the last AIMessage (in parallel)
  and returns ToolMessages; `tools_condition` routes to "tools" or END.
- Native vs emulated tool calling:
    native   : llm.bind_tools(tools) -> the provider returns structured tool_calls
               (vLLM: --enable-auto-tool-choice --tool-call-parser hermes). Used here.
    emulated : describe tools in the prompt, parse the text (lesson 03b), for
               models without function-calling training.
  Either way the graph only ever sees AIMessage.tool_calls, so it's portable.
- Parallel tool calls: one AIMessage can request several tools; ToolNode runs
  them concurrently.
- Prebuilt shortcut: `langchain.agents.create_agent(model, tools)` (LangChain 1.x;
  replaces the old `langgraph.prebuilt.create_react_agent`) with *middleware*
  hooks (before/after model, wrap tool call) for summarisation, HITL, guardrails.
  Know that it exists, but interviewers want you to be able to build the loop yourself.
- Guard rails: recursion_limit, max tool rounds, tool error handling
  (ToolNode(handle_tool_errors=True) turns exceptions into ToolMessages the
  model can recover from).

Run:  python langgraph_course/03_tool_calling_react.py
"""

import math

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from common import banner, get_llm, print_messages

INVENTORY = {"widget": {"price": 4.25, "qty": 120}, "gadget": {"price": 19.99, "qty": 35}, "gizmo": {"price": 7.5, "qty": 0}}


@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression, e.g. '19.99 * 35'. Supports + - * / ** and math functions like sqrt()."""
    allowed = {k: getattr(math, k) for k in ("sqrt", "log", "exp", "ceil", "floor")}
    return str(eval(expression, {"__builtins__": {}}, allowed))  # demo only, never eval untrusted input in prod


@tool
def lookup_product(name: str) -> str:
    """Look up price (USD) and quantity in stock for a product by name, e.g. 'widget'."""
    # Be forgiving about input ("Gadgets " -> "gadget"): cheap and saves a model round-trip.
    item = INVENTORY.get(name.lower().strip().removesuffix("s"))
    if not item:
        # Actionable error: tell the model what IS valid.
        return f"Unknown product '{name}'. Known products: {', '.join(INVENTORY)}"
    return f"{name}: price=${item['price']}, qty={item['qty']}"


TOOLS = [calculator, lookup_product]
SYSTEM = (
    "You are an inventory assistant. Use tools for every number; do not compute in your head. "
    "When the user mentions a product, look it up with the tool instead of asking them for details."
)
model = get_llm().bind_tools(TOOLS)  # tool JSON schemas are sent with every request


def agent(state: MessagesState) -> dict:
    # The system prompt is prepended at call time rather than stored in state.
    return {"messages": [model.invoke([SystemMessage(SYSTEM), *state["messages"]])]}


builder = StateGraph(MessagesState)
builder.add_node("agent", agent)
builder.add_node("tools", ToolNode(TOOLS, handle_tool_errors=True))
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)  # -> "tools" or END
builder.add_edge("tools", "agent")
graph = builder.compile()


if __name__ == "__main__":
    banner("ReAct agent: multi-step tool use")
    question = "What is the total stock value of gadgets (price * qty)?"
    # stream_mode="updates" yields {node_name: update} after each step,
    # so you can watch the loop happen.
    for step in graph.stream({"messages": [HumanMessage(question)]}, {"recursion_limit": 12}, stream_mode="updates"):
        for node, update in step.items():
            print(f"--- {node}")
            print_messages(update["messages"])

    banner("Parallel tool calls in one turn")
    out = graph.invoke({"messages": [HumanMessage("Look up widget, gadget and gizmo, then tell me which are out of stock.")]}, {"recursion_limit": 12})
    print_messages(out["messages"])

    banner("Tool error recovery (unknown product)")
    out = graph.invoke({"messages": [HumanMessage("How many sprockets do we have in stock?")]}, {"recursion_limit": 12})
    print_messages(out["messages"])

    banner("Same agent with the prebuilt: langchain.agents.create_agent")
    from langchain.agents import create_agent

    prebuilt = create_agent(get_llm(), TOOLS, system_prompt=SYSTEM)
    out = prebuilt.invoke({"messages": [HumanMessage("What is the stock value of widgets?")]})
    print_messages(out["messages"][-1:])
