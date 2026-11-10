"""
03b - Tool calling with a model that has NO native function calling
===================================================================

Interview topics
----------------
- Not every model (or serving stack) supports function calling. Common in
  on-prem / open-weight deployments: reasoning distills (DeepSeek-R1-Distill),
  base models, vLLM started without --tool-call-parser.
- Technique: describe tools in the prompt, ask for a strict text format,
  parse it into AIMessage(tool_calls=[...]). The rest of the graph (ToolNode,
  tools_condition) is unchanged. That's the value of LangGraph's message
  abstraction.
- What goes wrong in practice (all observed with R1-Distill-14B while building
  this course) and the fix for each:
    * drops the tags / emits fenced JSON  -> lenient parser, known tool names only
    * narrates a multi-step plan with several calls -> take only the FIRST call
    * describes a tool but never calls it -> one-shot "nudge" re-prompt
    * claims an action succeeded with no tool result -> instruct + verify state
    * malformed JSON for structured output -> validate and retry with the error
    * the SERVER ate the reply: with Qwen3.5, vLLM's qwen3_coder tool parser
      claimed our <tool_call> tags, failed to parse them and returned empty
      content -> use a tag no server-side parser owns (<action>)
- Interview answer to "the model doesn't support tools, what do you do?":
  1) prefer a model/server with native support, 2) constrained decoding
  (grammar / JSON schema), 3) prompt protocol + robust parser + retries.

This lesson runs Qwen3 WITHOUT bind_tools, so the model only sees the text protocol.

Run:  python langgraph_course/03b_tool_calling_without_native_support.py
"""

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel

from common import banner, get_llm, print_messages, show
from text_tool_adapter import TextToolCallingModel, ask_json, parse_tool_calls


@tool
def get_weather(city: str) -> str:
    """Current weather for a city."""
    return {"paris": "14C, rain", "cairo": "31C, sunny"}.get(city.lower(), "unknown city")


@tool
def convert_c_to_f(celsius: float) -> str:
    """Convert Celsius to Fahrenheit."""
    return f"{celsius * 9 / 5 + 32:.1f}F"


TOOLS = [get_weather, convert_c_to_f]
model = TextToolCallingModel(get_llm(), TOOLS, "You are a weather assistant.")


def agent(state: MessagesState) -> dict:
    return {"messages": [model.invoke(state["messages"])]}


builder = StateGraph(MessagesState)
builder.add_node("agent", agent)
builder.add_node("tools", ToolNode(TOOLS, handle_tool_errors=True))
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")
graph = builder.compile()


class Weather(BaseModel):
    city: str
    temp_c: float
    raining: bool


if __name__ == "__main__":
    banner("Parser: messy outputs real models produce")
    names = {t.name for t in TOOLS}
    samples = [
        '<action>{"name": "get_weather", "args": {"city": "Paris"}}</action>',
        'Sure!\n```json\n{\n  "name": "get_weather",\n  "args": {"city": "Cairo"}\n}\n```\nThen I will convert it: {"name": "convert_c_to_f", "args": {"celsius": 31}}',
        'The answer is {"temp": 14}',  # JSON, but not a tool call
    ]
    for s in samples:
        ai = parse_tool_calls(s, names)
        show("parsed", {"content": ai.content, "tool_calls": [(c["name"], c["args"]) for c in ai.tool_calls]})

    banner("Emulated tool-calling agent")
    out = graph.invoke({"messages": [HumanMessage("What's the weather in Cairo, in Fahrenheit?")]}, {"recursion_limit": 12})
    print_messages(out["messages"])

    banner("Structured output by prompt + validate + retry")
    show("weather", ask_json("Paris is 14 degrees C with light rain today.", Weather))
