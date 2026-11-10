"""
20 - "Deep agents": planning tool, virtual filesystem, sub-agents
=================================================================

Interview topics
----------------
- Simple ReAct loops are "shallow": they lose the plot on long tasks. Systems
  like Claude Code, Deep Research and Manus share four ingredients
  (LangChain packages this as `deepagents.create_deep_agent`):
    1. a planning tool    write_todos: the agent keeps an explicit, updatable
                          todo list in state (a no-op tool that is really
                          context engineering: the plan stays in view)
    2. a file system      ls / read_file / write_file: offload large content
                          out of the message history into state (or disk/store)
    3. sub-agents         task(description): spawn a fresh agent with a clean
                          context; only its final summary returns (context
                          isolation, parallelisable)
    4. a detailed system prompt explaining how to use all of the above
- Mechanics shown here:
    * tools that UPDATE GRAPH STATE by returning Command(update={...}) with a
      ToolMessage (InjectedToolCallId), which ToolNode applies
    * tools that READ state via InjectedState (hidden from the model's schema)
    * a custom reducer for the files dict (merge, not overwrite)
- Long-running agent hygiene: summarise/compact history (lesson 11), cap
  recursion, persist with a checkpointer so work survives restarts.

Run:  python langgraph_course/20_deep_agent.py
"""

from typing import Annotated, Literal

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import InjectedState, ToolNode, tools_condition
from langgraph.types import Command
from typing_extensions import TypedDict

from common import banner, get_llm, make_react_agent, show


# --- State ----------------------------------------------------------------------------

class Todo(TypedDict):
    task: str
    status: Literal["pending", "in_progress", "done"]


def merge_files(old: dict[str, str] | None, new: dict[str, str]) -> dict[str, str]:
    return {**(old or {}), **new}


class DeepState(MessagesState):
    todos: list[Todo]
    files: Annotated[dict[str, str], merge_files]


# --- 1. Planning tool -------------------------------------------------------------------

@tool
def write_todos(todos: list[Todo], tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """Create or replace the todo list. Call at the start of a multi-step task and whenever a step's status changes."""
    show("todos", [f"[{t['status']}] {t['task']}" for t in todos])
    return Command(update={"todos": todos, "messages": [ToolMessage(f"Todo list updated ({len(todos)} items).", tool_call_id=tool_call_id)]})


# --- 2. Virtual file system (lives in graph state, so it is checkpointed) ----------------

@tool
def ls(state: Annotated[dict, InjectedState]) -> str:
    """List files in the workspace."""
    return ", ".join(state.get("files", {})) or "(empty)"


@tool
def read_file(path: str, state: Annotated[dict, InjectedState]) -> str:
    """Read a file from the workspace."""
    return state.get("files", {}).get(path, f"No file {path}")


@tool
def write_file(path: str, content: str, tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """Write (create or overwrite) a file in the workspace."""
    show("write_file", f"{path} ({len(content)} chars)")
    return Command(update={"files": {path: content}, "messages": [ToolMessage(f"Wrote {path}", tool_call_id=tool_call_id)]})


# --- 3. Sub-agent spawning tool (context isolation) --------------------------------------

KB = {
    "postgres": "Postgres checkpointer: durable, supports concurrent threads, production default; needs a managed DB.",
    "sqlite": "SQLite checkpointer: single file, zero setup, great for local dev and single-process apps; poor for concurrent writers.",
    "redis": "Redis checkpointer: very low latency, TTLs for expiring threads; durability depends on persistence config (AOF/RDB).",
}


@tool
def search_kb(query: str) -> str:
    """Search the engineering knowledge base."""
    hits = [v for k, v in KB.items() if k in query.lower()]
    return "\n".join(hits) or f"Nothing found. Topics: {', '.join(KB)}"


researcher = make_react_agent([search_kb], "You are a research sub-agent. Use search_kb, then return a concise summary (max 4 sentences).", name="researcher")


@tool
def task(description: str) -> str:
    """Delegate a self-contained research task to a sub-agent with a fresh context. Returns only its final summary."""
    show("spawn sub-agent", description)
    out = researcher.invoke({"messages": [HumanMessage(description)]}, {"recursion_limit": 10})
    return out["messages"][-1].content


# --- 4. The main agent ---------------------------------------------------------------------

TOOLS = [write_todos, ls, read_file, write_file, task]
SYSTEM = """You are a deep agent that completes multi-step tasks.
Working method:
1. First call write_todos with a short plan (3-5 items).
2. Delegate each research item to a sub-agent with the `task` tool (one topic per call).
3. Save intermediate findings with write_file (e.g. notes/<topic>.md) instead of repeating them in chat.
4. Update todo statuses with write_todos as you progress.
5. Write the final deliverable with write_file, then reply with a 2-sentence summary."""

model = get_llm(max_tokens=2048).bind_tools(TOOLS)


def agent(state: DeepState) -> dict:
    todo_view = "\n".join(f"- [{t['status']}] {t['task']}" for t in state.get("todos", [])) or "(no plan yet)"
    files_view = ", ".join(state.get("files", {})) or "(none)"
    # Re-inject the plan and file list every turn: keeps the agent oriented on long runs.
    system = SystemMessage(f"{SYSTEM}\n\nCurrent todos:\n{todo_view}\nFiles: {files_view}")
    return {"messages": [model.invoke([system, *state["messages"]])]}


builder = StateGraph(DeepState)
builder.add_node("agent", agent)
builder.add_node("tools", ToolNode(TOOLS))
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")
graph = builder.compile()


if __name__ == "__main__":
    banner("Deep agent run")
    request = "Compare the postgres, sqlite and redis checkpointers for LangGraph and write a recommendation to report.md."
    out = graph.invoke({"messages": [HumanMessage(request)]}, {"recursion_limit": 60})
    banner("Result")
    show("final todos", out.get("todos"))
    show("files", list(out.get("files", {})))
    print(out.get("files", {}).get("report.md", "(no report.md written)"))
    print("\n" + out["messages"][-1].content)
