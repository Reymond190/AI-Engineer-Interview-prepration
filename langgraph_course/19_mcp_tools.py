"""
19 - MCP: plugging external tool servers into a LangGraph agent
===============================================================

Interview topics
----------------
- Model Context Protocol (MCP, Anthropic, Nov 2024; now the industry
  standard, governed by the Linux Foundation's Agentic AI Foundation):
  a JSON-RPC protocol so any agent can use any tool server. "USB-C for tools."
  M agents x N tools becomes M + N integrations.
- Server primitives:
    tools      actions the model can call (model-controlled)
    resources  read-only context addressed by URI (app-controlled)
    prompts    reusable prompt templates (user-controlled)
- Transports: stdio (local subprocess) and Streamable HTTP (remote; replaced
  SSE). Remote servers use OAuth 2.1 for auth.
- langchain-mcp-adapters: MultiServerMCPClient(...).get_tools() converts MCP
  tools into LangChain tools, so ToolNode / bind_tools just work (async-only).
- Security talking points: tool poisoning / prompt injection through tool
  descriptions and results, over-broad permissions, confused deputy. Mitigate
  with allow-lists, human approval for writes (lesson 10), least privilege.
- MCP vs A2A: MCP connects an agent to tools/data; A2A (Agent2Agent, Google ->
  Linux Foundation) connects agents to other agents (agent cards, tasks).

Run:  python langgraph_course/19_mcp_tools.py
"""

import asyncio
import sys
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from common import banner, get_llm, print_messages, show

SERVER = Path(__file__).with_name("mcp_servers") / "tickets_server.py"


async def main() -> None:
    client = MultiServerMCPClient(
        {
            "tickets": {"transport": "stdio", "command": sys.executable, "args": [str(SERVER)]},
            # A remote server would look like:
            # "github": {"transport": "streamable_http", "url": "https://.../mcp", "headers": {"Authorization": "Bearer ..."}},
        }
    )

    banner("Discover what the server offers")
    tools = await client.get_tools()
    show("tools", [t.name for t in tools])
    resources = await client.get_resources("tickets", uris=["tickets://runbook"])
    runbook = resources[0].as_string()
    show("resource tickets://runbook", runbook)
    prompt = await client.get_prompt("tickets", "triage_prompt", arguments={"team": "platform"})
    show("prompt", prompt[0].content)

    banner("Agent using MCP tools (async graph)")
    model = get_llm().bind_tools(tools)
    system = SystemMessage(f"{prompt[0].content}\nRunbook: {runbook}")

    async def agent(state: MessagesState) -> dict:
        return {"messages": [await model.ainvoke([system, *state["messages"]])]}

    g = StateGraph(MessagesState)
    g.add_node("agent", agent)
    g.add_node("tools", ToolNode(tools))
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", tools_condition)
    g.add_edge("tools", "agent")
    graph = g.compile()

    out = await graph.ainvoke({"messages": [HumanMessage("Check open tickets and fix any that break the runbook.")]}, {"recursion_limit": 15})
    print_messages(out["messages"][1:])


if __name__ == "__main__":
    asyncio.run(main())
