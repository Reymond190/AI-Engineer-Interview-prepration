"""
12 - Streaming: values, updates, messages, custom, debug
========================================================

Interview topics
----------------
- Agents are slow; streaming is how you make them feel fast and observable.
  graph.stream(input, config, stream_mode=...):
    "values"   full state after every super-step
    "updates"  only {node: delta} after each node (cheapest, most common)
    "messages" LLM tokens as they are generated: (AIMessageChunk, metadata),
               metadata["langgraph_node"] says which node produced the token
    "custom"   anything a node emits via get_stream_writer() / runtime.stream_writer
               (progress bars, tool progress, retrieved doc ids)
    "debug"    / "tasks" / "checkpoints": detailed execution events
  Pass a list for several modes -> yields (mode, chunk) tuples.
- Token streaming works even when the node calls llm.invoke() (not .stream()):
  LangGraph hooks the model's callbacks. Filter by node or by tags so only the
  user-facing node's tokens reach the UI.
- subgraphs=True adds a namespace to every event (lesson 07).
- Async: `async for ... in graph.astream(...)`; for event-level detail,
  graph.astream_events(version="v2").
- Production: LangGraph Server streams these same modes over SSE to clients.

Run:  python langgraph_course/12_streaming.py
"""

import time
from typing import TypedDict

from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from common import banner, get_llm

drafter = get_llm().with_config(tags=["draft"])  # tags let you filter token streams
polisher = get_llm().with_config(tags=["final"])


class State(TypedDict, total=False):
    topic: str
    sources: list[str]
    draft: str
    final: str


def retrieve(state: State) -> dict:
    writer = get_stream_writer()
    sources = []
    for i, src in enumerate(["docs/langgraph.md", "blog/agents.md", "notes/streaming.txt"], 1):
        time.sleep(0.3)  # pretend I/O
        sources.append(src)
        writer({"progress": f"retrieved {i}/3", "source": src})  # -> "custom" stream
    return {"sources": sources}


def draft(state: State) -> dict:
    return {"draft": drafter.invoke(f"Two sentences on {state['topic']}.").content}


def polish(state: State) -> dict:
    return {"final": polisher.invoke(f"Rewrite more vividly, same length:\n{state['draft']}").content}


builder = StateGraph(State)
builder.add_node("retrieve", retrieve)
builder.add_node("draft", draft)
builder.add_node("polish", polish)
builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "draft")
builder.add_edge("draft", "polish")
builder.add_edge("polish", END)
graph = builder.compile()

INPUT = {"topic": "why streaming matters for agent UX"}


if __name__ == "__main__":
    banner('stream_mode="updates"')
    for chunk in graph.stream(INPUT, stream_mode="updates"):
        for node, delta in chunk.items():
            print(f"  {node}: keys={list(delta)}")

    banner('stream_mode="values" (full state each step)')
    for state in graph.stream(INPUT, stream_mode="values"):
        print(f"  keys so far: {list(state)}")

    banner('multiple modes: ["custom", "messages"], tokens from the "polish" node only')
    for mode, chunk in graph.stream(INPUT, stream_mode=["custom", "messages"]):
        if mode == "custom":
            print(f"  [custom] {chunk}")
        else:
            token, meta = chunk
            if meta["langgraph_node"] == "polish" and token.content:
                print(token.content, end="", flush=True)
    print()

    banner("filtering tokens by tag instead of node")
    for token, meta in graph.stream(INPUT, stream_mode="messages"):
        if "draft" in meta.get("tags", []) and token.content:
            print(token.content, end="", flush=True)
    print()
