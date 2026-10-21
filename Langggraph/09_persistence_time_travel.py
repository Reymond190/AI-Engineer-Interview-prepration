"""
09 - Persistence, threads, and time travel
==========================================

Interview topics
----------------
- A checkpointer saves a snapshot of state after every super-step, keyed by
  `thread_id` (a conversation/session) and `checkpoint_id`. That one feature
  enables: multi-turn memory, human-in-the-loop (pause/resume), fault
  tolerance (resume after a crash), and time travel (replay / fork).
- Checkpointers: InMemorySaver (tests), SqliteSaver (local), PostgresSaver /
  Redis (production). The LangGraph Platform/Server provides one for you.
- Durability modes (invoke(..., durability=...)):
    "exit"  persist only when the run finishes/pauses (fastest, least safe)
    "async" persist in the background while the next step runs (default)
    "sync"  persist before starting the next step (safest)
- State API:
    graph.get_state(config)            latest snapshot (.values, .next, .tasks)
    graph.get_state_history(config)    all snapshots, newest first
    graph.update_state(config, values, as_node=...)  write as if a node ran
- Time travel:
    replay : invoke(None, {thread_id, checkpoint_id}) re-runs from that point
             (steps after it are re-executed, so LLM calls happen again)
    fork   : update_state() on an old checkpoint creates a new branch with
             edited state, then invoke(None, new_config) continues from there.
- Use cases: debugging ("what if the planner had said X?"), letting a user
  edit an earlier step, A/B testing prompts against real traces.

Run:  python langgraph_course/09_persistence_time_travel.py
"""

from pathlib import Path
from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, MessagesState, StateGraph

from common import ask, banner, get_llm, print_messages, show


# --- Part 1: multi-turn chat memory via thread_id ---------------------------------

llm = get_llm()


def chat(state: MessagesState) -> dict:
    return {"messages": [llm.invoke(state["messages"])]}


chat_builder = StateGraph(MessagesState)
chat_builder.add_node("chat", chat)
chat_builder.add_edge(START, "chat")


# --- Part 2: a 3-step pipeline to time-travel through ------------------------------

class Story(TypedDict, total=False):
    genre: str
    premise: str
    title: str


def premise(state: Story) -> dict:
    return {"premise": ask(f"One-sentence premise for a {state['genre']} short story.")}


def title(state: Story) -> dict:
    return {"title": ask(f"Give one short title (just the title) for: {state['premise']}")}


story_builder = StateGraph(Story)
story_builder.add_node("premise", premise)
story_builder.add_node("title", title)
story_builder.add_edge(START, "premise")
story_builder.add_edge("premise", "title")
story_builder.add_edge("title", END)


if __name__ == "__main__":
    banner("1. Threads: same graph, separate memories (SQLite checkpointer)")
    db = Path(__file__).with_name("checkpoints.sqlite")
    with SqliteSaver.from_conn_string(str(db)) as saver:
        graph = chat_builder.compile(checkpointer=saver)
        alice = {"configurable": {"thread_id": "alice"}}
        bob = {"configurable": {"thread_id": "bob"}}
        graph.invoke({"messages": [("user", "Hi, I'm Alice and my favourite language is Rust.")]}, alice)
        graph.invoke({"messages": [("user", "Hi, I'm Bob.")]}, bob)
        out = graph.invoke({"messages": [("user", "What's my name and favourite language?")]}, alice, durability="sync")
        print_messages(out["messages"][-1:])
        out = graph.invoke({"messages": [("user", "What's my favourite language?")]}, bob)
        print_messages(out["messages"][-1:])
        show("persisted to", str(db))

    banner("2. Inspect checkpoints")
    graph = story_builder.compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "story"}}
    graph.invoke({"genre": "sci-fi"}, config)
    history = list(graph.get_state_history(config))  # newest first
    for snap in history:
        show(f"step {snap.metadata['step']} next={snap.next}", {k: str(v)[:60] for k, v in snap.values.items()})

    banner("3. Replay from the checkpoint before 'title' (re-runs title only)")
    before_title = next(s for s in history if s.next == ("title",))
    replayed = graph.invoke(None, before_title.config)
    show("same premise", replayed["premise"][:100])
    show("new title", replayed["title"])

    banner("4. Fork: edit the premise at that checkpoint, continue from there")
    fork_config = graph.update_state(before_title.config, {"premise": "A lighthouse keeper on Europa discovers the ice is listening."})
    forked = graph.invoke(None, fork_config)
    show("forked premise", forked["premise"])
    show("forked title", forked["title"])
    show("history length now", len(list(graph.get_state_history(config))))
