"""
11 - Memory: short-term (thread) and long-term (cross-thread store)
===================================================================

Interview topics
----------------
- Short-term memory = the message history of ONE thread, persisted by the
  checkpointer. Problem: it grows without bound and the context window does not.
  Strategies:
    * trim     keep the last N tokens/messages (trim_messages), cheap, lossy
    * summarise  fold old turns into a running summary, delete them from state
               with RemoveMessage (add_messages treats it as "delete by id")
    * filter   drop tool chatter / reasoning before sending to the model
  Trimming at *call time* leaves state intact; deleting changes state for good.
- Long-term memory = facts that outlive a thread (user preferences, profile).
  Lives in a Store (InMemoryStore / PostgresStore), namespaced like
  ("users", user_id, "memories"), accessed in nodes via runtime.store.
  Optional semantic search: give the store an embedding index.
- Memory types (CoALA paper / LangMem terms):
    semantic   facts ("prefers Python", "works at a bank")
    episodic   past experiences / successful trajectories (few-shot examples)
    procedural how to behave: the system prompt itself, updated from feedback
- When to write: in the hot path (agent saves memories during the conversation,
  adds latency) or in the background (after the conversation, more thorough).
- Context engineering = choosing what goes into the window: write, select,
  compress, isolate (Anthropic / LangChain terminology).

Run:  python langgraph_course/11_memory_short_and_long_term.py
"""

import hashlib
import math
import uuid
from dataclasses import dataclass

from langchain_core.messages import HumanMessage, RemoveMessage, SystemMessage, trim_messages
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.runtime import Runtime
from langgraph.store.memory import InMemoryStore
from pydantic import BaseModel, Field

from common import ask, banner, get_llm, print_messages, show, structured

llm = get_llm()


# =============================================================================
# Part 1: short-term memory - trimming + running summary
# =============================================================================

class ChatState(MessagesState):
    summary: str


KEEP_LAST = 4  # messages kept verbatim after summarising


def chat(state: ChatState) -> dict:
    system = "You are a concise assistant. Reply in at most 2 sentences."
    if state.get("summary"):
        system += f"\nSummary of the earlier conversation: {state['summary']}"
    # Trim at call time as a safety net: the state keeps everything, the model sees <= 1000 tokens.
    window = trim_messages(state["messages"], max_tokens=1000, token_counter=count_tokens_approximately, strategy="last", start_on="human")
    return {"messages": [llm.invoke([SystemMessage(system), *window])]}


def should_summarise(state: ChatState) -> str:
    return "summarise" if len(state["messages"]) > KEEP_LAST + 2 else END


def summarise(state: ChatState) -> dict:
    old = state["messages"][:-KEEP_LAST]
    transcript = "\n".join(f"{m.type}: {m.content}" for m in old)
    summary = ask(
        f"<existing_summary>{state.get('summary', '(none)')}</existing_summary>\n"
        f"<new_turns>\n{transcript}\n</new_turns>",
        system="You compress conversations. Output ONLY an updated summary (max 3 sentences) of the facts the USER "
        "stated about themselves and their goals. Do not answer or continue the conversation.",
    )
    # RemoveMessage(id) deletes from state via the add_messages reducer.
    return {"summary": summary, "messages": [RemoveMessage(id=m.id) for m in old]}


short_builder = StateGraph(ChatState)
short_builder.add_node("chat", chat)
short_builder.add_node("summarise", summarise)
short_builder.add_edge(START, "chat")
short_builder.add_conditional_edges("chat", should_summarise)
short_builder.add_edge("summarise", END)
short_graph = short_builder.compile(checkpointer=InMemorySaver())


# =============================================================================
# Part 2: long-term memory - Store with namespaces and semantic search
# =============================================================================

def toy_embed(texts: list[str]) -> list[list[float]]:
    """Hashing bag-of-words embedding so the demo needs no embedding model.
    In production: OpenAIEmbeddings, a local bge/nomic model served by vLLM, etc."""
    out = []
    for t in texts:
        v = [0.0] * 256
        for w in t.lower().split():
            v[int(hashlib.md5(w.strip(".,!?").encode()).hexdigest(), 16) % 256] += 1
        n = math.sqrt(sum(x * x for x in v)) or 1
        out.append([x / n for x in v])
    return out


store = InMemoryStore(index={"dims": 256, "embed": toy_embed})


@dataclass
class UserCtx:
    user_id: str


class Memories(BaseModel):
    facts: list[str] = Field(description="Durable facts about the user worth remembering (preferences, job, projects). Empty if none.")


extractor = structured(Memories)


def assistant(state: MessagesState, runtime: Runtime[UserCtx]) -> dict:
    ns = ("users", runtime.context.user_id, "memories")
    query = state["messages"][-1].content
    # Semantic search over this user's memories only.
    hits = runtime.store.search(ns, query=query, limit=3)
    known = "\n".join(f"- {h.value['fact']}" for h in hits) or "(nothing yet)"
    system = f"You are a personal assistant. What you know about the user:\n{known}"
    return {"messages": [llm.invoke([SystemMessage(system), *state["messages"]])]}


def remember(state: MessagesState, runtime: Runtime[UserCtx]) -> dict:
    """Hot-path memory writing: extract facts from the latest user message."""
    ns = ("users", runtime.context.user_id, "memories")
    user_msg = next(m for m in reversed(state["messages"]) if isinstance(m, HumanMessage))
    for fact in extractor.invoke(f"Extract durable facts about the user from: {user_msg.content!r}").facts:
        runtime.store.put(ns, str(uuid.uuid4()), {"fact": fact})
        show("stored memory", fact)
    return {}


long_builder = StateGraph(MessagesState, context_schema=UserCtx)
long_builder.add_node("assistant", assistant)
long_builder.add_node("remember", remember)
long_builder.add_edge(START, "assistant")
long_builder.add_edge("assistant", "remember")
long_builder.add_edge("remember", END)
long_graph = long_builder.compile(checkpointer=InMemorySaver(), store=store)


if __name__ == "__main__":
    banner("Short-term: running summary + RemoveMessage")
    cfg = {"configurable": {"thread_id": "s1"}}
    for turn in [
        "I'm planning a trip to Japan in April.",
        "I like hiking and ramen.",
        "My budget is around $3000.",
        "Should I get a JR pass?",
        "Remind me: what month, hobbies and budget did I mention?",
    ]:
        out = short_graph.invoke({"messages": [HumanMessage(turn)]}, cfg)
    state = short_graph.get_state(cfg).values
    show("messages kept in state", len(state["messages"]))
    show("summary", state.get("summary", ""))
    print_messages(state["messages"][-1:])

    banner("Long-term: memories survive across threads (same user)")
    ctx = UserCtx(user_id="u-42")
    long_graph.invoke({"messages": [HumanMessage("I'm a data engineer and I mostly write Python and SQL.")]}, {"configurable": {"thread_id": "monday"}}, context=ctx)
    out = long_graph.invoke({"messages": [HumanMessage("Which language should I use for a quick ETL script?")]}, {"configurable": {"thread_id": "friday"}}, context=ctx)
    print_messages(out["messages"][-2:-1] + out["messages"][-1:])

    banner("Another user sees none of it (namespace isolation)")
    out = long_graph.invoke({"messages": [HumanMessage("What do you know about me?")]}, {"configurable": {"thread_id": "x"}}, context=UserCtx("u-99"))
    print_messages(out["messages"][-1:])
    show("store namespaces", store.list_namespaces())
