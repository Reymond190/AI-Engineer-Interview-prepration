"""
16 - Reliability: retries, caching, error handlers, timeouts, limits
====================================================================

Interview topics
----------------
- Production agents fail constantly: rate limits, flaky tools, malformed
  model output, slow APIs, loops. LangGraph node-level policies:
    retry_policy=RetryPolicy(max_attempts, backoff, retry_on=...)
    cache_policy=CachePolicy(ttl=...) + compile(cache=InMemoryCache())
        (identical input -> cached output, no LLM call; key_func customises the key)
    error_handler=fn(state, error: NodeError) -> Command   (LangGraph 1.2+)
        graceful degradation: route to a fallback instead of crashing the run
    timeout=TimeoutPolicy(run_timeout=..., idle_timeout=...)  (LangGraph 1.2+)
    builder.set_node_defaults(...) applies policies to every node
- Run-level guards: config["recursion_limit"] (GraphRecursionError), explicit
  iteration counters in state, max_concurrency.
- Model-level: llm.with_retry(), llm.with_fallbacks([backup_llm]), request
  timeouts, structured output validation (lesson 04).
- Durability (lesson 09): with a checkpointer, a crashed run resumes from the
  last super-step; finished parallel branches are not re-run.
- Also expected in interviews: idempotent tools, tracing/observability
  (LangSmith / OpenTelemetry), guardrails on inputs/outputs, cost budgets.

Run:  python langgraph_course/16_reliability_production.py
"""

import asyncio
import time
from typing import TypedDict

from langchain_openai import ChatOpenAI
from langgraph.cache.memory import InMemoryCache
from langgraph.errors import GraphRecursionError, NodeError
from langgraph.graph import END, START, StateGraph
from langgraph.types import CachePolicy, Command, RetryPolicy, TimeoutPolicy

from common import ask, banner, get_llm, show


class State(TypedDict, total=False):
    query: str
    result: str
    log: list[str]


# --- 1. RetryPolicy on a flaky tool --------------------------------------------------

class RateLimited(Exception):
    pass


calls = {"n": 0}


def flaky_api(state: State) -> dict:
    calls["n"] += 1
    if calls["n"] < 3:
        print(f"  attempt {calls['n']}: 429 rate limited")
        raise RateLimited()
    print(f"  attempt {calls['n']}: ok")
    return {"result": "data"}


def retry_demo() -> None:
    g = StateGraph(State)
    g.add_node("api", flaky_api, retry_policy=RetryPolicy(max_attempts=5, initial_interval=0.2, retry_on=RateLimited))
    g.add_edge(START, "api")
    show("result", g.compile().invoke({"query": "x"}))


# --- 2. CachePolicy: skip identical LLM calls -----------------------------------------

def summarise(state: State) -> dict:
    return {"result": ask(f"Summarise in 8 words: {state['query']}")}


def cache_demo() -> None:
    g = StateGraph(State)
    g.add_node("summarise", summarise, cache_policy=CachePolicy(ttl=300))
    g.add_edge(START, "summarise")
    graph = g.compile(cache=InMemoryCache())
    for i in range(2):
        t0 = time.perf_counter()
        out = graph.invoke({"query": "LangGraph adds cycles, state and persistence to LLM apps."})
        show(f"call {i + 1} ({time.perf_counter() - t0:.2f}s)", out["result"])


# --- 3. error_handler: degrade gracefully instead of crashing ---------------------------

def primary_search(state: State) -> dict:
    raise ConnectionError("search backend down")


def search_failed(state: State, error: NodeError) -> Command:
    show("error_handler", f"{error.node} failed with {error.error!r} -> fallback")
    return Command(goto="fallback", update={"log": [f"{error.node}: {error.error}"]})


def fallback_answer(state: State) -> dict:
    return {"result": "Search is unavailable; answering from cached knowledge instead."}


def error_handler_demo() -> None:
    g = StateGraph(State)
    g.add_node("search", primary_search, retry_policy=RetryPolicy(max_attempts=2, initial_interval=0.1), error_handler=search_failed)
    g.add_node("fallback", fallback_answer)
    g.add_edge(START, "search")
    g.add_edge("search", END)
    g.add_edge("fallback", END)
    show("result", g.compile().invoke({"query": "x"}))


# --- 4. Timeouts (async nodes; cancellation is cooperative) ----------------------------

async def slow_tool(state: State) -> dict:
    await asyncio.sleep(5)
    return {"result": "too late"}


def timeout_demo() -> None:
    g = StateGraph(State)
    g.add_node("slow", slow_tool, timeout=TimeoutPolicy(run_timeout=1.0), retry_policy=RetryPolicy(max_attempts=1))
    g.add_edge(START, "slow")
    t0 = time.perf_counter()
    try:
        asyncio.run(g.compile().ainvoke({"query": "x"}))
    except Exception as e:  # NodeTimeoutError
        show(f"timed out after {time.perf_counter() - t0:.1f}s", repr(e)[:200])


# --- 5. recursion_limit stops runaway loops -------------------------------------------

def loop_demo() -> None:
    g = StateGraph(State)
    g.add_node("think", lambda s: {"log": (s.get("log") or []) + ["thinking..."]})
    g.add_edge(START, "think")
    g.add_conditional_edges("think", lambda s: "think")  # a buggy "never done" router
    try:
        g.compile().invoke({"query": "x"}, {"recursion_limit": 5})
    except GraphRecursionError as e:
        show("GraphRecursionError", str(e).splitlines()[0])


# --- 6. Model-level fallbacks -----------------------------------------------------------

def fallback_model_demo() -> None:
    broken = ChatOpenAI(base_url="http://localhost:9/v1", api_key="x", model="primary", max_retries=0, timeout=2)  # nothing listens here
    llm = broken.with_fallbacks([get_llm()])
    show("answer via fallback model", llm.invoke("Say 'fallback works' and nothing else.").content)


if __name__ == "__main__":
    for title, fn in [
        ("RetryPolicy", retry_demo),
        ("error_handler (graceful degradation)", error_handler_demo),
        ("TimeoutPolicy", timeout_demo),
        ("recursion_limit", loop_demo),
        ("CachePolicy (2nd call is instant)", cache_demo),
        ("with_fallbacks", fallback_model_demo),
    ]:
        banner(title)
        fn()
