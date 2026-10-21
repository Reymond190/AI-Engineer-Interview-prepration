"""
15 - The Functional API: @entrypoint and @task
==============================================

Interview topics
----------------
- Same runtime as StateGraph (checkpointing, interrupts, streaming, memory),
  but written as ordinary Python control flow (if/for/while) instead of
  nodes + edges. Good when the flow is naturally procedural.
- @task: a unit of work whose result is checkpointed. Calling a task returns a
  future; call .result() to wait. Calling several tasks before .result() runs
  them in PARALLEL.
- @entrypoint(checkpointer=...): the workflow. On resume after an interrupt or
  crash, completed tasks are NOT re-run; their saved results are replayed.
  That's why side effects / LLM calls belong inside tasks (determinism rule).
- `previous`: an entrypoint parameter that receives the last return value for
  the thread (simple short-term memory). entrypoint.final(value=, save=) lets
  you return one thing and save another.
- Graph API vs Functional API: graph gives visualisation and explicit
  state/reducers; functional gives less boilerplate. They interoperate: a
  task can call a compiled graph and vice versa.

Run:  python langgraph_course/15_functional_api.py
"""

import time

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.func import entrypoint, task
from langgraph.types import Command, interrupt

from common import ask, banner, show


@task
def outline(topic: str) -> list[str]:
    text = ask(f"List exactly 3 short section titles (one per line, no numbering) for a post about: {topic}")
    return [line.strip("-*# 0123456789.").strip() for line in text.splitlines() if line.strip()][:3]


@task
def write_section(topic: str, title: str) -> str:
    return ask(f"Write 2 sentences for the section '{title}' of a post about {topic}.")


@task
def publish(post: str) -> str:
    # Side effect lives in a task, so a resume will not publish twice.
    return f"published {len(post)} chars"


@entrypoint(checkpointer=InMemorySaver())
def blog_workflow(topic: str) -> dict:
    titles = outline(topic).result()

    t0 = time.perf_counter()
    futures = [write_section(topic, t) for t in titles]  # fan out: all start now
    sections = [f.result() for f in futures]  # join
    show("parallel sections took (s)", round(time.perf_counter() - t0, 1))

    post = "\n\n".join(f"## {t}\n{s}" for t, s in zip(titles, sections))
    decision = interrupt({"review": post[:300] + "...", "question": "publish? (yes/no)"})
    if decision != "yes":
        return {"status": "rejected", "post": post}
    return {"status": publish(post).result(), "post": post}


# `previous` = what this entrypoint returned last time on the same thread.
@entrypoint(checkpointer=InMemorySaver())
def counter(increment: int, *, previous: int | None = None) -> int:
    return (previous or 0) + increment


if __name__ == "__main__":
    banner("Functional workflow with parallel tasks + interrupt")
    config = {"configurable": {"thread_id": "blog-1"}}
    for event in blog_workflow.stream("idempotency in agent tools", config, stream_mode="updates"):
        name = next(iter(event))
        print(f"  event: {name}")
        if name == "__interrupt__":
            show("waiting for human", event["__interrupt__"][0].value["question"])

    banner("Resume: outline + sections are NOT recomputed (replayed from checkpoint)")
    t0 = time.perf_counter()
    result = blog_workflow.invoke(Command(resume="yes"), config)
    show("resume took (s)", round(time.perf_counter() - t0, 2))
    show("status", result["status"])
    print(result["post"])

    banner("`previous` as per-thread memory")
    cfg = {"configurable": {"thread_id": "c"}}
    print([counter.invoke(n, cfg) for n in (1, 2, 3)])
