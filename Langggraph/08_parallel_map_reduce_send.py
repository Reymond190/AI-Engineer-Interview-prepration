"""
08 - Parallelism: static fan-out, Send API map-reduce, deferred joins
=====================================================================

Interview topics
----------------
- Static fan-out: several edges from one node -> the targets run in the SAME
  super-step, in parallel. A list-source edge add_edge(["a","b"], "join")
  waits for all of them.
- Dynamic fan-out (map-reduce): the number of branches is only known at
  runtime (N documents, N sub-questions). A conditional edge returns
  [Send("worker", payload), ...]; each Send runs `worker` with its OWN input
  (not the global state). Results come back through a reducer
  (Annotated[list, operator.add]) = the "reduce".
- This is the orchestrator-workers / sectioning pattern from Anthropic's
  "Building effective agents", and the core of "deep research" systems
  (lead agent spawns N search sub-agents in parallel).
- defer=True on a node: run it only after every other pending branch has
  finished (useful for joins over branches of uneven length).
- Concurrency limit: config {"max_concurrency": k}. Here vLLM is started with
  --max-num-seqs 4, so ~4 requests actually decode at once.
- Failure semantics: if one branch raises, the super-step fails; completed
  branches' writes are saved by the checkpointer ("pending writes"), so a
  resume only re-runs what failed.

Run:  python langgraph_course/08_parallel_map_reduce_send.py
"""

import operator
import time
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field

from common import ask, banner, show, structured


class SubQuestions(BaseModel):
    questions: list[str] = Field(description="3 to 5 independent sub-questions", min_length=2, max_length=5)


class Finding(TypedDict):
    question: str
    answer: str
    start: float
    end: float


class State(TypedDict, total=False):
    question: str
    sub_questions: list[str]
    findings: Annotated[list[Finding], operator.add]  # the reduce
    report: str


class WorkerInput(TypedDict):  # what each Send carries
    sub_question: str


planner = structured(SubQuestions)


def plan(state: State) -> dict:
    sq = planner.invoke(f"Break this research question into independent sub-questions:\n{state['question']}")
    show("plan", sq.questions)
    return {"sub_questions": sq.questions}


def fan_out(state: State) -> list[Send]:
    # One Send per sub-question; each worker gets its own small payload.
    return [Send("research", {"sub_question": q}) for q in state["sub_questions"]]


def research(inp: WorkerInput) -> dict:
    start = time.perf_counter()
    answer = ask(f"Answer concisely (max 3 sentences): {inp['sub_question']}")
    return {"findings": [{"question": inp["sub_question"], "answer": answer, "start": start, "end": time.perf_counter()}]}


def synthesize(state: State) -> dict:
    notes = "\n\n".join(f"Q: {f['question']}\nA: {f['answer']}" for f in state["findings"])
    return {"report": ask(f"Write a short, structured answer to '{state['question']}' using these findings:\n{notes}")}


builder = StateGraph(State)
builder.add_node("plan", plan)
builder.add_node("research", research)
builder.add_node("synthesize", synthesize)
builder.add_edge(START, "plan")
builder.add_conditional_edges("plan", fan_out, ["research"])
builder.add_edge("research", "synthesize")  # runs once, after ALL Sends complete
builder.add_edge("synthesize", END)
graph = builder.compile()


def defer_demo() -> None:
    """Branches of uneven length: without defer, `join` would fire early."""

    class S(TypedDict):
        trail: Annotated[list[str], operator.add]

    g = StateGraph(S)
    g.add_node("short", lambda s: {"trail": ["short"]})
    g.add_node("long_1", lambda s: {"trail": ["long_1"]})
    g.add_node("long_2", lambda s: {"trail": ["long_2"]})
    g.add_node("join", lambda s: {"trail": ["join"]}, defer=True)
    g.add_edge(START, "short")
    g.add_edge(START, "long_1")
    g.add_edge("long_1", "long_2")
    g.add_edge("short", "join")
    g.add_edge("long_2", "join")
    g.add_edge("join", END)
    show("defer=True trail", g.compile().invoke({"trail": []})["trail"])


if __name__ == "__main__":
    banner("Deferred join")
    defer_demo()

    banner("Map-reduce research with Send")
    out = graph.invoke(
        {"question": "What should a team consider before putting an LLM agent into production?"},
        {"max_concurrency": 4},
    )
    findings = out["findings"]
    for f in findings:
        show(f"{f['end'] - f['start']:.1f}s", f["question"])
    sequential = sum(f["end"] - f["start"] for f in findings)
    parallel = max(f["end"] for f in findings) - min(f["start"] for f in findings)
    show("fan-out phase", f"{parallel:.1f}s wall clock vs {sequential:.1f}s if run one by one")
    show("report", out["report"], width=1500)
