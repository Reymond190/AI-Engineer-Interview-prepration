"""
17 - Agentic RAG: route, retrieve, grade, rewrite, generate, verify
===================================================================

Interview topics
----------------
- Naive RAG: retrieve top-k -> stuff into prompt -> answer. Fails silently
  when retrieval misses or the model ignores/overrides the context.
- Agentic RAG puts decisions around retrieval:
    route        does this question need retrieval at all?
    grade docs   drop irrelevant chunks (CRAG: Corrective RAG)
    rewrite      if nothing relevant, reformulate the query and retry
                 (or fall back to web search)
    generate     answer with citations to chunk ids
    verify       is every claim supported by the cited chunks? (Self-RAG /
                 groundedness check); regenerate or abstain if not
- Retrieval quality matters more than any of this: chunking, hybrid search
  (BM25 + embeddings), reranking, metadata filters. Here: a tiny keyword
  scorer so the demo has no dependencies.
- Evaluate with retrieval metrics (recall@k, MRR) AND answer metrics
  (faithfulness, answer relevance), e.g. RAGAS-style LLM judges (lesson 21).

Run:  python langgraph_course/17_agentic_rag.py
"""

import re
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from common import ask, banner, show, structured

CORPUS = {
    "hr-01": "Employees accrue 2.5 days of paid vacation per month, up to 30 days per year. Unused days expire on March 31 of the following year.",
    "hr-02": "Parental leave is 16 weeks fully paid for the primary caregiver and 6 weeks for the secondary caregiver.",
    "it-01": "Laptops are refreshed every 3 years. Engineers may request a 64GB RAM model with manager approval.",
    "it-02": "VPN access requires the Okta Verify app. Lost hardware tokens must be reported to security within 24 hours.",
    "fin-01": "Travel expenses above 500 EUR need pre-approval. Receipts must be uploaded within 30 days of the trip.",
    "fin-02": "The company reimburses home-office equipment up to 400 EUR once every two years.",
}


def keyword_search(query: str, k: int = 3) -> list[tuple[str, str]]:
    stop = {"the", "a", "an", "is", "of", "to", "for", "how", "what", "do", "i", "my", "can", "many", "much", "are", "in", "on"}
    terms = {w for w in re.findall(r"[a-z0-9]+", query.lower()) if w not in stop}
    scored = []
    for doc_id, text in CORPUS.items():
        words = set(re.findall(r"[a-z0-9]+", text.lower()))
        score = sum(1 for t in terms if any(w.startswith(t[:5]) for w in words))
        if score:
            scored.append((score, doc_id, text))
    return [(d, t) for _, d, t in sorted(scored, reverse=True)[:k]]


class State(TypedDict, total=False):
    question: str
    query: str
    docs: list[tuple[str, str]]
    rewrites: int
    answer: str
    grounded: bool
    attempts: int


class RouteQ(BaseModel):
    # A router is only as good as its description of each destination: say what the docs COVER.
    datasource: Literal["company_docs", "direct"] = Field(
        description="company_docs: anything about the user's employer, job, benefits or workplace rules "
        "(vacation, leave, equipment, laptops, VPN, expenses, reimbursements, stock/compensation). "
        "direct: general world knowledge unrelated to the company (geography, science, chit-chat)."
    )


class Relevance(BaseModel):
    relevant: bool


class Grounded(BaseModel):
    grounded: bool = Field(description="True only if every claim in the answer is supported by the documents")
    unsupported: str = Field(description="Unsupported claims, if any")


router, grader, checker = structured(RouteQ), structured(Relevance), structured(Grounded)


def route_question(state: State) -> Literal["retrieve", "direct_answer"]:
    # Few-shot examples fix routing mistakes that descriptions alone don't.
    r = router.invoke(
        "Route the question.\nExamples:\n"
        "- 'How do I reset my VPN token?' -> company_docs\n"
        "- 'Who wrote Hamlet?' -> direct\n"
        "- 'Is my gym membership reimbursed?' -> company_docs\n"
        "- 'How far is the Moon?' -> direct\n"
        f"Question: {state['question']}"
    )
    show("route", r.datasource)
    return "retrieve" if r.datasource == "company_docs" else "direct_answer"


def direct_answer(state: State) -> dict:
    return {"answer": ask(state["question"])}


def retrieve(state: State) -> dict:
    query = state.get("query") or state["question"]
    docs = keyword_search(query)
    show("retrieved", [d for d, _ in docs])
    return {"docs": docs, "query": query}


def grade_documents(state: State) -> dict:
    kept = [(d, t) for d, t in state["docs"] if grader.invoke(f"Question: {state['question']}\nDocument: {t}\nIs the document relevant to the question?").relevant]
    show("relevant after grading", [d for d, _ in kept])
    return {"docs": kept}


def after_grading(state: State) -> Literal["generate", "rewrite", "abstain"]:
    if state["docs"]:
        return "generate"
    return "rewrite" if state.get("rewrites", 0) < 2 else "abstain"


def rewrite(state: State) -> dict:
    q = ask(f"Rewrite this as a short keyword search query for an internal policy wiki (output only the query): {state['question']}")
    show("rewritten query", q)
    return {"query": q, "rewrites": state.get("rewrites", 0) + 1}


def generate(state: State) -> dict:
    ctx = "\n".join(f"[{d}] {t}" for d, t in state["docs"])
    extra = "" if state.get("grounded", True) else "\nYour previous answer contained unsupported claims. Use ONLY the documents."
    answer = ask(f"Answer using only these documents and cite ids like [hr-01].{extra}\n{ctx}\n\nQuestion: {state['question']}")
    return {"answer": answer, "attempts": state.get("attempts", 0) + 1}


def verify(state: State) -> dict:
    ctx = "\n".join(f"[{d}] {t}" for d, t in state["docs"])
    g = checker.invoke(f"Documents:\n{ctx}\n\nAnswer:\n{state['answer']}\n\nIs the answer fully supported by the documents?")
    show("grounded", g)
    return {"grounded": g.grounded}


def after_verify(state: State) -> Literal["generate", "__end__"]:
    return "generate" if not state["grounded"] and state["attempts"] < 2 else END


def abstain(state: State) -> dict:
    return {"answer": "I couldn't find this in the company documents. Please ask HR/IT directly."}


builder = StateGraph(State)
for name, fn in [("direct_answer", direct_answer), ("retrieve", retrieve), ("grade", grade_documents), ("rewrite", rewrite), ("generate", generate), ("verify", verify), ("abstain", abstain)]:
    builder.add_node(name, fn)
builder.add_conditional_edges(START, route_question)
builder.add_edge("retrieve", "grade")
builder.add_conditional_edges("grade", after_grading)
builder.add_edge("rewrite", "retrieve")
builder.add_edge("generate", "verify")
builder.add_conditional_edges("verify", after_verify)
builder.add_edge("direct_answer", END)
builder.add_edge("abstain", END)
graph = builder.compile()


if __name__ == "__main__":
    for q in [
        "How many vacation days can I carry over, and when do they expire?",
        "Can I get money back for buying a desk chair for working from home?",  # vocabulary mismatch
        "What is the capital of Australia?",  # no retrieval needed
        "What is the company's stock option vesting schedule?",  # not in corpus -> abstain
    ]:
        banner(q)
        out = graph.invoke({"question": q}, {"recursion_limit": 25})
        show("answer", out["answer"], width=600)
