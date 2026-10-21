"""
18 - Network / collaboration pattern: multi-agent debate with a judge
=====================================================================

Interview topics
----------------
- Network (collaboration) topology: agents share a scratchpad (message list)
  and take turns; any agent can see everything the others said. Contrast with
  supervisor (star topology) and hierarchical (tree).
- Multi-agent debate (Du et al. 2023, "Improving Factuality and Reasoning
  through Multiagent Debate"): several agents argue, critique each other over
  rounds, then a judge (or majority vote) decides. Helps on reasoning and
  factuality; costs N x rounds LLM calls.
- Design details that matter:
    * distinct personas / instructions, otherwise agents just agree
    * fixed round budget (termination is a code decision)
    * judge with structured output and a rubric, ideally a different/stronger model
    * name each message (AIMessage(name=...)) so agents know who said what
    * debaters under pressure to "win" FABRICATE evidence (fake pilot stats), and
      a naive judge rewards it. Instruct against it and make the judge penalise
      unverifiable claims, or give debaters a retrieval tool and require citations.
- Related: "LLM council" / mixture-of-agents (several models answer, an
  aggregator combines), and self-consistency (sample N, majority vote).

Run:  python langgraph_course/18_debate_network.py
"""

from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from common import banner, get_llm, show, structured

ROUNDS = 2
llm = get_llm(max_tokens=400)

NO_FABRICATION = " Do not invent statistics, studies or pilot results; argue from reasoning and well-known facts."
PERSONAS = {
    "optimist": "You argue FOR the proposal. Be concrete and rebut the other side's latest points. Max 4 sentences." + NO_FABRICATION,
    "skeptic": "You argue AGAINST the proposal. Find risks, hidden costs, failure modes; rebut the other side's latest points. Max 4 sentences." + NO_FABRICATION,
}


class Verdict(BaseModel):
    winner: Literal["optimist", "skeptic"]
    decision: str = Field(description="Recommended decision in one sentence")
    strongest_argument: str
    rubric_scores: dict[str, int] = Field(description="Score 1-5 for each side on evidence and rebuttal quality, e.g. {'optimist': 3, 'skeptic': 4}")


judge = structured(Verdict, thinking=True)


class State(TypedDict, total=False):
    proposal: str
    messages: Annotated[list[AnyMessage], add_messages]
    round: int
    verdict: Verdict


def transcript(messages: list[AnyMessage]) -> str:
    return "\n".join(f"{m.name}: {m.content}" for m in messages if isinstance(m, AIMessage))


def debater(name: str):
    def node(state: State) -> dict:
        prompt = [
            SystemMessage(PERSONAS[name]),
            HumanMessage(f"Proposal: {state['proposal']}\n\nDebate so far:\n{transcript(state.get('messages', [])) or '(you open)'}\n\nYour turn, {name}:"),
        ]
        reply = llm.invoke(prompt)
        print(f"\n[{name}] {reply.content}")
        return {"messages": [AIMessage(reply.content, name=name)]}

    return node


def next_round(state: State) -> dict:
    return {"round": state.get("round", 0) + 1}


def continue_or_judge(state: State) -> Literal["optimist", "judge"]:
    return "optimist" if state["round"] < ROUNDS else "judge"


def judge_node(state: State) -> dict:
    v = judge.invoke(
        f"You are an impartial judge. Proposal: {state['proposal']}\n\nDebate:\n{transcript(state['messages'])}\n\n"
        "Decide which side argued better. Heavily penalise specific numbers or studies that cannot be verified: "
        "debaters are LLMs and may fabricate evidence. Reward sound reasoning and direct rebuttals."
    )
    return {"verdict": v}


builder = StateGraph(State)
builder.add_node("optimist", debater("optimist"))
builder.add_node("skeptic", debater("skeptic"))
builder.add_node("next_round", next_round)
builder.add_node("judge", judge_node)
builder.add_edge(START, "optimist")
builder.add_edge("optimist", "skeptic")
builder.add_edge("skeptic", "next_round")
builder.add_conditional_edges("next_round", continue_or_judge)
builder.add_edge("judge", END)
graph = builder.compile()


if __name__ == "__main__":
    banner("Debate")
    out = graph.invoke({"proposal": "Replace our rule-based customer-support chatbot with a multi-agent LLM system this quarter."})
    banner("Verdict")
    show("verdict", out["verdict"], width=1500)
