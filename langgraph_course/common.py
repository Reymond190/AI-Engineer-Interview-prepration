"""
common.py
---------
Shared helpers for every lesson in this course.

The local model is Qwen3.5-9B (INT4 w4a16) served by vLLM with:
  --enable-auto-tool-choice --tool-call-parser qwen3_coder -> native tool calling
  --reasoning-parser qwen3                               -> <think> is split out
                                                            of `content`
So the lessons use the standard LangChain APIs (`bind_tools`,
`with_structured_output`, `ToolNode`) exactly as you would with OpenAI or
Anthropic models. Swap `get_llm()` for `ChatOpenAI(model="gpt-...")` or
`ChatAnthropic(...)` and every lesson still runs.

Qwen3 is a *hybrid* reasoning model: thinking can be switched per request.
  get_llm()               -> fast mode (no thinking): routing, tool calls, extraction
  get_llm(thinking=True)  -> deliberate mode: planning, critique, hard reasoning
Choosing the mode per node is itself a useful design point (latency/cost vs quality).
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Sequence

import httpx
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# Windows consoles default to cp1252; make unicode output safe.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_URL = os.getenv("AGENT_BASE_URL", "http://localhost:8000/v1")


def _served_model() -> str:
    """Use whatever model the server is running (so switching serve.sh profiles needs no config)."""
    try:
        return httpx.get(f"{BASE_URL}/models", timeout=3).json()["data"][0]["id"]
    except Exception:  # noqa: BLE001  server down: fall back to the default
        return "Qwen/Qwen3.5-9B"


MODEL = os.getenv("AGENT_MODEL") or _served_model()

# Sampling recommended by each model card: (temperature, top_p, top_k, presence_penalty)
SAMPLING = {
    "Qwen3.5": {True: (1.0, 0.95, 20, 1.5), False: (0.7, 0.8, 20, 1.5)},
    "Qwen3": {True: (0.6, 0.95, 20, 0.0), False: (0.7, 0.8, 20, 0.0)},
}


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def get_llm(thinking: bool = False, max_tokens: int | None = None, temperature: float | None = None, **kwargs: Any) -> ChatOpenAI:
    """Chat model pointed at the local vLLM server, with the model card's sampling."""
    family = next((f for f in SAMPLING if f.lower() in MODEL.lower()), "Qwen3")
    temp, top_p, top_k, presence = SAMPLING[family][thinking]
    return ChatOpenAI(
        base_url=BASE_URL,
        api_key="not-needed",
        model=MODEL,
        temperature=temperature if temperature is not None else temp,
        top_p=top_p,
        presence_penalty=presence,
        # Thinking traces can run to thousands of tokens; if the budget is hit
        # before </think>, there is no answer at all (LengthFinishReasonError).
        max_tokens=max_tokens or (8192 if thinking else 2048),
        timeout=300,
        extra_body={"top_k": top_k, "chat_template_kwargs": {"enable_thinking": thinking}},
        **kwargs,
    )


def strip_think(text: str) -> str:
    """Safety net: drop a <think> block if one ever leaks into `content`."""
    return re.sub(r"^.*?</think>", "", text, count=1, flags=re.DOTALL).strip() if "</think>" in text else text.strip()


def ask(prompt: str | Sequence[BaseMessage], system: str | None = None, thinking: bool = False, **llm_kwargs: Any) -> str:
    """One LLM call, returns the answer text."""
    msgs: list[BaseMessage] = [SystemMessage(system)] if system else []
    msgs += [HumanMessage(prompt)] if isinstance(prompt, str) else list(prompt)
    return strip_think(str(get_llm(thinking=thinking, **llm_kwargs).invoke(msgs).content))


def describe_schema(schema: dict, indent: str = "") -> str:
    """Render a JSON schema as a short human-readable field list."""
    defs = schema.get("$defs", {})

    def resolve(p: dict) -> dict:
        return defs[p["$ref"].split("/")[-1]] if "$ref" in p else p

    lines = []
    for name, prop in schema.get("properties", {}).items():
        prop = resolve(prop)
        kind = prop.get("type", "object")
        if "enum" in prop:
            kind = " | ".join(map(str, prop["enum"]))
        elif kind == "array":
            kind = f"list of {resolve(prop.get('items', {})).get('type', 'object')}"
        desc = f": {prop['description']}" if prop.get("description") else ""
        lines.append(f"{indent}- {name} ({kind}){desc}")
        nested = resolve(prop.get("items", {})) if prop.get("type") == "array" else prop
        if nested.get("properties"):
            lines.append(describe_schema({**nested, "$defs": defs}, indent + "  "))
    return "\n".join(lines)


def structured(schema: type, thinking: bool = False, **llm_kwargs: Any):
    """`llm.with_structured_output(schema)`, plus the schema in the prompt.

    On vLLM, method="json_schema" maps to guided (constrained) decoding, so the
    output is guaranteed to parse, and Pydantic still validates field constraints.
    Gotcha: the grammar enforces the SHAPE, but the model never SEES the schema.
    OpenAI injects it server-side; vLLM does not. Without the hint below, field
    descriptions (rubrics, routing definitions) are invisible to the model.
    """
    llm = get_llm(thinking=thinking, **llm_kwargs).with_structured_output(schema, method="json_schema")
    # A readable field list, NOT a raw JSON-schema dump: measured on the lesson-02
    # router, the raw dump flipped 3/3 correct "billing" answers to "general".
    hint = SystemMessage("Reply with a JSON object with these fields:\n" + describe_schema(schema.model_json_schema()))

    def add_hint(inp: str | Sequence[BaseMessage]) -> list[BaseMessage]:
        return [hint, *([HumanMessage(inp)] if isinstance(inp, str) else inp)]

    return RunnableLambda(add_hint) | llm


def make_react_agent(tools: Sequence[BaseTool], instructions: str, name: str = "agent", thinking: bool = False):
    """Compiled ReAct sub-agent (model <-> tools loop), same shape as lesson 03.

    Used as a building block by the multi-agent lessons. Equivalent to
    `langchain.agents.create_agent(model, tools, system_prompt=...)`.
    """
    from langgraph.graph import START, MessagesState, StateGraph
    from langgraph.prebuilt import ToolNode, tools_condition

    model = get_llm(thinking=thinking).bind_tools(tools)

    def call_model(state: MessagesState) -> dict:
        return {"messages": [model.invoke([SystemMessage(instructions), *state["messages"]])]}

    g = StateGraph(MessagesState)
    g.add_node("model", call_model)
    g.add_node("tools", ToolNode(list(tools), handle_tool_errors=True))
    g.add_edge(START, "model")
    g.add_conditional_edges("model", tools_condition)
    g.add_edge("tools", "model")
    return g.compile(name=name)


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def banner(title: str) -> None:
    print(f"\n{'=' * 70}\n  {title}\n{'=' * 70}")


def show(label: str, value: Any, width: int = 400) -> None:
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    text = value if isinstance(value, str) else json.dumps(value, default=str, indent=2)
    if len(text) > width:
        text = text[:width] + " ..."
    print(f"[{label}] {text}")


def print_messages(messages: Sequence[BaseMessage], width: int = 300) -> None:
    for m in messages:
        role = type(m).__name__.replace("Message", "")
        content = strip_think(str(m.content))
        extra = f"  tool_calls={[(c['name'], c['args']) for c in m.tool_calls]}" if isinstance(m, AIMessage) and m.tool_calls else ""
        name = f"({m.name})" if getattr(m, "name", None) else ""
        print(f"  {role}{name}: {content[:width]}{extra}")


if __name__ == "__main__":
    banner(f"Connectivity check -> {BASE_URL} ({MODEL})")
    print("fast    :", ask("In one sentence, what is LangGraph?"))
    print("thinking:", ask("Is 391 prime? Answer yes/no and the factors.", thinking=True))
