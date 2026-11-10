"""
text_tool_adapter.py
--------------------
Tool calling and structured output for models WITHOUT native support
(e.g. DeepSeek-R1-Distill, many small/older open models, or a vLLM server
started without --tool-call-parser). Used by lesson 03b.

Techniques:
  - describe tools in the prompt, ask for <action>{...}</action> blocks
  - parse the text into a normal AIMessage(tool_calls=[...]) so LangGraph's
    ToolNode / tools_condition work unchanged
  - lenient parsing (models drop tags / emit fenced JSON) restricted to known tool names
  - a one-shot "nudge" re-prompt when the model describes a tool but doesn't call it
  - JSON extraction + validate-and-retry with the error fed back (ask_json)
  - normalising history for models that dislike system/tool roles (to_r1_messages)
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any, Sequence, TypeVar

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ValidationError

from common import get_llm, strip_think

T = TypeVar("T", bound=BaseModel)

def to_r1_messages(messages: Sequence[BaseMessage]) -> list[BaseMessage]:
    """Make an arbitrary LangChain history safe for R1.

    - system messages are merged into the first user message
    - tool calls / tool results become plain assistant / user text
    - consecutive same-role messages are merged (the chat template dislikes them)
    """
    system_parts: list[str] = []
    out: list[BaseMessage] = []

    for m in messages:
        if isinstance(m, SystemMessage):
            system_parts.append(str(m.content))
            continue
        if isinstance(m, ToolMessage):
            new: BaseMessage = HumanMessage(f"[tool result: {m.name}]\n{m.content}")
        elif isinstance(m, AIMessage):
            text = strip_think(str(m.content))
            for tc in m.tool_calls:
                text += "\n<action>" + json.dumps({"name": tc["name"], "args": tc["args"]}) + "</action>"
            new = AIMessage(text.strip() or "(no content)")
        else:
            new = HumanMessage(str(m.content))

        if out and type(out[-1]) is type(new):
            out[-1] = type(new)(f"{out[-1].content}\n\n{new.content}")
        else:
            out.append(new)

    if system_parts:
        sys_text = "\n\n".join(system_parts)
        if out and isinstance(out[0], HumanMessage):
            out[0] = HumanMessage(f"{sys_text}\n\n{out[0].content}")
        else:
            out.insert(0, HumanMessage(sys_text))
    return out



# ---------------------------------------------------------------------------
# Structured output (JSON -> Pydantic, with self-correcting retries)
# ---------------------------------------------------------------------------

def extract_json(text: str) -> Any:
    """Pull the last JSON object/array out of free text (fenced or bare)."""
    fenced = re.findall(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    candidates = fenced[::-1] + [text]
    for cand in candidates:
        cand = cand.strip()
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            pass
        # Fall back to the last balanced {...} block.
        for start in [m.start() for m in re.finditer(r"[{\[]", cand)][::-1]:
            depth, opener = 0, cand[start]
            closer = "}" if opener == "{" else "]"
            for i in range(start, len(cand)):
                if cand[i] == opener:
                    depth += 1
                elif cand[i] == closer:
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(cand[start : i + 1])
                        except json.JSONDecodeError:
                            break
    raise ValueError(f"No JSON found in: {text[:200]!r}")


def ask_json(prompt: str, schema: type[T], retries: int = 2, llm: ChatOpenAI | None = None) -> T:
    """Ask for JSON matching `schema`; on parse/validation failure, feed the error back.

    This is what `llm.with_structured_output(Schema)` does for models that
    support JSON mode or tool calling. For models that don't, a
    validate-and-retry loop is the standard technique.
    """
    llm = llm or get_llm()
    schema_json = json.dumps(schema.model_json_schema(), indent=None)
    msgs: list[BaseMessage] = [
        HumanMessage(
            f"{prompt}\n\nRespond with ONLY a JSON object that matches this JSON schema "
            f"(no prose after it):\n{schema_json}"
        )
    ]
    last_err: Exception | None = None
    for _ in range(retries + 1):
        raw = strip_think(str(llm.invoke(msgs).content))
        try:
            return schema.model_validate(extract_json(raw))
        except (ValueError, ValidationError) as e:
            last_err = e
            msgs += [AIMessage(raw or "(empty)"), HumanMessage(f"That was invalid: {e}\nReturn ONLY the corrected JSON.")]
    raise ValueError(f"Model never produced valid {schema.__name__}: {last_err}")


# ---------------------------------------------------------------------------
# Tool calling for models without native function calling
# ---------------------------------------------------------------------------

_TOOL_CALL_RE = re.compile(r"<action>\s*(.*?)\s*</action>", re.DOTALL)


def render_tools(tools: Sequence[BaseTool]) -> str:
    lines = []
    for t in tools:
        fn = convert_to_openai_tool(t)["function"]
        params = fn.get("parameters", {}).get("properties", {})
        args = ", ".join(f'"{k}": {v.get("type", "any")}' for k, v in params.items())
        lines.append(f"- {fn['name']}({{{args}}}): {fn.get('description', '').strip()}")
    return "\n".join(lines)


TOOL_PROTOCOL = """## Tools
{tools}

## How to call a tool
Reply with a block like this and nothing after it:
<action>{{"name": "<tool name>", "args": {{"<arg>": "<value>"}}}}</action>
You may emit several blocks only if they are independent of each other.
If a call needs the result of another call, make the first call, STOP, and wait:
the result comes back in the next message as "[tool result: <name>]".
Never guess or invent tool results, and only use values that appear in the conversation.
When you have everything you need, reply with the final answer as plain text, with no <action> block."""


class TextToolCallingModel:
    """Wraps a chat model so `.invoke(messages)` returns an AIMessage whose
    `tool_calls` are parsed from the text protocol above.

    Usage mirrors `llm.bind_tools(tools)`:
        model = TextToolCallingModel(get_llm(), tools)
        ai_msg = model.invoke(state["messages"])   # -> AIMessage(tool_calls=[...])
    """

    def __init__(self, llm: ChatOpenAI, tools: Sequence[BaseTool], instructions: str = ""):
        self.llm = llm
        self.tools = list(tools)
        self.instructions = instructions

    def invoke(self, messages: Sequence[BaseMessage]) -> AIMessage:
        protocol = TOOL_PROTOCOL.format(tools=render_tools(self.tools))
        system = f"{self.instructions}\n\n{protocol}".strip()
        names = {t.name for t in self.tools}
        msgs = to_r1_messages([SystemMessage(system), *messages])
        text = strip_think(str(self.llm.invoke(msgs).content))
        ai = parse_tool_calls(text, names)
        # Nudge: the model *talked about* a tool but didn't call it (or ran out of
        # tokens while thinking). Re-prompt once with an explicit correction.
        if not ai.tool_calls and (not text or any(n in text for n in names)):
            msgs += [AIMessage(text or "(no answer)"), HumanMessage(
                "You described tool use but did not call a tool. If a tool is needed, reply with ONLY the "
                "single next <action> block now. Otherwise reply with the final answer only."
            )]
            text = strip_think(str(self.llm.invoke(msgs).content))
            ai = parse_tool_calls(text, names)
        return ai


def _json_objects(text: str):
    """Yield (start, end, obj) for every top-level balanced {...} that parses as JSON."""
    i = 0
    while (start := text.find("{", i)) != -1:
        depth, in_str, esc = 0, False, False
        for j in range(start, len(text)):
            c = text[j]
            if in_str:
                esc, in_str = (not esc and c == "\\"), (in_str if esc or c != '"' else False)
                continue
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        yield start, j + 1, json.loads(text[start : j + 1])
                    except json.JSONDecodeError:
                        pass
                    break
        i = start + 1 if depth else j + 1


def parse_tool_calls(text: str, tool_names: set[str] | None = None) -> AIMessage:
    """Parse <action> blocks into AIMessage.tool_calls.

    Small models often drop the tags and write the call as (fenced) JSON, so
    as a fallback accept the FIRST bare {"name": <known tool>, "args": {...}}
    object. Only the first, because later ones usually depend on results the
    model hasn't seen yet (it is narrating a plan, not calling).
    """
    def to_call(data: Any) -> dict | None:
        if isinstance(data, dict) and isinstance(data.get("name"), str) and (tool_names is None or data["name"] in tool_names):
            return {"name": data["name"], "args": data.get("args") or {}, "id": f"call_{uuid.uuid4().hex[:12]}", "type": "tool_call"}
        return None

    calls = []
    for block in _TOOL_CALL_RE.findall(text):
        try:
            if call := to_call(extract_json(block)):
                calls.append(call)
        except ValueError:
            continue
    if calls:
        return AIMessage(content=_TOOL_CALL_RE.sub("", text).strip(), tool_calls=calls)

    for start, _, obj in _json_objects(text):
        if call := to_call(obj):
            content = re.sub(r"```(?:json)?\s*$", "", text[:start].rstrip())
            return AIMessage(content=content.strip(), tool_calls=[call])
    return AIMessage(content=text.strip())
