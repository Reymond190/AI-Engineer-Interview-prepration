"""
agent/schema.py
---------------
Pydantic models and parsing logic for the tool-use JSON action schema.

The harness instructs the model to produce tool calls inside
<tool_call>...</tool_call> XML-like tags and final answers inside
<final_answer>...</final_answer> tags.

Reasoning traces wrapped in <think>...</think> are stripped before
parsing so they do not interfere with extraction.
"""

from __future__ import annotations

import json
import re
from enum import Enum
from typing import Any

from pydantic import BaseModel, ValidationError, field_validator


# ---------------------------------------------------------------------------
# Allowed tool names
# ---------------------------------------------------------------------------

class ToolName(str, Enum):
    CALCULATOR = "calculator"
    READ_FILE  = "read_file"
    SHELL      = "shell"


# ---------------------------------------------------------------------------
# Per-tool argument schemas
# ---------------------------------------------------------------------------

class CalculatorArgs(BaseModel):
    """Evaluate a safe arithmetic expression."""
    expression: str

    @field_validator("expression")
    @classmethod
    def no_dangerous_chars(cls, v: str) -> str:
        # Allow only digits, operators, parens, dots, spaces
        if re.search(r"[^0-9+\-*/().\s%]", v):
            raise ValueError(f"Unsafe characters in expression: {v!r}")
        return v


class ReadFileArgs(BaseModel):
    """Read a file from the fixtures directory (read-only, path-contained)."""
    filename: str

    @field_validator("filename")
    @classmethod
    def safe_filename(cls, v: str) -> str:
        # Prevent directory traversal
        if ".." in v or "/" in v or "\\" in v:
            raise ValueError(f"Path traversal not allowed: {v!r}")
        return v


SHELL_ALLOWLIST: frozenset[str] = frozenset(
    ["ls", "pwd", "date", "echo", "uname", "hostname", "whoami", "df", "free"]
)


class ShellArgs(BaseModel):
    """Run an allowlisted read-only shell command."""
    command: str

    @field_validator("command")
    @classmethod
    def must_be_allowed(cls, v: str) -> str:
        base = v.strip().split()[0] if v.strip() else ""
        if base not in SHELL_ALLOWLIST:
            raise ValueError(
                f"Command {base!r} not in allowlist. "
                f"Allowed: {sorted(SHELL_ALLOWLIST)}"
            )
        return v


ARGS_SCHEMA: dict[ToolName, type[BaseModel]] = {
    ToolName.CALCULATOR: CalculatorArgs,
    ToolName.READ_FILE:  ReadFileArgs,
    ToolName.SHELL:      ShellArgs,
}


# ---------------------------------------------------------------------------
# Parsed model output
# ---------------------------------------------------------------------------

class ToolCall(BaseModel):
    action: ToolName
    args:   dict[str, Any]

    def validated_args(self) -> BaseModel:
        """Parse and validate args according to the tool's schema."""
        schema_cls = ARGS_SCHEMA[self.action]
        return schema_cls(**self.args)


class ParseResult(BaseModel):
    kind: str                  # "tool_call" | "final_answer" | "malformed" | "empty"
    tool_call:    ToolCall | None = None
    final_answer: str | None  = None
    raw:          str         = ""
    error:        str | None  = None


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

_THINK_RE        = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_TOOL_CALL_RE    = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL | re.IGNORECASE)
_FINAL_RE        = re.compile(r"<final_answer>(.*?)</final_answer>", re.DOTALL | re.IGNORECASE)


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> reasoning traces from model output."""
    return _THINK_RE.sub("", text).strip()


def parse_model_output(raw: str) -> ParseResult:
    """
    Parse a single model turn into a typed ParseResult.

    Priority:
      1. <tool_call> block → attempt JSON parse and Pydantic validation
      2. <final_answer> block → return final answer text
      3. Neither present but non-empty → treat as final answer (graceful)
      4. Empty → return empty kind
    """
    stripped = strip_thinking(raw)

    tool_match  = _TOOL_CALL_RE.search(stripped)
    final_match = _FINAL_RE.search(stripped)

    # --- Tool call ---
    if tool_match:
        json_text = tool_match.group(1).strip()
        try:
            data = json.loads(json_text)
        except json.JSONDecodeError as exc:
            return ParseResult(
                kind="malformed",
                raw=raw,
                error=f"JSON parse failed: {exc} | Raw JSON: {json_text[:200]}",
            )

        try:
            tc = ToolCall.model_validate(data)
            # Eagerly validate args so we catch schema errors here
            tc.validated_args()
            return ParseResult(kind="tool_call", tool_call=tc, raw=raw)
        except (ValidationError, ValueError) as exc:
            return ParseResult(kind="malformed", raw=raw, error=str(exc))

    # --- Final answer ---
    if final_match:
        return ParseResult(
            kind="final_answer",
            final_answer=final_match.group(1).strip(),
            raw=raw,
        )

    # --- Graceful fallback: non-empty text without tags ---
    if stripped:
        return ParseResult(
            kind="final_answer",
            final_answer=stripped,
            raw=raw,
        )

    return ParseResult(kind="empty", raw=raw)
