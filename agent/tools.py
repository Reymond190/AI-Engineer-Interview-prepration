"""
agent/tools.py
--------------
Implementations of the three local tools available to the agent:

  1. calculator  – safe arithmetic evaluation (no exec/eval tricks)
  2. read_file   – read-only access to tests/fixtures/ directory only
  3. shell       – run an allowlisted read-only shell command

All tools return (result: str, error: str | None).
A non-None error means the tool failed; the harness records it and
feeds the error back to the model as an observation.
"""

from __future__ import annotations

import os
import subprocess
import ast
import operator
from pathlib import Path
from typing import Any

from agent.schema import (
    CalculatorArgs,
    ReadFileArgs,
    ShellArgs,
    SHELL_ALLOWLIST,
)

# Fixtures directory is resolved relative to this file's package root
_PROJECT_ROOT = Path(__file__).parent.parent.resolve()
FIXTURES_DIR  = _PROJECT_ROOT / "tests" / "fixtures"


# ---------------------------------------------------------------------------
# Calculator
# ---------------------------------------------------------------------------

# Whitelist of safe AST node types for the arithmetic evaluator
# Note: ast.Num was removed in Python 3.14; ast.Constant covers all literals.
_SAFE_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Constant,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Mod,
    ast.Pow,
    ast.FloorDiv,
    ast.UAdd,
    ast.USub,
)

_OPS: dict[type, Any] = {
    ast.Add:      operator.add,
    ast.Sub:      operator.sub,
    ast.Mult:     operator.mul,
    ast.Div:      operator.truediv,
    ast.Mod:      operator.mod,
    ast.Pow:      operator.pow,
    ast.FloorDiv: operator.floordiv,
    ast.UAdd:     operator.pos,
    ast.USub:     operator.neg,
}


def _eval_ast(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_ast(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp):
        left  = _eval_ast(node.left)
        right = _eval_ast(node.right)
        op_fn = _OPS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        return op_fn(left, right)
    if isinstance(node, ast.UnaryOp):
        operand = _eval_ast(node.operand)
        op_fn   = _OPS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        return op_fn(operand)
    raise ValueError(f"Unsafe AST node: {type(node).__name__}")


def calculator(args: CalculatorArgs) -> tuple[str, str | None]:
    """Evaluate a safe arithmetic expression and return the result as a string."""
    expr = args.expression.strip()
    try:
        tree = ast.parse(expr, mode="eval")
        # Validate all nodes are in the safe whitelist
        for node in ast.walk(tree):
            if not isinstance(node, _SAFE_NODES):
                return "", f"Unsafe expression node: {type(node).__name__}"
        result = _eval_ast(tree)
        # Format: drop trailing .0 for whole numbers
        if result == int(result):
            return str(int(result)), None
        return str(round(result, 10)), None
    except Exception as exc:  # noqa: BLE001
        return "", f"Calculator error: {exc}"


# ---------------------------------------------------------------------------
# File reader
# ---------------------------------------------------------------------------

def read_file(args: ReadFileArgs) -> tuple[str, str | None]:
    """
    Read a file from the fixtures directory.

    The path is strictly contained; no directory traversal is possible.
    """
    target = (FIXTURES_DIR / args.filename).resolve()

    # Double-check containment even after pydantic validation
    try:
        target.relative_to(FIXTURES_DIR)
    except ValueError:
        return "", f"Access denied: {args.filename!r} is outside fixtures directory."

    if not target.exists():
        available = [p.name for p in FIXTURES_DIR.glob("*") if p.is_file()]
        return "", (
            f"File not found: {args.filename!r}. "
            f"Available files: {available}"
        )

    try:
        content = target.read_text(encoding="utf-8")
        return content, None
    except Exception as exc:  # noqa: BLE001
        return "", f"Read error: {exc}"


# ---------------------------------------------------------------------------
# Shell sandbox
# ---------------------------------------------------------------------------

def shell(args: ShellArgs) -> tuple[str, str | None]:
    """
    Run an allowlisted read-only shell command.

    Allowed commands: ls, pwd, date, echo, uname, hostname, whoami, df, free.
    Pipes, semicolons, and other shell metacharacters are not interpreted
    (argv is split and passed directly to the kernel, not a shell).
    """
    parts = args.command.strip().split()
    if not parts:
        return "", "Empty command."

    base = parts[0]
    if base not in SHELL_ALLOWLIST:
        return "", (
            f"Command {base!r} is not in the allowlist. "
            f"Allowed: {sorted(SHELL_ALLOWLIST)}"
        )

    # Reject any shell metacharacters by ensuring all parts are plain tokens
    bad_chars = set("|;&$`><\\")
    for part in parts:
        if any(c in part for c in bad_chars):
            return "", f"Shell metacharacters not allowed: {part!r}"

    try:
        result = subprocess.run(
            parts,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if result.returncode != 0:
            return result.stdout, f"Command exited {result.returncode}: {result.stderr.strip()}"
        return result.stdout.strip(), None
    except subprocess.TimeoutExpired:
        return "", "Command timed out (10 s limit)."
    except FileNotFoundError:
        return "", f"Command not found: {base!r}"
    except Exception as exc:  # noqa: BLE001
        return "", f"Shell error: {exc}"


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

from agent.schema import ToolName, ToolCall  # noqa: E402 (avoid circular at module level)


def dispatch_tool(tc: ToolCall) -> tuple[str, str | None]:
    """Validate args and call the right tool implementation."""
    validated = tc.validated_args()

    if tc.action == ToolName.CALCULATOR:
        return calculator(validated)  # type: ignore[arg-type]
    if tc.action == ToolName.READ_FILE:
        return read_file(validated)  # type: ignore[arg-type]
    if tc.action == ToolName.SHELL:
        return shell(validated)      # type: ignore[arg-type]

    return "", f"Unknown tool: {tc.action}"
