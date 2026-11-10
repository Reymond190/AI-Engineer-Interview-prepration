"""
tests/test_schema.py
--------------------
Unit tests for agent/schema.py (no model server needed).
Run with: pytest tests/test_schema.py -v
"""

import pytest
from agent.schema import parse_model_output, strip_thinking, ToolName


class TestStripThinking:
    def test_removes_think_block(self):
        raw = "<think>I need to add these.</think>\nHello."
        assert strip_thinking(raw) == "Hello."

    def test_removes_multiline_think(self):
        raw = "<think>\nLine 1\nLine 2\n</think>\nResult"
        assert strip_thinking(raw) == "Result"

    def test_no_think_block(self):
        raw = "Just text."
        assert strip_thinking(raw) == "Just text."

    def test_case_insensitive(self):
        raw = "<THINK>hidden</THINK>\nVisible"
        assert "hidden" not in strip_thinking(raw)


class TestParseToolCall:
    def test_valid_calculator(self):
        raw = '<tool_call>\n{"action": "calculator", "args": {"expression": "3 + 4"}}\n</tool_call>'
        result = parse_model_output(raw)
        assert result.kind == "tool_call"
        assert result.tool_call is not None
        assert result.tool_call.action == ToolName.CALCULATOR
        assert result.tool_call.args["expression"] == "3 + 4"

    def test_valid_read_file(self):
        raw = '<tool_call>{"action": "read_file", "args": {"filename": "inventory.txt"}}</tool_call>'
        result = parse_model_output(raw)
        assert result.kind == "tool_call"
        assert result.tool_call.action == ToolName.READ_FILE

    def test_valid_shell(self):
        raw = '<tool_call>{"action": "shell", "args": {"command": "date"}}</tool_call>'
        result = parse_model_output(raw)
        assert result.kind == "tool_call"
        assert result.tool_call.action == ToolName.SHELL

    def test_think_stripped_before_parse(self):
        raw = "<think>Let me use calculator.</think><tool_call>{\"action\":\"calculator\",\"args\":{\"expression\":\"1+1\"}}</tool_call>"
        result = parse_model_output(raw)
        assert result.kind == "tool_call"

    def test_invalid_json(self):
        raw = "<tool_call>NOT_JSON</tool_call>"
        result = parse_model_output(raw)
        assert result.kind == "malformed"
        assert "JSON" in (result.error or "")

    def test_unknown_tool_name(self):
        raw = '<tool_call>{"action": "hack_system", "args": {}}</tool_call>'
        result = parse_model_output(raw)
        assert result.kind == "malformed"

    def test_calculator_dangerous_chars(self):
        raw = '<tool_call>{"action": "calculator", "args": {"expression": "__import__(\"os\")"}}</tool_call>'
        result = parse_model_output(raw)
        assert result.kind == "malformed"

    def test_read_file_traversal(self):
        raw = '<tool_call>{"action": "read_file", "args": {"filename": "../../../etc/passwd"}}</tool_call>'
        result = parse_model_output(raw)
        assert result.kind == "malformed"

    def test_shell_not_in_allowlist(self):
        raw = '<tool_call>{"action": "shell", "args": {"command": "rm -rf /"}}</tool_call>'
        result = parse_model_output(raw)
        assert result.kind == "malformed"
        assert "allowlist" in (result.error or "").lower()


class TestParseFinalAnswer:
    def test_explicit_tag(self):
        raw = "<final_answer>\nThe answer is 42.\n</final_answer>"
        result = parse_model_output(raw)
        assert result.kind == "final_answer"
        assert "42" in (result.final_answer or "")

    def test_fallback_no_tags(self):
        raw = "The capital of France is Paris."
        result = parse_model_output(raw)
        assert result.kind == "final_answer"
        assert "Paris" in (result.final_answer or "")

    def test_empty_output(self):
        result = parse_model_output("")
        assert result.kind == "empty"

    def test_only_think_block(self):
        raw = "<think>Nothing to say.</think>"
        result = parse_model_output(raw)
        # After stripping think block, content is empty
        assert result.kind in ("empty", "final_answer")
