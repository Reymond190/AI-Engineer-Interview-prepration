"""
tests/test_tools.py
-------------------
Unit tests for agent/tools.py (no model server needed).
Run with: pytest tests/test_tools.py -v

Note: Shell tests that invoke Linux commands (date, echo, uname) are
skipped on Windows; they pass in WSL/Linux where the agent runs in production.
"""

import sys
import pytest
from agent.schema import CalculatorArgs, ReadFileArgs, ShellArgs
from agent.tools import calculator, read_file, shell

linux_only = pytest.mark.skipif(sys.platform == "win32", reason="Linux shell commands only available in WSL")


class TestCalculator:
    def test_addition(self):
        result, err = calculator(CalculatorArgs(expression="3 + 4"))
        assert err is None
        assert result == "7"

    def test_multiplication(self):
        result, err = calculator(CalculatorArgs(expression="347 * 58"))
        assert err is None
        assert result == "20126"

    def test_division(self):
        result, err = calculator(CalculatorArgs(expression="150 * 12.50"))
        assert err is None
        assert result == "1875"

    def test_power(self):
        result, err = calculator(CalculatorArgs(expression="2 ** 10"))
        assert err is None
        assert result == "1024"

    def test_complex_expression(self):
        result, err = calculator(CalculatorArgs(expression="(100 + 200) * 3 / 9"))
        assert err is None
        assert result == "100"

    def test_float_result(self):
        result, err = calculator(CalculatorArgs(expression="532400 / 640000 * 100"))
        assert err is None
        # Should be approximately 83.1875
        assert "83" in result

    def test_dangerous_expression_rejected(self):
        with pytest.raises(Exception):
            CalculatorArgs(expression="__import__('os')")

    def test_letters_rejected(self):
        with pytest.raises(Exception):
            CalculatorArgs(expression="abc + 1")


class TestReadFile:
    def test_read_inventory(self):
        result, err = read_file(ReadFileArgs(filename="inventory.txt"))
        assert err is None
        assert "Widget A" in result
        assert "150" in result

    def test_read_budget(self):
        result, err = read_file(ReadFileArgs(filename="budget.txt"))
        assert err is None
        assert "Engineering" in result

    def test_file_not_found(self):
        result, err = read_file(ReadFileArgs(filename="nonexistent.txt"))
        assert result == ""
        assert "not found" in (err or "").lower()

    def test_path_traversal_rejected(self):
        with pytest.raises(Exception):
            ReadFileArgs(filename="../setup/02_setup_env.sh")

    def test_absolute_path_rejected(self):
        with pytest.raises(Exception):
            ReadFileArgs(filename="/etc/passwd")


class TestShell:
    @linux_only
    def test_date_command(self):
        result, err = shell(ShellArgs(command="date"))
        assert err is None
        assert len(result) > 0

    @linux_only
    def test_echo_command(self):
        result, err = shell(ShellArgs(command="echo hello"))
        assert err is None
        assert "hello" in result

    @linux_only
    def test_uname_command(self):
        result, err = shell(ShellArgs(command="uname"))
        assert err is None
        assert len(result) > 0

    def test_disallowed_rm(self):
        with pytest.raises(Exception):
            ShellArgs(command="rm -rf /")

    def test_disallowed_python(self):
        with pytest.raises(Exception):
            ShellArgs(command="python3 malicious.py")

    def test_disallowed_curl(self):
        with pytest.raises(Exception):
            ShellArgs(command="curl http://example.com")

    def test_metachar_pipe_rejected(self):
        result, err = shell(ShellArgs(command="date | cat"))
        # ShellArgs passes (| is in the command), but the tool itself
        # rejects metacharacters at execution time
        assert err is not None
        assert "metachar" in (err or "").lower()
