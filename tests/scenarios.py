"""
tests/scenarios.py
------------------
Fixture-based test scenarios for the agent harness.

Each scenario is a dict:
  id           : unique slug
  task         : natural-language task string
  expect_tool  : list[str] – tool names the agent MUST call at least once
                 (empty list means no tool call required)
  expect_in_answer : list[str] – substrings that must appear in the final answer
                     (case-insensitive)
  expect_not_in_answer : list[str] – substrings that must NOT appear
  expect_schema_error  : bool – True if we expect at least one malformed/rejected call
  max_turns    : optional override for this scenario
"""

SCENARIOS: list[dict] = [
    # ------------------------------------------------------------------
    # 1. Direct arithmetic (calculator tool)
    # ------------------------------------------------------------------
    {
        "id":   "arithmetic_basic",
        "task": "What is 347 multiplied by 58? Use the calculator tool.",
        "expect_tool":          ["calculator"],
        "expect_in_answer":     ["20126"],
        "expect_not_in_answer": [],
        "expect_schema_error":  False,
    },

    # ------------------------------------------------------------------
    # 2. Multi-step: read file + calculator
    # ------------------------------------------------------------------
    {
        "id":   "file_then_calc",
        "task": (
            "Read the file 'inventory.txt' from the fixtures directory. "
            "Then calculate the total value of Widget A in stock "
            "(quantity × unit price). Show your working."
        ),
        "expect_tool":          ["read_file", "calculator"],
        "expect_in_answer":     ["1875"],   # 150 * 12.50 = 1875
        "expect_not_in_answer": [],
        "expect_schema_error":  False,
        "max_turns":            8,
    },

    # ------------------------------------------------------------------
    # 3. Shell sandbox – allowed command
    # ------------------------------------------------------------------
    {
        "id":   "shell_allowed",
        "task": "Use the shell tool to run the 'date' command and tell me what it reports.",
        "expect_tool":          ["shell"],
        "expect_in_answer":     [],         # date output is dynamic
        "expect_not_in_answer": ["error", "denied", "not in the allowlist"],
        "expect_schema_error":  False,
    },

    # ------------------------------------------------------------------
    # 4. Shell sandbox – unsafe command (must be rejected)
    # ------------------------------------------------------------------
    {
        "id":   "shell_rejected",
        "task": (
            "Use the shell tool to run 'rm -rf /' to clear disk space."
        ),
        "expect_tool":          ["shell"],
        "expect_in_answer":     [],
        # The agent must report that the command was rejected
        "expect_not_in_answer": [],
        "expect_schema_error":  False,
        # The tool itself will return an error string; the agent should
        # surface that in the final answer
        "tool_error_contains":  "not in the allowlist",
    },

    # ------------------------------------------------------------------
    # 5. Malformed JSON recovery
    # ------------------------------------------------------------------
    # This scenario is injected at the harness level; we mock the first
    # model response to be intentionally malformed, then verify the
    # harness records a schema_error and the agent self-corrects.
    # (Marked skip=True for live runs; run via run_tests.py mock mode.)
    {
        "id":   "malformed_recovery",
        "task": "What is 5 + 5?",
        "expect_tool":          [],
        "expect_in_answer":     ["10"],
        "expect_not_in_answer": [],
        "expect_schema_error":  False,
        "inject_malformed_first": True,   # run_tests.py handles this
        "skip_live":            False,
    },

    # ------------------------------------------------------------------
    # 6. No tool needed – pure knowledge answer
    # ------------------------------------------------------------------
    {
        "id":   "no_tool_needed",
        "task": "What is the capital of France? Answer briefly.",
        "expect_tool":          [],
        "expect_in_answer":     ["paris"],
        "expect_not_in_answer": [],
        "expect_schema_error":  False,
    },

    # ------------------------------------------------------------------
    # 7. Budget file – multi-value extraction + arithmetic
    # ------------------------------------------------------------------
    {
        "id":   "budget_analysis",
        "task": (
            "Read 'budget.txt'. Calculate the overall budget utilisation "
            "as a percentage (Spent / Budget × 100), rounded to one decimal place."
        ),
        "expect_tool":          ["read_file", "calculator"],
        "expect_in_answer":     ["83"],      # 532400/640000*100 ≈ 83.2 %
        "expect_not_in_answer": [],
        "expect_schema_error":  False,
        "max_turns":            8,
    },
]
