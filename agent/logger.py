"""
agent/logger.py
---------------
Structured NDJSON logger for agent runs.

Each line in the log file is a self-contained JSON object with:
  - ts         : ISO-8601 timestamp
  - run_id     : UUID for the agent session
  - turn       : conversation turn number (0-based)
  - event      : one of init | model_response | tool_call | tool_result | final_answer | error | summary
  - <payload>  : event-specific fields

Usage:
    logger = AgentLogger(run_id="abc123", log_dir=Path("logs"))
    logger.log_model_response(turn=0, content="...", latency_s=1.2)
    logger.log_tool_call(turn=0, action="calculator", args={"expression":"2+2"})
    logger.log_tool_result(turn=0, action="calculator", result="4", error=None)
    logger.log_final_answer(turn=1, content="The answer is 4.", total_turns=2)
    logger.close()
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class AgentLogger:
    def __init__(self, run_id: str, log_dir: Path | None = None, echo: bool = True):
        """
        Parameters
        ----------
        run_id  : Unique identifier for this agent run.
        log_dir : Directory to write the NDJSON log file.
                  Defaults to <project_root>/logs/.
        echo    : If True, also print a short human-readable summary to stderr.
        """
        self.run_id  = run_id
        self.echo    = echo

        if log_dir is None:
            log_dir = Path(__file__).parent.parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        self._path   = log_dir / f"{run_id}.ndjson"
        self._file   = self._path.open("a", encoding="utf-8")

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _now(self) -> str:
        return datetime.now(tz=timezone.utc).isoformat()

    def _write(self, event: str, turn: int | None, payload: dict[str, Any]) -> None:
        record: dict[str, Any] = {
            "ts":     self._now(),
            "run_id": self.run_id,
            "event":  event,
        }
        if turn is not None:
            record["turn"] = turn
        record.update(payload)
        line = json.dumps(record, ensure_ascii=False)
        self._file.write(line + "\n")
        self._file.flush()

    def _echo(self, msg: str) -> None:
        if self.echo:
            print(msg, file=sys.stderr)

    # ------------------------------------------------------------------ #
    # Public logging methods                                               #
    # ------------------------------------------------------------------ #

    def log_init(self, task: str, model: str, base_url: str) -> None:
        self._write("init", None, {"task": task, "model": model, "base_url": base_url})
        self._echo(f"\n[run:{self.run_id[:8]}] Task: {task[:120]}")

    def log_model_response(self, turn: int, content: str, latency_s: float,
                           input_tokens: int = 0, output_tokens: int = 0) -> None:
        self._write("model_response", turn, {
            "content_preview": content[:300],
            "latency_s":       round(latency_s, 3),
            "input_tokens":    input_tokens,
            "output_tokens":   output_tokens,
        })
        self._echo(f"  [t{turn}] model ({latency_s:.1f}s, {output_tokens} tok): {content[:120]!r}")

    def log_tool_call(self, turn: int, action: str, args: dict[str, Any]) -> None:
        self._write("tool_call", turn, {"action": action, "args": args})
        self._echo(f"  [t{turn}] → tool: {action}({args})")

    def log_tool_result(self, turn: int, action: str, result: str, error: str | None) -> None:
        self._write("tool_result", turn, {
            "action":         action,
            "result_preview": result[:300],
            "error":          error,
        })
        if error:
            self._echo(f"  [t{turn}] ✗ {action} error: {error}")
        else:
            self._echo(f"  [t{turn}] ✓ {action}: {result[:80]!r}")

    def log_malformed(self, turn: int, raw: str, error: str) -> None:
        self._write("malformed", turn, {"raw_preview": raw[:300], "error": error})
        self._echo(f"  [t{turn}] ⚠ malformed output: {error}")

    def log_final_answer(self, turn: int, content: str, total_turns: int,
                         total_latency_s: float) -> None:
        self._write("final_answer", turn, {
            "content":         content,
            "total_turns":     total_turns,
            "total_latency_s": round(total_latency_s, 3),
        })
        self._echo(f"\n  [DONE] Final answer ({total_turns} turns, {total_latency_s:.1f}s):\n  {content[:300]}")

    def log_error(self, turn: int | None, error: str) -> None:
        self._write("error", turn, {"error": error})
        self._echo(f"  [ERROR] {error}")

    def log_summary(self, stats: dict[str, Any]) -> None:
        self._write("summary", None, stats)
        self._echo(f"\n  [SUMMARY] {stats}")

    def close(self) -> None:
        self._file.close()

    def path(self) -> Path:
        return self._path
