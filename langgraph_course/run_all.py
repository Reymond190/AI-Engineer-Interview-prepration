"""
run_all.py - run every lesson and score it, to compare models / prompts.

Each lesson is checked on OUTCOMES (did the agent get the right number, take
the right action, route correctly), not just "did it exit 0". This is the same
idea as lesson 21's evals, applied to the whole course.

Usage:
    python langgraph_course/run_all.py                    # all lessons once
    python langgraph_course/run_all.py --repeat 3 05 06   # flaky ones, 3x each
    python langgraph_course/run_all.py --save results_qwen3.json
    python langgraph_course/run_all.py --compare results_a.json results_b.json
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

# lesson prefix -> [(check description, regex that must match the output)]
CHECKS: dict[str, list[tuple[str, str]]] = {
    "01": [("produces a summary", r"\[summary\] \S")],
    "02": [("billing routed", r'"route": "billing"'), ("technical routed", r'"route": "technical"')],
    "03": [
        ("gadget value 699.65", r"699\.65"),
        ("3 parallel lookups", r"tool_calls=\[\('lookup_product'.*\('lookup_product'.*\('lookup_product'"),
        ("looks up unknown product", r"Unknown product 'sprocket"),
        ("create_agent widget value 510", r"\b510(\.00?)?\b"),
    ],
    "03b": [("emulated tools: 87.8F", r"87\.8"), ("prompt-JSON parsed", r'"temp_c": 14')],
    "04": [("invoice total fixed to 644", r'"total": 644\.0'), ("bat & ball = 5", r'"answer": 5\b')],
    "05": [
        ("researcher used", r"supervisor round \d\] researcher"),
        ("analyst used", r"supervisor round \d\] analyst"),
        ("1,440 in final", r"Final answer[\s\S]*1,?440"),
    ],
    "06": [("flight moved", r'"date": "2026-11-05"'), ("hotel 5 nights", r'"hotel": "Hilton Midtown, 5 nights"')],
    "07": [("article written", r"\[final article\] (?!\(skipped\))\S")],
    "08": [("parallel speed-up", r"fan-out phase\] ([\d.]+)s wall clock vs ([\d.]+)s")],
    "09": [("thread memory (Rust)", r"Rust"), ("fork applied", r"forked premise\] A lighthouse keeper")],
    "10": [("approved refund", r"REFUNDED \$120\.00"), ("edited refund", r"REFUNDED \$60\.00"), ("rejection fed back", r"Rejected by human reviewer")],
    "11": [("summary keeps April", r"\[summary\].*April"), ("long-term recall Python", r"Which language[\s\S]*Python")],
    "12": [("custom events streamed", r"\[custom\] \{'progress': 'retrieved 3/3'")],
    "13": [("all tests pass", r"best score\] 8/8")],
    "14": [("correct total 7068.6", r"Answer[\s\S]*7,?068\.6")],
    "15": [("published after resume", r"status\] published")],
    "16": [("cache hit", r"call 2 \(0\.0\ds\)"), ("fallback model", r"fallback works")],
    "17": [
        ("routes: docs, docs, direct, docs", r"company_docs[\s\S]*company_docs[\s\S]*\] direct[\s\S]*company_docs"),
        ("Canberra", r"Canberra"),
        ("abstains on unknown", r"couldn't find this in the company documents"),
    ],
    "18": [("judge verdict", r'"winner": "(optimist|skeptic)"')],
    "19": [("assigned PLAT-102 to sam", r"assign_ticket', \{'ticket_id': 'PLAT-102', 'owner': 'sam'")],
    "20": [("planned with todos", r"\[todos\]"), ("used sub-agents", r"spawn sub-agent"), ("report.md written", r"write_file\] report\.md")],
    "21": [("judge accuracy 3/3", r"accuracy\] 3/3"), ("trajectories 3/3", r"superset match\] 3/3")],
}


def lessons(selected: list[str]) -> list[Path]:
    files = sorted(p for p in HERE.glob("[0-9]*.py"))
    if selected:
        files = [p for p in files if any(p.name.startswith(s + "_") for s in selected)]
    return files


def run_lesson(path: Path, timeout: int) -> dict:
    key = path.name.split("_")[0]
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            [sys.executable, str(path)], capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=timeout, stdin=subprocess.DEVNULL,
        )
        out, code = proc.stdout + proc.stderr, proc.returncode
    except subprocess.TimeoutExpired:
        out, code = "TIMEOUT", -1
    seconds = round(time.perf_counter() - t0, 1)
    results = {name: bool(re.search(rx, out)) for name, rx in CHECKS.get(key, [])}
    return {"lesson": path.stem, "exit": code, "seconds": seconds, "checks": results, "tail": out[-600:] if code else ""}


def summarise(runs: list[dict]) -> dict:
    by_lesson: dict[str, dict] = {}
    for r in runs:
        s = by_lesson.setdefault(r["lesson"], {"runs": 0, "ok": 0, "checks_passed": 0, "checks_total": 0, "seconds": 0.0})
        s["runs"] += 1
        s["ok"] += r["exit"] == 0
        s["checks_passed"] += sum(r["checks"].values())
        s["checks_total"] += len(r["checks"])
        s["seconds"] += r["seconds"]
    return by_lesson


def print_table(model: str, summary: dict) -> None:
    print(f"\n{'lesson':<45} {'exit ok':>8} {'checks':>8} {'avg s':>7}")
    tp = tt = 0
    total_s = 0.0
    for lesson, s in summary.items():
        tp, tt, total_s = tp + s["checks_passed"], tt + s["checks_total"], total_s + s["seconds"] / s["runs"]
        print(f"{lesson:<45} {s['ok']}/{s['runs']:<6} {s['checks_passed']:>3}/{s['checks_total']:<4} {s['seconds'] / s['runs']:>7.1f}")
    print(f"{'TOTAL (' + model + ')':<45} {'':>8} {tp:>3}/{tt:<4} {total_s:>7.1f}")


def compare(a_path: str, b_path: str) -> None:
    a, b = (json.loads(Path(p).read_text()) for p in (a_path, b_path))
    print(f"\n{'lesson':<45} {a['model'][:16]:>16} {b['model'][:16]:>16}")
    for lesson in a["summary"]:
        cells = []
        for d in (a, b):
            s = d["summary"].get(lesson)
            cells.append(f"{s['checks_passed']}/{s['checks_total']} {s['seconds'] / s['runs']:5.1f}s" if s else "-")
        print(f"{lesson:<45} {cells[0]:>16} {cells[1]:>16}")
    for d in (a, b):
        s = d["summary"].values()
        print(f"{d['model']}: checks {sum(x['checks_passed'] for x in s)}/{sum(x['checks_total'] for x in s)}, "
              f"time {sum(x['seconds'] / x['runs'] for x in s):.0f}s")


if __name__ == "__main__":
    sys.path.insert(0, str(HERE))
    ap = argparse.ArgumentParser()
    ap.add_argument("lessons", nargs="*", help="lesson prefixes, e.g. 05 06 (default: all)")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--save", help="write results JSON here")
    ap.add_argument("--compare", nargs=2, metavar=("A.json", "B.json"))
    args = ap.parse_args()

    if args.compare:
        compare(*args.compare)
        sys.exit(0)

    from common import MODEL  # noqa: E402

    runs = []
    for path in lessons(args.lessons):
        for i in range(args.repeat):
            r = run_lesson(path, args.timeout)
            failed = [k for k, v in r["checks"].items() if not v]
            print(f"{r['lesson']:<45} run {i + 1}: exit={r['exit']} {r['seconds']:>6.1f}s  failed={failed or '-'}", flush=True)
            runs.append(r)

    summary = summarise(runs)
    print_table(MODEL, summary)
    if args.save:
        Path(args.save).write_text(json.dumps({"model": MODEL, "runs": runs, "summary": summary}, indent=2))
        print(f"\nsaved {args.save}")
