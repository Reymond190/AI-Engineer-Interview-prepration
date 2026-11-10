"""
run_all.py - run the fine-tuning lessons in order and check their OUTCOMES.

Same idea as langgraph_course/run_all.py, but checks can compare numbers
captured from the output (accuracy after > before, Liger peak < baseline...),
not just match text. Lessons run with --quick by default (minutes each), in
order, because later lessons load adapters saved by earlier ones
(05 -> 07/09/13/17, 06 -> 12, 11 -> 17).

Run in WSL, env `ft`, with vLLM stopped:
    python finetune_course/run_all.py                  # all (except 08), quick
    python finetune_course/run_all.py 05 07            # selected lessons
    python finetune_course/run_all.py --full --save results_ft.json
    python finetune_course/run_all.py --compare a.json b.json

Full output of every lesson is kept in finetune_course/outputs/logs/<lesson>.log.
Lesson 08 is skipped unless named: it runs on Windows against the vLLM server.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from collections.abc import Callable

HERE = Path(__file__).resolve().parent
LOGS = HERE / "outputs" / "logs"
F = r"([\d.]+)"

Check = tuple[str, str, Callable[..., bool] | None]  # (description, regex, predicate on float captures)


def gt(margin: float = 0.0) -> Callable[[float, float], bool]:
    return lambda a, b: b > a + margin


# lesson prefix -> checks. A regex with captures is passed (as floats) to the predicate.
CHECKS: dict[str, list[Check]] = {
    "00": [("16 B/param rule within 25%", rf"ratio={F}", lambda r: 0.9 < r < 1.25)],
    "01": [("prompt masked, EOS trained", r"\[eos in labels\] True", None)],
    "02": [("accuracy improves", rf"\[result\] accuracy {F} -> {F}", gt(0.1)),
           ("manual loss == HF loss", r"manual=(\S+)\s+hf=\1", None)],
    "03": [("full FT improves", rf"\[result\] accuracy {F} -> {F}", gt(0.1)),
           ("overfitting: final val > best val", rf"\[overfit\] best val_loss {F} at step \d+, final {F}", gt())],
    "04": [("hand LoRA == PEFT param count", r"match=True", None),
           ("hand LoRA learns", rf"\[eval hand LoRA\] accuracy={F}", lambda a: a > 0.3)],
    "05": [("SFT LoRA improves", rf"\[result\] accuracy {F} -> {F}", gt(0.2))],
    "06": [("QLoRA 4B improves", rf"\[result\] accuracy {F} -> {F}", gt(0.1)),
           ("NF4 beats FP4 error", rf"nf4: rmse={F}[\s\S]*fp4: rmse={F}", gt())],
    "07": [("LoRA beats base", rf"\[eval base test\] accuracy={F}[\s\S]*\[eval lora test\] accuracy={F}", gt()),
           ("forgetting probe ran", r"\[forgetting lora\] generic_ppl", None)],
    "09": [("DPO prefers chosen", rf"\[dpo\] reward_acc={F}", lambda a: a > 0.5)],
    "10": [("GRPO reward rises", rf"\[grpo\] reward first={F} last={F}", gt())],
    "11": [("tool calling improves", rf"\[result\] tool_exact {F} -> {F}", gt(0.1))],
    "12": [("KD student trained", rf"\[eval logit-KD\] accuracy={F}", lambda a: a > 0.2),
           ("seq-KD student trained", rf"\[eval seq-KD\] accuracy={F}", lambda a: a > 0.2)],
    "13": [("merged checkpoint works", rf"\[export\] merged accuracy={F}", lambda a: a > 0.3)],
    "14": [("Liger lowers peak VRAM", rf"\[liger\] peak {F} -> {F} GB", lambda a, b: b < a)],
    "15": [("all variants ran", r"\[variant\] \{'variant': '\(IA\)\^3'", None)],
    "16": [("GaLore ran", r"\[frontier\] \{'method': 'GaLore[^\n]*accuracy", None),
           ("Spectrum ran", r"\[frontier\] \{'method': 'Spectrum[^\n]*accuracy", None)],
    "17": [("cat-merge keeps both skills", rf"\[merge cat \(exact sum of BA\)\] tickets={F} tools={F}",
            lambda t, c: min(t, c) > 0.3)],
    "18": [("all RL variants ran", r"\[rl-variant GSPO\]", None)],
}


def lessons(selected: list[str]) -> list[Path]:
    files = sorted(HERE.glob("[0-9][0-9]_*.py"))
    if selected:
        return [p for p in files if any(p.name.startswith(s + "_") for s in selected)]
    return [p for p in files if not p.name.startswith("08_")]


def check(out: str, desc: str, rx: str, pred) -> bool:
    m = re.search(rx, out)
    if not m:
        return False
    if pred is None:
        return True
    try:
        return bool(pred(*(float(g) for g in m.groups())))
    except ValueError:
        return False


def run_lesson(path: Path, quick: bool, timeout: int) -> dict:
    key = path.name.split("_")[0]
    t0 = time.perf_counter()
    cmd = [sys.executable, str(path)] + (["--quick"] if quick else [])
    try:
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True, encoding="utf-8", errors="replace",
                              timeout=timeout, stdin=subprocess.DEVNULL)
        out, code = proc.stdout + "\n" + proc.stderr, proc.returncode
    except subprocess.TimeoutExpired as e:
        out, code = (e.stdout or "") if isinstance(e.stdout, str) else "", -1
        out += "\nTIMEOUT"
    LOGS.mkdir(parents=True, exist_ok=True)
    (LOGS / f"{path.stem}.log").write_text(out, encoding="utf-8")
    results = {desc: check(out, desc, rx, pred) for desc, rx, pred in CHECKS.get(key, [])}
    return {"lesson": path.stem, "exit": code, "seconds": round(time.perf_counter() - t0, 1), "checks": results,
            "tail": out[-800:] if code else ""}


def print_table(runs: list[dict]) -> None:
    print(f"\n{'lesson':<36} {'exit':>5} {'checks':>7} {'seconds':>8}")
    tp = tt = 0
    for r in runs:
        p, t = sum(r["checks"].values()), len(r["checks"])
        tp, tt = tp + p, tt + t
        print(f"{r['lesson']:<36} {r['exit']:>5} {p:>3}/{t:<3} {r['seconds']:>8.0f}")
    print(f"{'TOTAL':<36} {'':>5} {tp:>3}/{tt:<3} {sum(r['seconds'] for r in runs):>8.0f}")


def compare(a_path: str, b_path: str) -> None:
    a, b = (json.loads(Path(p).read_text()) for p in (a_path, b_path))
    bl = {r["lesson"]: r for r in b["runs"]}
    for r in a["runs"]:
        o = bl.get(r["lesson"])
        cell = lambda x: f"{sum(x['checks'].values())}/{len(x['checks'])} {x['seconds']:5.0f}s" if x else "-"  # noqa: E731
        print(f"{r['lesson']:<36} {cell(r):>14} {cell(o):>14}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("lessons", nargs="*", help="lesson prefixes, e.g. 05 07 (default: all except 08)")
    ap.add_argument("--full", action="store_true", help="full-length runs instead of --quick")
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--save", help="write results JSON here")
    ap.add_argument("--compare", nargs=2, metavar=("A.json", "B.json"))
    args = ap.parse_args()
    if args.compare:
        compare(*args.compare)
        return

    runs = []
    for path in lessons(args.lessons):
        r = run_lesson(path, quick=not args.full, timeout=args.timeout)
        failed = [k for k, v in r["checks"].items() if not v]
        print(f"{r['lesson']:<36} exit={r['exit']} {r['seconds']:>6.0f}s  failed={failed or '-'}", flush=True)
        if r["exit"]:
            print("  " + r["tail"].replace("\n", "\n  "))
        runs.append(r)
    print_table(runs)
    if args.save:
        Path(args.save).write_text(json.dumps({"mode": "full" if args.full else "quick", "runs": runs}, indent=2))
        print(f"saved {args.save}")


if __name__ == "__main__":
    main()
