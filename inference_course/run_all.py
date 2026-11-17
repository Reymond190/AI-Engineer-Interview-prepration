"""
run_all.py - run the inference optimization lessons and check their OUTCOMES.

Same design as ml_course/run_all.py: lessons print `[tag] key=value` lines,
parsed into {tag: {key: value}}; each check is a predicate on the EFFECT the
lesson teaches (batching is nearly free in decode, roofline is a ceiling, ...),
not on exact numbers, so the checks survive driver/vLLM upgrades.

Lessons need different machine states, so run them in groups:
    vLLM STOPPED : 00 (GPU microbenchmarks; writes outputs/gpu.json), 01 (math, no GPU)
    vLLM UP      : 02 (serving benchmark)
A serving lesson with no server prints `[skip]`, and its checks show as skipped
instead of failed.

Run in WSL, env `ft`, from the project root:
    python inference_course/run_all.py 00 01           # vLLM stopped
    python inference_course/run_all.py 02              # after `bash server/serve.sh`
    python inference_course/run_all.py --full --save results_infer.json
Full output of each lesson: inference_course/outputs/logs/<lesson>.log
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOGS = HERE / "outputs" / "logs"

R = dict[str, dict[str, object]]
Check = tuple[str, Callable[[R], bool]]

CHECKS: dict[str, list[Check]] = {
    "00": [("measured read bandwidth within 75-100% of spec", lambda r: 0.75 < r["bandwidth"]["frac_of_spec"] <= 1.0),
           ("read-only beats copy (copy also writes)", lambda r: r["bandwidth"]["read_gbps"] > r["bandwidth"]["copy_gbps"]),
           ("tensor cores: bf16 > 2x fp32", lambda r: r["matmul"]["tflops_bf16"] > 2 * r["matmul"]["tflops_fp32"]),
           ("batching ~free below the ridge: M=64 < 1.3x M=1", lambda r: r["batch-sweep"]["ms_m64"] < 1.3 * r["batch-sweep"]["ms_m1"]),
           ("M=1 GEMV is bandwidth-bound (>60% of read BW)",
            lambda r: r["batch-sweep"]["gbps_m1"] > 0.6 * r["bandwidth"]["read_gbps"]),
           ("large M reaches >80% of peak matmul", lambda r: r["batch-sweep"]["tflops_m4096"] > 0.8 * r["matmul"]["tflops_bf16"]),
           ("knee within 4x of the roofline ridge",
            lambda r: r["batch-sweep"]["ridge"] / 4 <= r["batch-sweep"]["knee_m"] <= r["batch-sweep"]["ridge"] * 4),
           ("fusion speeds up elementwise > 1.5x", lambda r: r["fusion"]["speedup"] > 1.5)],
    "01": [("BF16 lm_head is >20% of qwen35 decode bytes", lambda r: r["decode-qwen35"]["lm_head_share"] > 0.2),
           ("batch-1 decode is memory-bound (>20x)", lambda r: r["decode-qwen35"]["mem_over_comp"] > 20),
           ("hybrid KV/token < 1/4 of Qwen3-8B", lambda r: r["kv-qwen35"]["kib_per_tok"] < r["kv-qwen3-8b"]["kib_per_tok"] / 4),
           ("GQA: Qwen3-8B KV = MHA / 4", lambda r: r["kv-qwen3-8b"]["mha_kib_per_tok"] == 4 * r["kv-qwen3-8b"]["kib_per_tok"]),
           ("short ctx: batch 4 > 3.5x batch 1", lambda r: r["batch-qwen3-8b-c1024-b4"]["tok_s"] > 3.5 * r["batch-qwen3-8b-c1024-b1"]["tok_s"]),
           ("32k ctx, batch 4: KV reads > weight reads (Qwen3-8B)", lambda r: r["batch-qwen3-8b-c32768-b4"]["kv_over_weights"] > 1),
           ("32k ctx: batching gains < 2x (Qwen3-8B)",
            lambda r: r["batch-qwen3-8b-c32768-b4"]["tok_s"] < 2 * r["batch-qwen3-8b-c32768-b1"]["tok_s"]),
           ("prefill pushes >20 tokens per decode-step time", lambda r: r["prefill-qwen35"]["tokens_per_decode_step_time"] > 20)],
    "02": [("roofline is a ceiling (measured <= 1.05x)", lambda r: r["single"]["efficiency"] <= 1.05),
           ("serving reaches >50% of the decode roofline", lambda r: r["single"]["efficiency"] > 0.5),
           ("TTFT grows with prompt (4k > 3x 128)", lambda r: r["prefill"]["ttft_4k"] > 3 * r["prefill"]["ttft_128"]),
           ("continuous batching: 4 streams > 2.5x throughput", lambda r: r["concurrency"]["tok_s_c4"] > 2.5 * r["concurrency"]["tok_s_c1"]),
           ("TPOT barely moves at 4 streams (<1.5x)", lambda r: r["concurrency"]["tpot_c4"] < 1.5 * r["concurrency"]["tpot_c1"]),
           ("past max-num-seqs: requests queue (TTFT p50 c8 > 2x c4)",
            lambda r: r["concurrency"]["ttft_p50_c8"] > 2 * r["concurrency"]["ttft_p50_c4"]),
           ("past max-num-seqs: throughput stops growing (c8 < 1.3x c4)",
            lambda r: r["concurrency"]["tok_s_c8"] < 1.3 * r["concurrency"]["tok_s_c4"])],
}

KV_LINE = re.compile(r"^\[([\w.-]+)\] (.*)$", re.MULTILINE)


def parse_value(s: str) -> object:
    if s in ("True", "False"):
        return s == "True"
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        return s


def parse(out: str) -> R:
    return {tag: {k: parse_value(v) for k, v in (p.split("=", 1) for p in rest.split() if "=" in p)}
            for tag, rest in KV_LINE.findall(out)}


def lessons(selected: list[str]) -> list[Path]:
    files = sorted(HERE.glob("[0-9][0-9]_*.py"))
    return [p for p in files if any(p.name.startswith(s + "_") for s in selected)] if selected else files


def run_lesson(path: Path, quick: bool, timeout: int) -> dict:
    key = path.name.split("_")[0]
    t0 = time.perf_counter()
    cmd = [sys.executable, str(path)] + (["--quick"] if quick else [])
    try:
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True, encoding="utf-8", errors="replace",
                              timeout=timeout, stdin=subprocess.DEVNULL, cwd=HERE)
        out, code = proc.stdout + "\n" + proc.stderr, proc.returncode
    except subprocess.TimeoutExpired as e:
        out, code = ((e.stdout or "") if isinstance(e.stdout, str) else "") + "\nTIMEOUT", -1
    LOGS.mkdir(parents=True, exist_ok=True)
    (LOGS / f"{path.stem}.log").write_text(out, encoding="utf-8")
    parsed = parse(out)
    skipped = "skip" in parsed
    checks: dict[str, bool | None] = {}
    for desc, pred in CHECKS.get(key, []):
        if skipped:
            checks[desc] = None
            continue
        try:
            checks[desc] = bool(pred(parsed))
        except (KeyError, TypeError):   # tag missing (crash) or wrong type
            checks[desc] = False
    return {"lesson": path.stem, "exit": code, "seconds": round(time.perf_counter() - t0, 1), "checks": checks,
            "skipped": skipped, "results": parsed, "tail": out[-800:] if code else ""}


def print_table(runs: list[dict]) -> None:
    print(f"\n{'lesson':<36} {'exit':>5} {'checks':>7} {'seconds':>8}")
    tp = tt = 0
    for r in runs:
        vals = [v for v in r["checks"].values() if v is not None]
        p, t = sum(vals), len(vals)
        tp, tt = tp + p, tt + t
        note = "  (skipped: needs vLLM up)" if r["skipped"] else ""
        print(f"{r['lesson']:<36} {r['exit']:>5} {p:>3}/{t:<3} {r['seconds']:>8.0f}{note}")
    print(f"{'TOTAL':<36} {'':>5} {tp:>3}/{tt:<3} {sum(r['seconds'] for r in runs):>8.0f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("lessons", nargs="*", help="lesson prefixes, e.g. 00 01 (default: all)")
    ap.add_argument("--full", action="store_true", help="full-length runs instead of --quick")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--save", help="write results JSON here")
    args = ap.parse_args()

    runs = []
    for path in lessons(args.lessons):
        r = run_lesson(path, quick=not args.full, timeout=args.timeout)
        failed = [k for k, v in r["checks"].items() if v is False]
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
