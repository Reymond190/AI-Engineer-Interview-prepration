"""
run.py - launch a multi-agent run.

    python agent_lab/run.py --task pe_semiprimes                 # 4 solvers + verifier, 15 min
    python agent_lab/run.py --task pe_semiprimes --mode single   # 1 solver, no verifier (baseline)
    python agent_lab/run.py "Your own task text ..." --agents 3 --minutes 10
    python agent_lab/run.py --list

Everything is saved to agent_lab/runs/<timestamp>-<task>/:
    solver-*/        each agent's workspace (code in snippets/, data, outputs, charts)
    verifier-r*/     the verifier's workspace
    events.jsonl     every event; commands.log every sandbox command
    plan_round*.json, report.md, result.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from core import EventLog  # noqa: E402
from orchestrator import build_graph  # noqa: E402
from sandbox import IMAGE, wsl  # noqa: E402
from tasks import TASKS  # noqa: E402


def preflight() -> None:
    from common import BASE_URL, MODEL  # noqa: PLC0415
    import httpx  # noqa: PLC0415

    try:
        httpx.get(f"{BASE_URL}/models", timeout=3).raise_for_status()
    except Exception:  # noqa: BLE001
        sys.exit(f"vLLM server not reachable at {BASE_URL}. Start it in WSL: bash server/serve.sh")
    if wsl("docker", "image", "inspect", IMAGE).returncode != 0:
        sys.exit(f"Sandbox image {IMAGE} missing. Build it: docker build -t {IMAGE} agent_lab/sandbox")
    print(f"model: {MODEL} | sandbox image: {IMAGE}")


def cleanup_containers() -> None:
    ids = wsl("docker", "ps", "-aq", "--filter", "name=agentlab-").stdout.split()
    if ids:
        wsl("docker", "rm", "-f", *ids)
        print(f"removed {len(ids)} leftover container(s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt", nargs="?", help="task text (or use --task)")
    ap.add_argument("--task", help="task id from tasks.py")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--mode", choices=["team", "single"], default="team")
    ap.add_argument("--agents", type=int, default=4)
    ap.add_argument("--minutes", type=float, default=15)
    ap.add_argument("--rounds", type=int, default=2, help="max plan->solve->verify rounds")
    args = ap.parse_args()

    if args.list:
        for k, v in TASKS.items():
            print(f"{k:<32} {'[graded] ' if v['answer'] else ''}{v['prompt'][:90]}")
        sys.exit()
    spec = TASKS.get(args.task or "", {})
    prompt = spec.get("prompt") or args.prompt
    if not prompt:
        sys.exit("give a task text or --task <id> (see --list)")

    preflight()
    team = args.mode == "team"
    n = args.agents if team else 1
    slug = args.task or re.sub(r"[^a-z0-9]+", "-", prompt.lower())[:30].strip("-")
    run_dir = HERE / "runs" / f"{time.strftime('%Y%m%d-%H%M%S')}-{slug}-{args.mode}"
    run_dir.mkdir(parents=True)
    log = EventLog(run_dir)
    graph = build_graph(run_dir, log, verify=team)

    t0 = time.time()
    state = {"task": prompt, "n_solvers": n, "deadline": t0 + args.minutes * 60, "max_rounds": args.rounds if team else 1}
    print(f"run dir: {run_dir}\n")
    try:
        out = graph.invoke(state, {"max_concurrency": n, "recursion_limit": 50})
    except KeyboardInterrupt:
        print("\ninterrupted")
        cleanup_containers()
        sys.exit(1)
    finally:
        cleanup_containers()

    final = out.get("final", {})
    expected = spec.get("answer")
    result = {**final, "task": args.task or prompt, "mode": args.mode, "agents": n, "seconds": round(time.time() - t0),
              "expected": expected, "correct": (final.get("answer") == expected) if expected else None}
    (run_dir / "result.json").write_text(json.dumps(result, indent=2))
    print(f"\nANSWER: {final.get('answer')!r}  [{final.get('status')}, {final.get('support')}]  in {result['seconds']}s")
    if expected:
        print("CORRECT" if result["correct"] else f"WRONG (expected {expected})")
    print(f"report: {run_dir / 'report.md'}")
