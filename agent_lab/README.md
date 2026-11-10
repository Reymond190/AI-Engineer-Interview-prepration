# agent_lab: parallel sandboxed agents for hard problems

A LangGraph orchestrator that splits a task across N solver agents, each in its
own Docker container with Python, bash and internet access; checks their work
with an independent verifier; and retries with feedback until an answer is
verified or the time budget runs out.

```
START -> plan -> [solver-1 .. solver-N in parallel, one container each] -> verify -> decide -> report
                  ^                                                                    |
                  `---------------- retry with verifier feedback ----------------------'
```

## Quick start

```bash
# WSL terminal: model server (Qwen3.5-9B)
bash server/serve.sh

# once: build the sandbox image (WSL)
docker build -t agent-lab-sandbox:latest agent_lab/sandbox

# Windows terminal, project root
.venv\Scripts\python.exe agent_lab\run.py --list
.venv\Scripts\python.exe agent_lab\run.py --task pe_semiprimes                  # 4 solvers + verifier, 15 min
.venv\Scripts\python.exe agent_lab\run.py --task pe_semiprimes --mode single     # baseline: 1 agent, no verifier
.venv\Scripts\python.exe agent_lab\run.py "Your own task ..." --agents 3 --minutes 10
```

Keep the MetaTrader 5 terminal open for forex tasks.

## What each piece does

| File | Role |
|---|---|
| `orchestrator.py` | LangGraph graph: planner (thinking mode) gives each solver a *different* strategy; `Send` fan-out; verifier; consensus + retry logic; report |
| `core.py` | Agent loop with wall-clock deadline, step cap, context compaction (~12k tokens per agent), terminal `submit` tool; sandbox tools; live event log |
| `sandbox.py` | One container per agent via `wsl docker`: non-root, `--cap-drop ALL`, CPU/RAM/pids limits, per-command timeouts, only its own workspace mounted, audit log |
| `fx_data.py` | Host-side, read-only MetaTrader 5 candles (`fetch_fx` tool). Windows-only package, so it never runs in the container |
| `tasks.py` | Benchmark tasks: math with known answers (auto-graded) and open-ended forex research |
| `sandbox/Dockerfile` | Python 3.12 + numpy/pandas/scipy/statsmodels/sklearn/sympy/numba/arch/matplotlib |

## Output: `agent_lab/runs/<time>-<task>-<mode>/`

- `report.md`: answer, verification status, each solver's method and the verifier's evidence (plus a synthesis for research tasks)
- `result.json`: answer, status, time, correct (for graded tasks)
- `solver-*/`, `verifier-r*/`: each agent's workspace (`snippets/NNN.py` = every script it ran, data, charts)
- `events.jsonl`: every event; `commands.log`: every sandbox command

## Why this architecture (and what to experiment with)

- **Diversity beats copies.** N copies of one prompt make correlated mistakes; the planner assigns different methods.
- **Verification beats voting.** The verifier re-runs and re-derives from the read-only candidate workspaces. Majority vote is only the fallback.
- **Small model, small contexts.** The GPU has room for about 56k KV-cache tokens across all agents, so each agent compacts old tool output and summarises its history.
- Ideas to try: more solvers vs more rounds; a thinking-mode verifier; tournament/evolution (keep the best program and mutate it); a shared "lessons" file between rounds; a frontier-model planner; per-agent CPU limits for heavy searches.

## Safety notes

Agents have full internet access (by choice) and run code you didn't write. The
container can't see your files except its workspace, has no credentials, and is
deleted after the run, but it *can* reach the network. Don't add secrets to tasks
or mounts, review `commands.log`, and don't point it at systems you don't own.
`fetch_fx` only exposes MT5 *read* functions (no orders); keep "Algo Trading"
disabled in the terminal anyway.
