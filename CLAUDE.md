# CLAUDE.md

Personal lab for running open LLMs locally (RTX 5080, 16 GB) behind vLLM and building/evaluating agents on top of them. No packaging (`pyproject.toml`), no CI.

Git: this folder is a checkout of `github.com/Reymond190/AI-Engineer-Interview-prepration` (`main`), shared with the user's interview-prep notes (the `*.md` files at the root besides `README.md`/`CLAUDE.md`). Commits go straight to `main`. `.gitignore` keeps out `.env`, `.venv/`, caches, and generated output (`finetune_course/outputs/`, `agent_lab/runs/`, `logs/`, sqlite checkpoints). Check `git status` before committing so nothing large or private slips in.

## Two runtimes: WSL vs Windows

Most mistakes here come from running something in the wrong place.

| Where | What runs there |
|---|---|
| **WSL** `Ubuntu-24.04`, conda env `deepseek` (`~/miniconda3`) | `server/serve.sh` (vLLM), `setup/*.sh`, Docker Engine (agent_lab sandboxes), optionally the legacy `agent/` harness and `server/webchat.py` |
| **WSL**, conda env `ft` (separate on purpose: vLLM pins torch) | `finetune_course/` (training). Stop vLLM first: it holds ~85% of VRAM |
| **Windows** `.venv` (uv, CPython 3.12) | `langgraph_course/`, `agent_lab/` (it calls `wsl docker ...` itself, and `MetaTrader5` is Windows-only), `finetune_course/08_*` (talks to vLLM) |

Run Windows-side scripts with `.venv\Scripts\python.exe` from the project root. Everything talks to the server at `AGENT_BASE_URL` (default `http://localhost:8000/v1`, from `.env`).

`.gitattributes` forces LF on `*.sh`; keep shell scripts LF or WSL bash breaks.

## Model server (`server/serve.sh`)

```bash
bash server/serve.sh            # qwen35 (default): RedHatAI/Qwen3.5-9B-quantized.w4a16, served as Qwen/Qwen3.5-9B, 32k ctx, 4 seqs, qwen3_coder tool parser
bash server/serve.sh qwen3      # Qwen3-8B-AWQ, hermes tool parser
bash server/serve.sh deepseek   # DeepSeek-R1-Distill-Qwen-14B AWQ, no tool calling, 8k ctx, 1 seq
```

Wait for `Application startup complete`. All profiles use pre-quantized checkpoints (vLLM ≥ 0.32 dropped bitsandbytes). Qwen profiles use `--reasoning-parser qwen3`, so `<think>` goes into `reasoning_content`, and thinking is toggled per request with `chat_template_kwargs={"enable_thinking": ...}`. Structured output uses xgrammar with `disable_any_whitespace` (otherwise the model can loop on newlines after long thinking).

`python server/smoke_test.py` checks the server; `python server/webchat.py` is a Gradio chat UI on :7860.

## Components

### `langgraph_course/` (main active work)
21 numbered lesson scripts on LangGraph / multi-agent patterns, each with an "Interview topics" docstring; `INTERVIEW_GUIDE.md` holds the Q&A notes. See its README for the lesson table.

- `common.py` is the shared base. `get_llm(thinking=False)` returns a `ChatOpenAI` pointed at vLLM. It detects the served model from `/models` (or uses `AGENT_MODEL`) and applies the model card's sampling. Use `thinking=True` for planners, critics and judges. Also `ask()`, `structured()` and `make_react_agent()`.
  - `structured()` adds a readable field list to the prompt, because vLLM's guided decoding enforces the schema without showing it to the model. Don't swap that list for a raw JSON-schema dump: measured, it made routing worse.
- `text_tool_adapter.py`: prompt-level tool calling for models without native support (lesson 03b, DeepSeek profile).
- `run_all.py` runs lessons and regex-checks their **outcomes** (`CHECKS` dict). When you change a lesson's printed output, update its checks. Results are saved as `results_<model>.json`.
  ```
  .venv\Scripts\python.exe langgraph_course\run_all.py [--repeat 3 05 06] [--save f.json] [--compare a.json b.json]
  ```
- `stress_evals.py`: evals that leave out the lessons' guardrails, for comparing raw model capability (`stress_<model>.json`).
- `mcp_servers/tickets_server.py`: MCP server for lesson 19. `checkpoints.sqlite` (lesson 09) is gitignored.

### `finetune_course/` (LLM fine-tuning, basics to advanced)
19 numbered lessons (00–18) on the HF stack (transformers + PEFT + TRL), same style as `langgraph_course/`: an "Interview topics" docstring per script, Q&A in `INTERVIEW_GUIDE.md`, and the lesson table in its README. Env: WSL conda `ft`, created by `setup/05_setup_finetune_env.sh`.

- `common.py`: `load_model(name, quant4=, adapter=)`, `render()` (splits the model's own chat-template rendering into prompt/completion, so train text == serve text, including Qwen3's empty `<think>` block), `encode()` (-100 prompt masking), `eval_tickets/eval_toolcalls/eval_math` (fixed `[eval ...] accuracy=` lines that `run_all.py` regexes), `free_gpu()`.
- `data.py`: seeded synthetic tasks (tickets → JSON, arithmetic, tool calls). Ticket val/test use **held-out phrasings**. Don't merge the pools: with shared phrasing the task saturates at 100% in 40 steps.
- Lessons depend on earlier outputs (05 → 07/09/13/17, 06 → 12, 11 → 13/17) in `outputs/` (gitignored). `run_all.py` runs them in order with `--quick` and checks numeric outcomes (e.g. after > before), and writes `outputs/logs/<lesson>.log`.
- On Windows/WSL a full GPU spills to system RAM instead of raising OOM, so training silently gets 10–50× slower. Always `del model, trainer` and then `free_gpu()` between runs in one process.
- transformers 5: `warmup_ratio` was removed; use `warmup_steps=0.1` (a float < 1 is a ratio).
- TRL 1.x SFTTrainer computes the loss with a chunked fused LM head and disables Liger's fused CE. Measure logits/Liger effects with a plain `Trainer`.
- Qwen3-0.6B copies format placeholders from prompts literally ("Answer: <57>", "Answer: N"). Describe formats; don't show fillable templates.
- `bash server/serve.sh ft` serves Qwen3-0.6B plus the course's LoRA adapters (`tickets`, `toolcalls`) for lesson 13 `--serve-check`.

```
python finetune_course/run_all.py [05 07] [--full] [--save f.json]     # WSL, env ft, vLLM stopped
```

### `ml_course/` (classic ML + DL fundamentals)
16 numbered lessons (00–15), the runnable companion to `ml_fundamentals.md`: 00–07 classic ML (numpy + scikit-learn: leakage/CV, linear/logistic, GD, bias-variance, trees/boosting, SVM/kNN, k-means/PCA, metrics), 08–15 DL in plain torch (autograd, training loop, init, normalisation, optimizers, CNN, RNN/LSTM, attention). Same style: "Interview topics" docstring, `INTERVIEW_GUIDE.md` with measured numbers, README lesson table. Env: WSL conda `ft` (scikit-learn/scipy added by `setup/05_setup_finetune_env.sh`); models are tiny, so vLLM can stay up, and everything runs on CPU too.

- Algorithms are written from scratch and checked against sklearn/torch (parity lines like `max_diff=...`).
- Lessons print `[tag] key=value` lines via `common.kv()`. `run_all.py` parses them into `{tag: {key: value}}`, and `CHECKS` holds predicates on **effects** (e.g. leaky CV > honest CV + 0.2), not exact numbers. When you change a lesson's printed results, update its predicates.
- Every printed claim was checked against a run. Several first drafts were wrong (LN after a linear input layer erases input scale; ROC vs PR-AUC needs a fixed scorer; LSTM fails at length 30 in the quick budget), so re-measure before changing the explanatory text.

```
python ml_course/run_all.py [09 11] [--full] [--save f.json]     # WSL, env ft
```

### `inference_course/` (LLM inference optimization: the user's chosen specialization)
Roadmap in its README: Phase 0 fundamentals (00 GPU roofline, 01 decode/KV math, 02 serving benchmark) is done; next are serving knobs (03–06), quantization (07–09), speculative decoding (10–11), Triton kernels (12–15) and a mini-engine capstone (16–17). Same conventions as `ml_course/` (`kv()` lines, effect predicates in `run_all.py`, measured numbers in `INTERVIEW_GUIDE.md`). Env: WSL conda `ft`.

- Machine state matters: GPU lessons need vLLM **stopped**; serving lessons need it **up** and print `[skip]` otherwise (`run_all.py` reports those checks as skipped).
- Lesson 00 writes `outputs/gpu.json` (measured bandwidth/TFLOPs); later predictions use it, not spec sheets.
- `common.weight_bytes()` reads exact per-tensor bytes from safetensors headers in the HF cache. Bytes per decode step = layers + lm_head (not the embedding table, vision or MTP).
- WSL has 7.2 GiB of RAM: vLLM can't mmap the 11 GB `qwen35` checkpoint, so serving runs used `serve.sh qwen3` until `.wslconfig` gets more memory/swap.

### `agent_lab/` (parallel sandboxed agents)
LangGraph orchestrator: `plan → Send fan-out to N solvers → verify → decide (retry|report)`. See `agent_lab/README.md`.

- `orchestrator.py`: graph and prompts. The planner gives each solver a *different* strategy. The verifier re-runs the solvers' work from read-only `/candidates` mounts. Majority vote is only a fallback.
- `core.py`: hand-written agent loop with a wall-clock deadline, step cap, context compaction (~12k tokens per agent, because of the small KV budget), and a forced `submit_result` via `tool_choice` on timeout. The tools are `run_python`, `run_bash`, `write_file`, `read_file`, `fetch_fx` and `submit_result`.
- `sandbox.py`: one Docker container per agent through `wsl -d Ubuntu-24.04 docker ...`. Each runs non-root with `--cap-drop ALL`, CPU/RAM/pids limits and only its workspace mounted, but has **full internet access**. Image: `agent-lab-sandbox:latest` from `agent_lab/sandbox/Dockerfile`.
- `fx_data.py`: read-only MetaTrader 5 candles on the host, serialised with a lock. Timestamps are broker server time, not UTC. Keep it limited to copy/read functions; never add order functions.
- `tasks.py`: `TASKS` dict. `pe_*` tasks are math with known answers and are auto-graded. `fx_*` tasks are open-ended research.
- `core.py`, `orchestrator.py` and `run.py` use flat imports (`from core import ...`), and `core.py` adds `langgraph_course/` to `sys.path` to import `common`. Run them as scripts, not as a package.

```
docker build -t agent-lab-sandbox:latest agent_lab/sandbox      # WSL, once
.venv\Scripts\python.exe agent_lab\run.py --list
.venv\Scripts\python.exe agent_lab\run.py --task pe_semiprimes [--mode single]
.venv\Scripts\python.exe agent_lab\run.py "custom task" --agents 3 --minutes 10
```
Output goes to `agent_lab/runs/<timestamp>-<task>-<mode>/` (gitignored): `report.md`, `result.json`, `events.jsonl`, `commands.log`, and per-agent workspaces.

### `agent/` + `run_agent.py` + `tests/` (legacy, DeepSeek era)
The original deterministic harness: the model emits `<tool_call>{json}</tool_call>` / `<final_answer>`, which is validated with pydantic (`schema.py`). Tools (`tools.py`) are a safe AST calculator, `read_file` limited to `tests/fixtures/`, and an allowlisted shell. Logs go to `logs/<run_id>.ndjson`. The top-level `README.md` describes this setup and is partly outdated: the server now defaults to Qwen3.5-9B and no longer uses bitsandbytes.

```
pytest tests/test_schema.py tests/test_tools.py -v       # unit tests, no server needed
python tests/run_tests.py [--scenario arithmetic_basic]  # live scenarios against the server
```

### `setup/`
`01_install_wsl.ps1` (admin), `02_setup_env.sh` (Miniconda, PyTorch cu128, vLLM), `03_download_model.sh`, `04_install_docker.sh` (Docker Engine inside WSL), and `new_setup.sh`.

## Conventions

- Python 3.12, `from __future__ import annotations`, type hints. Modules open with a docstring that explains the *why* (design choices, gotchas found by measurement). Keep that style.
- Linted with ruff (default config, no config file). Broad excepts carry `# noqa: BLE001` with a reason.
- Windows consoles: `common.py` reconfigures stdout to UTF-8. Scripts that print model output should import it or do the same.
- The local 9B model is unreliable. Add guardrails in code (validation, retries, caps, deterministic checks) rather than relying on prompt wording alone, and verify reliability with `run_all.py --repeat N`.
- Never put secrets into agent_lab tasks or mounts, because sandboxes can reach the network. `.env` holds `HF_TOKEN`, `AGENT_BASE_URL` and `AGENT_MODEL`; don't print or commit it.
