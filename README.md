# Local DeepSeek Agent

Run [deepseek-ai/DeepSeek-R1-Distill-Qwen-14B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B) locally on an RTX 5080 and test its agentic tool-use capabilities.

| Component | Technology |
|-----------|-----------|
| Inference server | vLLM (OpenAI-compatible API) |
| Quantisation | bitsandbytes NF4 4-bit, BF16 compute |
| GPU | NVIDIA RTX 5080 (16 GB VRAM) |
| Host | Windows 11/10, WSL2 Ubuntu 24.04 |
| Agent framework | Custom deterministic harness (this repo) |

---

## Prerequisites

- Windows 10/11 with WSL2 available (if not installed, see below)
- NVIDIA RTX 5080 with driver ≥ 555 (you have 610.88 ✓)
- ~30 GB disk space inside WSL for model weights
- ~32 GB total RAM recommended (model uses ~12–14 GB VRAM)

---

## Quick-start

### Step 1 – Install WSL2 (run once, requires Admin)

Open PowerShell **as Administrator** and run:

```powershell
.\setup\01_install_wsl.ps1
```

Follow the on-screen instructions. A reboot may be required.

### Step 2 – Set up the Python environment (inside WSL)

```bash
# Open Ubuntu-24.04
wsl -d Ubuntu-24.04

# Navigate to the project (path inside WSL)
cd /mnt/c/Users/Joseph\ Raymond/Documents/projects/model_run

# Run the setup script (installs Miniconda, PyTorch with CUDA 12.8, vLLM, etc.)
bash setup/02_setup_env.sh
```

### Step 3 – Download the model

```bash
bash setup/03_download_model.sh
```

This downloads ~30 GB from Hugging Face into `~/.cache/huggingface`.  
To use a different location, set `HF_HOME` before running:

```bash
export HF_HOME=/mnt/d/hf_cache
bash setup/03_download_model.sh
```

### Step 4 – Copy and configure `.env`

```bash
cp .env.example .env
# Edit .env if you want to override HF_HOME or the server URL.
```

### Step 5 – Start the inference server

In a **dedicated terminal**:

```bash
conda activate deepseek
bash server/serve.sh
```

The server starts on `http://localhost:8000`. Wait for the line:

```
INFO:     Application startup complete.
```

Expected VRAM after loading: ~9–10 GB.

### Step 6 – Smoke test the server

```bash
python server/smoke_test.py
```

Expected output: all three checks PASS plus a VRAM snapshot.

### Step 7 – Run the agent

```bash
python run_agent.py "What is 1234 multiplied by 5678?"
python run_agent.py "Read inventory.txt and calculate the total value of Gadget Y."
python run_agent.py "What time is it? Use the date shell command."
```

---

## Running the test suite

```bash
python tests/run_tests.py
```

This runs all 7 scenarios and prints a table:

```
Scenario                       Status   Turns Tools  Latency  VRAM Δ
------------------------------------------------------------------------
arithmetic_basic               PASS         2     1    12.3s   0MiB
file_then_calc                 PASS         4     2    35.1s   0MiB
shell_allowed                  PASS         2     1     9.8s   0MiB
shell_rejected                 PASS         2     1    11.2s   0MiB
malformed_recovery             PASS         3     0    14.4s   0MiB
no_tool_needed                 PASS         1     0     6.5s   0MiB
budget_analysis                PASS         4     2    38.7s   0MiB
========================================================================
  Results  :  7/7 passed
  Avg latency : 18.3s per scenario
  Tool calls  : 8   Schema errors: 1
  VRAM now    : 9872 / 16303 MiB (60.6% used)
```

To run a single scenario:

```bash
python tests/run_tests.py --scenario arithmetic_basic
```

Unit tests (no server required):

```bash
pytest tests/test_schema.py tests/test_tools.py -v
```

---

## Project layout

```
model_run/
├── setup/
│   ├── 01_install_wsl.ps1      Windows Admin: install WSL2 + Ubuntu
│   ├── 02_setup_env.sh         WSL: Miniconda + PyTorch + vLLM
│   └── 03_download_model.sh    WSL: download model from HF
├── server/
│   ├── serve.sh                Start vLLM OpenAI-compatible server
│   ├── smoke_test.py           Quick server health check
│   └── requirements.txt        Pinned pip dependencies
├── agent/
│   ├── schema.py               JSON action schema + parser
│   ├── tools.py                Calculator, file reader, shell sandbox
│   ├── logger.py               NDJSON structured logger
│   └── harness.py              Deterministic agent loop
├── tests/
│   ├── fixtures/
│   │   ├── inventory.txt       Sample warehouse data
│   │   └── budget.txt          Sample Q3 budget data
│   ├── scenarios.py            Scenario definitions
│   ├── run_tests.py            Live scenario runner + metrics
│   ├── test_schema.py          pytest: schema parser unit tests
│   └── test_tools.py           pytest: tool implementation unit tests
├── run_agent.py                CLI wrapper for the agent harness
├── .env.example                Copy to .env and configure
├── .gitignore
└── README.md
```

---

## How the agent works

```
User task
   │
   ▼
System prompt (defines tool schema + XML output format)
   │
   ▼
┌─────────────────────────────────────────────────────────────┐
│  AGENT LOOP  (max 10 turns)                                 │
│                                                             │
│  Model (DeepSeek-R1-Distill-Qwen-14B, local, 4-bit)       │
│    │  emits <think>...</think> stripped first              │
│    │  then  <tool_call>{...}</tool_call>                   │
│    │     or <final_answer>...</final_answer>               │
│    ▼                                                        │
│  Schema validation (pydantic)                              │
│    │  ✓ valid tool call → dispatch                         │
│    │  ✗ malformed      → feed error back, let model retry  │
│    ▼                                                        │
│  Tool dispatch                                              │
│    • calculator  – safe AST arithmetic evaluator           │
│    • read_file   – read-only, fixtures/ directory only     │
│    • shell       – allowlist: ls pwd date echo uname …     │
│    ▼                                                        │
│  Observation appended to conversation as user message       │
└─────────────────────────────────────────────────────────────┘
   │
   ▼
Final answer   +   logs/<run_id>.ndjson
```

All model actions, tool calls, results, and errors are logged as
newline-delimited JSON under `logs/`.

---

## Troubleshooting

### `nvidia-smi` not found inside WSL

```bash
# In elevated Windows PowerShell
wsl --update
```

NVIDIA's Windows GPU driver (≥ 555) provides `nvidia-smi` inside WSL
automatically via `/usr/lib/wsl/lib`. If still missing, reinstall the
Windows NVIDIA driver from nvidia.com.

### VRAM OOM during model load

vLLM may OOM if other processes are using VRAM. Check and kill them:

```bash
nvidia-smi
# Then free VRAM by closing GPU-using apps before restarting serve.sh
```

You can also reduce the context window:

```bash
# Edit server/serve.sh and set:
MAX_LEN=4096
```

### Model outputs do not contain `<tool_call>` tags

DeepSeek-R1 distilled models may not follow tool-call instructions
reliably on complex tasks. Tips:
- Ensure the system prompt is unchanged (it defines the strict format).
- Use `temperature=0.0` (already the default).
- Increase `max_tokens` to give the model room to reason before acting.
- If the model only outputs `<think>...</think>`, check that the vLLM
  server isn't truncating output (verify `max-model-len` and `max_tokens`).

### `vllm` import error or CUDA version mismatch

The RTX 5080 (Blackwell / GB203) requires CUDA ≥ 12.6 and PyTorch ≥ 2.6.
The setup script installs the `cu128` PyTorch wheel. If vLLM doesn't
detect the GPU:

```bash
python -c "import torch; print(torch.cuda.get_device_name(0))"
# Should print: NVIDIA GeForce RTX 5080
pip install --upgrade vllm
```

### Slow first generation

The first call loads the model into VRAM and compiles CUDA kernels.
Subsequent calls are much faster. Expect 30–60 s for the first response.

---

## Shutdown

To stop the server cleanly, press `Ctrl+C` in the terminal running `serve.sh`.
