"""
13 - Merge, export, and serve fine-tuned adapters with vLLM (multi-LoRA)
=======================================================================

Two parts:
  A. (WSL, env ft, vLLM stopped) merge the lesson-05 adapter into the base,
     save a standalone checkpoint, verify it matches the adapter.
         python finetune_course/13_merge_export_serve.py
  B. (vLLM running with the `ft` profile) query the served adapters through
     the OpenAI API and grade them. Stdlib only: runs in any env/OS.
         bash server/serve.sh ft
         python finetune_course/13_merge_export_serve.py --serve-check

Interview topics
----------------
- Deployment options for a LoRA:
    1. Merge (W += scale*B@A) -> ordinary checkpoint, zero overhead, can be
       quantised (AWQ/GPTQ/FP8) and served anywhere. One model per task.
    2. Keep adapters separate, serve ONE base + MANY adapters (vLLM
       --enable-lora, S-LoRA/Punica kernels batch requests for different
       adapters together). Per-request `model="<adapter name>"`. Great for
       per-customer/per-task fine-tunes; small per-token overhead.
- --max-lora-rank must be >= your adapter's r; --max-loras = adapters
  resident on the GPU at once; others are swapped from CPU.
- Merging a QLoRA adapter: load the base in bf16 (NOT 4-bit), merge, then
  re-quantise. Merging into 4-bit weights rounds the update away.
- Export hygiene: safetensors (no pickle code execution), save the
  tokenizer + chat template + generation_config with the weights, record
  base model revision + data + hyperparameters (model card).
- Parity test before shipping: same prompts through (adapter in PyTorch)
  vs (served model) => same answers. Template or sampling drift shows up here.
- Serving a fine-tune also needs the right parsers: tool-call parser for
  lesson 11's adapter, reasoning parser for thinking models.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from data import grade_ticket, grade_toolcall, ticket_splits, toolcall_splits  # noqa: E402

BASE_URL = "http://localhost:8000/v1"


def chat(model: str, messages: list[dict], tools: list[dict] | None = None) -> dict:
    body = {"model": model, "messages": messages, "temperature": 0.0, "max_tokens": 128,
            "chat_template_kwargs": {"enable_thinking": False}}
    if tools:
        body["tools"] = tools
    req = urllib.request.Request(f"{BASE_URL}/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)["choices"][0]["message"]


def as_text(msg: dict) -> str:
    """Re-serialise parsed tool calls into Qwen's text format so data.grade_toolcall can grade them."""
    calls = msg.get("tool_calls") or []
    if calls:
        f = calls[0]["function"]
        return f'<tool_call>{json.dumps({"name": f["name"], "arguments": json.loads(f["arguments"])})}</tool_call>'
    return msg.get("content") or ""


def serve_check(n: int) -> None:
    with urllib.request.urlopen(f"{BASE_URL}/models", timeout=5) as r:
        models = [m["id"] for m in json.load(r)["data"]]
    print(f"served models: {models}")
    tickets = ticket_splits()["test"][:n]
    tools = toolcall_splits()["test"][:n]
    for name in [m for m in models if m in ("Qwen/Qwen3-0.6B", "tickets", "toolcalls")]:
        t0 = time.time()
        acc = sum(grade_ticket(chat(name, ex["messages"]).get("content") or "", ex["label"])["exact"]
                  for ex in tickets) / len(tickets)
        tacc = sum(grade_toolcall(as_text(chat(name, ex["messages"], ex["tools"])), ex["call"])["exact"]
                   for ex in tools) / len(tools)
        print(f"[served {name}] ticket_acc={acc:.3f} tool_exact={tacc:.3f} ({time.time() - t0:.0f}s, n={n})")
    print("Each adapter should win on its own task; the base on neither. Same server, same GPU, per-request routing.")


def merge_and_export(quick: bool) -> None:
    import torch
    from peft import PeftModel

    from common import BASE_SMALL, OUT, eval_tickets, free_gpu, load_model, out_dir, render_prompt, require, section

    test = ticket_splits()["test"][: 60 if quick else 200]
    adapter = require(OUT / "05_sft_lora", "05")

    section("1. The adapter on disk")
    cfg = json.loads((adapter / "adapter_config.json").read_text())
    print({k: cfg[k] for k in ("base_model_name_or_path", "r", "lora_alpha", "target_modules", "peft_type")})
    for f in sorted(adapter.iterdir()):
        print(f"  {f.name:32s} {f.stat().st_size / 1e6:8.2f} MB")

    section("2. Merge and save a standalone checkpoint")
    model, tok = load_model(BASE_SMALL)
    model = PeftModel.from_pretrained(model, str(adapter))
    probe = tok([render_prompt(tok, test[0]["messages"])], return_tensors="pt").to("cuda")
    with torch.no_grad():
        ref = model(**probe).logits
    merged = model.merge_and_unload()
    dest = out_dir("13_merged")
    merged.save_pretrained(dest, safe_serialization=True)
    tok.save_pretrained(dest)
    for f in sorted(dest.iterdir()):
        print(f"  {f.name:32s} {f.stat().st_size / 1e6:8.2f} MB")
    del merged, model
    free_gpu()

    section("3. Parity: reload the merged checkpoint and compare")
    reloaded, tok = load_model(str(dest))
    with torch.no_grad():
        diff = (reloaded(**probe).logits - ref).abs().max().item()
    print(f"max |logit diff| reloaded-merged vs adapter = {diff:.3f}")
    m = eval_tickets(reloaded, tok, test, "merged checkpoint")
    print(f"[export] merged accuracy={m['exact']:.3f} saved to {dest}")

    section("4. Serve it")
    print("Multi-LoRA (base + adapters, what `bash server/serve.sh ft` does):")
    print("  vllm serve Qwen/Qwen3-0.6B --enable-lora --max-lora-rank 64 \\")
    print("      --lora-modules tickets=finetune_course/outputs/05_sft_lora toolcalls=finetune_course/outputs/11_tool_lora")
    print("Merged checkpoint as a plain model:")
    print(f"  vllm serve {dest}")
    print("Then: python finetune_course/13_merge_export_serve.py --serve-check")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--serve-check", action="store_true", help="grade adapters served by vLLM (serve.sh ft)")
    ap.add_argument("-n", type=int, default=40)
    args = ap.parse_args()
    if args.serve_check:
        serve_check(args.n)
    else:
        merge_and_export(args.quick)


if __name__ == "__main__":
    main()
