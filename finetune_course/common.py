"""
common.py
---------
Shared helpers for every lesson in the fine-tuning course.

Runs in the WSL conda env `ft` (see setup/05_setup_finetune_env.sh), NOT the
Windows .venv: training needs CUDA PyTorch, bitsandbytes, etc. Stop the vLLM
server first; it reserves ~85% of the 16 GB card.

Base models are Qwen3 (0.6B / 1.7B / 4B), not the Qwen3.5-9B the server runs:
  * 0.6B trains in seconds-to-minutes on one GPU, so you can run many
    experiments. The skills transfer; the hyperparameters mostly do too.
  * Qwen3 is a plain transformer. Qwen3.5 mixes Gated-DeltaNet (linear
    attention) layers that need extra kernels to train efficiently.
  * All three share one tokenizer and chat template, so a 4B teacher can
    distil into a 0.6B student token-by-token (lesson 12).

The one rule that matters most here: the text you train on must be
byte-identical to the text you will prompt with at inference time (same chat
template, same empty <think></think> block for non-thinking mode, same EOS).
`render()` guarantees that by rendering the full conversation with the
tokenizer's own template and slicing off the prompt.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any
from collections.abc import Callable, Iterable

import torch

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
sys.path.insert(0, str(HERE))

try:
    from dotenv import load_dotenv
    load_dotenv(HERE.parent / ".env")  # HF_TOKEN for downloads; never printed
except ImportError:
    pass

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BARS", "1")  # keep lesson output readable
try:
    from transformers.utils import logging as _hf_logging
    _hf_logging.disable_progress_bar()
except ImportError:
    pass

BASE_SMALL = "Qwen/Qwen3-0.6B"
BASE_MED = "Qwen/Qwen3-1.7B"
BASE_QLORA = "Qwen/Qwen3-4B"

# Attention projections + MLP: the "all linear layers" LoRA target set that
# QLoRA paper found necessary to match full fine-tuning quality.
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


# ---------------------------------------------------------------------------
# CLI / reproducibility
# ---------------------------------------------------------------------------

def parse_args(doc: str | None = None, extra: Callable[[argparse.ArgumentParser], None] | None = None) -> argparse.Namespace:
    """Every lesson takes --quick (what run_all.py uses: fewer steps, smaller eval)."""
    p = argparse.ArgumentParser(description=(doc or "").strip().splitlines()[0] if doc else None)
    p.add_argument("--quick", action="store_true", help="fewer steps / smaller eval (minutes, not tens of minutes)")
    p.add_argument("--seed", type=int, default=0)
    if extra:
        extra(p)
    args = p.parse_args()
    seed_all(args.seed)
    return args


def seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def out_dir(name: str) -> Path:
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def require(path: Path, lesson: str) -> Path:
    if not path.exists():
        sys.exit(f"Missing {path}. Run lesson {lesson} first.")
    return path


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


# ---------------------------------------------------------------------------
# GPU memory
# ---------------------------------------------------------------------------

def reset_peak() -> None:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def peak_gb() -> float:
    """Peak memory ALLOCATED by tensors. nvidia-smi shows more (caching allocator + CUDA context)."""
    return torch.cuda.max_memory_allocated() / 1024**3


def free_gpu() -> None:
    """Call after `del model, trainer` in the CALLER (a helper cannot drop the caller's references).

    Why it matters here: on Windows/WSL the NVIDIA driver does not raise OOM when
    VRAM is full; it silently spills to system RAM ("sysmem fallback") and
    training gets 10-50x slower. A forgotten model reference looks like a slow GPU.
    """
    import gc
    gc.collect()
    torch.cuda.empty_cache()


def count_params(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


def print_trainable(model: torch.nn.Module, label: str = "") -> None:
    t, n = count_params(model)
    print(f"[params{' ' + label if label else ''}] trainable={t:,} / total={n:,} ({100 * t / n:.3f}%)")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_tokenizer(name: str = BASE_SMALL):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(name: str = BASE_SMALL, *, quant4: bool = False, dtype: torch.dtype = torch.bfloat16,
               attn: str = "sdpa", adapter: str | Path | None = None):
    """(model, tokenizer). quant4=True -> QLoRA-style NF4 + double quant, bf16 compute."""
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    free, total = torch.cuda.mem_get_info()
    if free < 0.5 * total:
        print(f"[warn] only {free / 1024**3:.1f} of {total / 1024**3:.1f} GB VRAM free before loading. "
              "Is vLLM still running, or is an old model still referenced? Expect sysmem-fallback slowness.")
    kwargs: dict[str, Any] = {"dtype": dtype, "attn_implementation": attn, "device_map": {"": 0}}
    if quant4:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
    if adapter is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter))
    return model, load_tokenizer(name)


# ---------------------------------------------------------------------------
# Chat template -> (prompt text, completion text) -> tokens with masked labels
# ---------------------------------------------------------------------------

def render_prompt(tok, messages: list[dict], tools: list[dict] | None = None) -> str:
    """Exactly what the model sees at inference: template + generation prompt, thinking OFF."""
    return tok.apply_chat_template(messages, tools=tools, tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)


def render(tok, messages: list[dict], assistant: dict | str, tools: list[dict] | None = None) -> tuple[str, str]:
    """Render prompt+answer with the model's own template and split at the prompt.

    Slicing the full rendering (rather than concatenating prompt + answer by
    hand) means special tokens, the empty <think> block, tool-call tags and the
    end-of-turn token are all exactly what the template would produce.
    """
    if isinstance(assistant, str):
        assistant = {"role": "assistant", "content": assistant}
    prompt = render_prompt(tok, messages, tools)
    full = tok.apply_chat_template([*messages, assistant], tools=tools, tokenize=False, enable_thinking=False)
    if not full.startswith(prompt):
        raise ValueError("chat template is not prefix-stable; cannot split prompt/completion\n"
                         f"PROMPT:\n{prompt!r}\nFULL:\n{full!r}")
    completion = full[len(prompt):]
    # Train up to and including the end-of-turn token; the trailing newline is
    # never generated (generation stops at <|im_end|>).
    return prompt, completion.rstrip("\n")


def example_texts(tok, ex: dict) -> tuple[str, str]:
    """Works for all three data.py tasks."""
    return render(tok, ex["messages"], ex.get("assistant") or ex["target"], tools=ex.get("tools"))


def encode(tok, prompt: str, completion: str, max_len: int = 512) -> dict[str, list[int]]:
    """Token ids + labels where prompt positions are -100 (ignored by cross-entropy).

    Prompt and completion are tokenised separately so the boundary is exact.
    (Tokenising the joined string can merge tokens across the boundary.)
    """
    p = tok(prompt, add_special_tokens=False)["input_ids"]
    c = tok(completion, add_special_tokens=False)["input_ids"]
    ids = (p + c)[:max_len]
    labels = ([-100] * len(p) + c)[:max_len]
    return {"input_ids": ids, "labels": labels, "attention_mask": [1] * len(ids)}


def encode_all(tok, examples: Iterable[dict], max_len: int = 512) -> list[dict[str, list[int]]]:
    return [encode(tok, *example_texts(tok, ex), max_len=max_len) for ex in examples]


def sft_records(tok, examples: Iterable[dict]) -> list[dict[str, str]]:
    """{"prompt", "completion"} strings for TRL's SFTTrainer (completion-only loss)."""
    return [dict(zip(("prompt", "completion"), example_texts(tok, ex))) for ex in examples]


def pad_collate(pad_id: int, multiple: int = 1) -> Callable[[list[dict]], dict[str, torch.Tensor]]:
    """Right-pad a batch. Padding gets label -100 and attention 0.

    multiple=16: FP8 matmuls (lesson 16) need every dim, including tokens, divisible by 16.
    """
    def collate(batch: list[dict]) -> dict[str, torch.Tensor]:
        n = max(len(b["input_ids"]) for b in batch)
        n = -(-n // multiple) * multiple
        def pad(key: str, value: int) -> torch.Tensor:
            return torch.tensor([b[key] + [value] * (n - len(b[key])) for b in batch])
        return {"input_ids": pad("input_ids", pad_id), "labels": pad("labels", -100),
                "attention_mask": pad("attention_mask", 0)}
    return collate


# ---------------------------------------------------------------------------
# Generation + evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate(model, tok, prompts: list[str], max_new_tokens: int = 96, batch_size: int = 32) -> list[str]:
    """Greedy batched generation. LEFT padding: new tokens must follow the real last token."""
    model.eval()
    tok.padding_side = "left"
    outs: list[str] = []
    for i in range(0, len(prompts), batch_size):
        enc = tok(prompts[i:i + batch_size], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, temperature=None,
                             top_p=None, top_k=None, pad_token_id=tok.pad_token_id)
        outs += tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=False)
    tok.padding_side = "right"
    return [o.split("<|im_end|>")[0].strip() for o in outs]


def _summarise(rows: list[dict[str, bool]]) -> dict[str, float]:
    return {k: sum(r[k] for r in rows) / len(rows) for k in rows[0]}


def eval_tickets(model, tok, examples: list[dict], label: str = "", show: int = 0) -> dict[str, float]:
    """Prints `[eval <label>] accuracy=0.xxx ...` (a fixed format run_all.py regex-checks)."""
    from data import grade_ticket
    t0 = time.time()
    outs = generate(model, tok, [render_prompt(tok, ex["messages"]) for ex in examples])
    rows = [grade_ticket(o, ex["label"]) for o, ex in zip(outs, examples)]
    m = _summarise(rows)
    fields = " ".join(f"{k}={m[k]:.2f}" for k in ("category", "priority", "product", "refund_requested"))
    print(f"[eval {label}] accuracy={m['exact']:.3f} valid_json={m['valid']:.2f} {fields} "
          f"(n={len(rows)}, {time.time() - t0:.0f}s)")
    for o, ex in list(zip(outs, examples))[:show]:
        print(f"    in : {ex['messages'][1]['content']}\n    out: {o[:200]!r}\n    want: {ex['target']}")
    return m


def eval_toolcalls(model, tok, examples: list[dict], label: str = "", show: int = 0) -> dict[str, float]:
    from data import grade_toolcall
    outs = generate(model, tok, [render_prompt(tok, ex["messages"], ex["tools"]) for ex in examples], max_new_tokens=128)
    rows = [grade_toolcall(o, ex["call"]) for o, ex in zip(outs, examples)]
    m = _summarise(rows)
    print(f"[eval {label}] tool_exact={m['exact']:.3f} right_tool={m['right_tool']:.2f} valid={m['valid']:.2f} (n={len(rows)})")
    for o, ex in list(zip(outs, examples))[:show]:
        print(f"    in : {ex['messages'][1]['content']}\n    out: {o[:200]!r}\n    want: {json.dumps(ex['call'])}")
    return m


def eval_math(model, tok, examples: list[dict], label: str = "", max_new_tokens: int = 256) -> dict[str, float]:
    from data import extract_answer
    outs = generate(model, tok, [render_prompt(tok, ex["messages"]) for ex in examples], max_new_tokens=max_new_tokens)
    preds = [extract_answer(o) for o in outs]
    acc = sum(p == ex["answer"] for p, ex in zip(preds, examples)) / len(examples)
    fmt = sum(p is not None for p in preds) / len(examples)
    avg_len = sum(len(tok(o)["input_ids"]) for o in outs) / len(outs)
    print(f"[eval {label}] math_acc={acc:.3f} format_ok={fmt:.2f} avg_tokens={avg_len:.0f} (n={len(examples)})")
    return {"acc": acc, "format": fmt, "avg_tokens": avg_len}


def table(rows: list[dict[str, Any]], cols: list[str]) -> None:
    """Small fixed-width table printer for the comparison lessons."""
    widths = {c: max(len(c), *(len(f"{r.get(c, '')}") for r in rows)) for c in cols}
    print("  ".join(c.ljust(widths[c]) for c in cols))
    print("  ".join("-" * widths[c] for c in cols))
    for r in rows:
        print("  ".join(f"{r.get(c, '')}".ljust(widths[c]) for c in cols))
