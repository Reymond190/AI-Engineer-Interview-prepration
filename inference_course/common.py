"""
common.py
---------
Shared helpers for the LLM inference optimization course.

Three kinds of lesson live here, and the helpers are grouped the same way:

1. GPU lessons (kernels, microbenchmarks): WSL conda env `ft` (torch + Triton),
   vLLM STOPPED. vLLM holds ~85% of VRAM, and on Windows/WSL a full GPU
   spills to system RAM instead of raising OOM, so measurements would be
   silently wrong. `cuda_ms()` times with CUDA events (warmup + median).
2. Pen-and-paper lessons (roofline, KV-cache math): no GPU. They read model
   shapes and exact per-tensor byte counts from the safetensors headers in the
   local HF cache (`weight_bytes()`), not from parameter-count folklore. A w4a16
   checkpoint is not "9B params x 0.5 bytes": embeddings and lm_head often stay
   BF16, and that changes the decode roofline a lot.
3. Serving lessons: talk to the vLLM server at AGENT_BASE_URL over plain HTTP
   (stdlib urllib, so they run in any env, WSL or Windows). `stream_completion()`
   uses the raw /completions endpoint with `ignore_eos`, so output length is
   exact and no chat template or thinking mode gets in the way.

Measured machine numbers (bandwidth, TFLOPs) are written by lesson 00 to
outputs/gpu.json and reused by later lessons, so every prediction in the
course is "this GPU" and not a spec sheet.

Every lesson prints `[tag] key=value` lines (kv()), which run_all.py parses and
checks against EFFECTS (batching is nearly free in decode, prefill is
compute-bound, ...), not exact numbers.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import statistics
import struct
import sys
import time
import urllib.request
from collections.abc import Callable
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "outputs"
GPU_JSON = OUT / "gpu.json"

# serve.sh profile -> checkpoint actually loaded (what the GPU reads per token)
MODELS = {
    "qwen35": "RedHatAI/Qwen3.5-9B-quantized.w4a16",
    "qwen3": "Qwen/Qwen3-8B-AWQ",
    "deepseek": "casperhansen/deepseek-r1-distill-qwen-14b-awq",
    "qwen3-0.6b": "Qwen/Qwen3-0.6B",
}
# name the server reports in /v1/models -> profile
SERVED = {"Qwen/Qwen3.5-9B": "qwen35", "Qwen/Qwen3-8B": "qwen3", "Qwen/Qwen3-0.6B": "qwen3-0.6b",
          "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B": "deepseek"}


# ---------------------------------------------------------------------------
# Output / CLI (same conventions as ml_course)
# ---------------------------------------------------------------------------

def parse_args(doc: str | None, extra: list[tuple[str, dict]] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=(doc or "").split("\n")[1] if doc else None)
    ap.add_argument("--quick", action="store_true", help="fewer repeats (seconds per lesson)")
    ap.add_argument("--seed", type=int, default=0)
    for flag, kw in extra or []:
        ap.add_argument(flag, **kw)
    args = ap.parse_args()
    random.seed(args.seed)
    return args


def section(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}", flush=True)


def _fmt(v: object) -> str:
    if isinstance(v, float):
        return f"{v:.4g}" if (abs(v) < 1e-3 and v != 0) or abs(v) >= 1e4 else f"{v:.4f}"
    return str(v)


def kv(tag: str, **values: float | str | bool) -> None:
    """Print one machine-checkable result line: [tag] a=0.912 b=3."""
    print(f"[{tag}] " + " ".join(f"{k}={_fmt(v)}" for k, v in values.items()), flush=True)


def table(rows: list[dict], cols: list[str] | None = None) -> None:
    if not rows:
        return
    cols = cols or list(rows[0])
    w = {c: max(len(c), *(len(_fmt(r.get(c, ""))) for r in rows)) for c in cols}
    print("  " + "  ".join(c.rjust(w[c]) for c in cols))
    for r in rows:
        print("  " + "  ".join(_fmt(r.get(c, "")).rjust(w[c]) for c in cols))


GB = 1e9      # bandwidth and model sizes in decimal GB, like spec sheets
GiB = 2**30


# ---------------------------------------------------------------------------
# GPU timing
# ---------------------------------------------------------------------------

def cuda_ms(fn: Callable[[], object], warmup: int = 3, iters: int = 10) -> float:
    """Median wall time of fn() on the GPU in ms, measured with CUDA events.

    time.perf_counter() around a CUDA call measures the *launch* (microseconds),
    not the kernel: CUDA is asynchronous. Events are recorded on the stream, so
    they bracket the actual GPU work. Median, because the first calls pay for
    autotuning / clock ramp-up and occasional outliers come from the desktop.
    """
    import torch
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b))
    return statistics.median(times)


def free_gpu() -> None:
    import gc
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except ImportError:
        pass


def load_gpu() -> dict:
    """Machine numbers measured by lesson 00. Falls back to spec values, flagged."""
    if GPU_JSON.exists():
        return json.loads(GPU_JSON.read_text()) | {"measured": True}
    print("  ! outputs/gpu.json missing (run lesson 00 with vLLM stopped); using RTX 5080 spec bandwidth "
          "and a placeholder 100 TFLOPS")
    return {"bw_gbps": 960.0, "tflops_bf16": 100.0, "measured": False}


# ---------------------------------------------------------------------------
# Model shapes and exact bytes from the local HF cache
# ---------------------------------------------------------------------------

def hf_snapshot(repo: str) -> Path | None:
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    snaps = sorted(glob.glob(str(hub / f"models--{repo.replace('/', '--')}" / "snapshots" / "*")))
    return Path(snaps[-1]) if snaps else None


def model_config(snap: Path) -> dict:
    """Text-model config (multimodal checkpoints nest it under text_config)."""
    c = json.loads((snap / "config.json").read_text())
    t = dict(c.get("text_config") or c)
    t.setdefault("tie_word_embeddings", c.get("tie_word_embeddings", False))
    t.setdefault("head_dim", None)
    if not t["head_dim"]:
        t["head_dim"] = t["hidden_size"] // t["num_attention_heads"]
    return t


_SKIP_PARAM = ("qzeros", "scales", "weight_shape", "g_idx", "zero_point", "weight_scale", "input_scale")


def _category(name: str) -> str:
    if "visual" in name:
        return "vision"
    if name.startswith("mtp"):
        return "mtp"
    if "embed_tokens" in name:
        return "embed"
    if "lm_head" in name:
        return "lm_head"
    return "layers"


def weight_bytes(snap: Path) -> dict[str, dict[str, float]]:
    """Exact bytes and (unpacked) parameter counts per category, from safetensors headers only.

    A safetensors file starts with an 8-byte little-endian header length and a
    JSON header {name: {dtype, shape, data_offsets}}, so sizes are known without
    loading a single weight. int4 checkpoints store 8 weights per int32
    (`qweight` / `weight_packed`), so params = int32 elements x 8.
    """
    out: dict[str, dict[str, float]] = {}
    for f in sorted(snap.glob("*.safetensors")):
        with open(f, "rb") as fh:
            header = json.loads(fh.read(struct.unpack("<Q", fh.read(8))[0]))
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            cat = out.setdefault(_category(name), {"bytes": 0.0, "params": 0.0})
            cat["bytes"] += meta["data_offsets"][1] - meta["data_offsets"][0]
            if any(s in name for s in _SKIP_PARAM):
                continue
            n = 1
            for d in meta["shape"]:
                n *= d
            packed = meta["dtype"] == "I32" and name.endswith(("qweight", "weight_packed"))
            cat["params"] += n * 8 if packed else n
    if "lm_head" not in out and "embed" in out:   # tied: the logits matmul reads the embedding table
        out["lm_head"] = dict(out["embed"])
    return out


def decode_bytes(w: dict[str, dict[str, float]]) -> float:
    """Weight bytes ONE decode step must stream from VRAM.

    Every layer weight and the whole lm_head are read once per step. The
    embedding table is not: a lookup reads one row. Vision tower (skipped with
    --language-model-only) and MTP heads (only used for speculative decoding)
    are not read either.
    """
    return w["layers"]["bytes"] + w["lm_head"]["bytes"]


def decode_params(w: dict[str, dict[str, float]]) -> float:
    """Parameters that take part in each token's matmuls (FLOPs/token ~ 2x this)."""
    return w["layers"]["params"] + w["lm_head"]["params"]


def kv_bytes_per_token(cfg: dict, dtype_bytes: int = 2) -> int:
    """K and V for every full-attention layer: 2 * layers * kv_heads * head_dim * bytes.

    Hybrid models (Qwen3.5: 3 Gated-DeltaNet layers per full-attention layer)
    only pay this for the full-attention layers.
    """
    types = cfg.get("layer_types")
    n_full = sum(t == "full_attention" for t in types) if types else cfg["num_hidden_layers"]
    return 2 * n_full * cfg["num_key_value_heads"] * cfg["head_dim"] * dtype_bytes


def linear_state_bytes(cfg: dict, dtype_bytes: int = 2) -> int:
    """Fixed per-SEQUENCE recurrent state of linear-attention layers (0 for plain transformers).

    Gated DeltaNet keeps a d_k x d_v matrix per value head per layer, independent
    of context length. The conv state is ignored (tiny). The state dtype is an
    assumption: vLLM's --mamba-ssm-cache-dtype decides it (auto = model dtype).
    """
    types = cfg.get("layer_types")
    if not types:
        return 0
    n_lin = sum(t == "linear_attention" for t in types)
    return n_lin * cfg["linear_num_value_heads"] * cfg["linear_key_head_dim"] * cfg["linear_value_head_dim"] * dtype_bytes


def n_full_attention(cfg: dict) -> int:
    types = cfg.get("layer_types")
    return sum(t == "full_attention" for t in types) if types else cfg["num_hidden_layers"]


def load_model_info(profile: str) -> dict | None:
    snap = hf_snapshot(MODELS[profile])
    if snap is None or not (snap / "config.json").exists():
        return None
    cfg = model_config(snap)
    w = weight_bytes(snap)
    return {"profile": profile, "repo": MODELS[profile], "cfg": cfg, "w": w,
            "decode_bytes": decode_bytes(w), "decode_params": decode_params(w),
            "kv_tok": kv_bytes_per_token(cfg), "state_seq": linear_state_bytes(cfg)}


def roofline_step(info: dict, gpu: dict, batch: int, ctx: int) -> dict:
    """Roofline time of ONE decode step for `batch` sequences that each hold `ctx` tokens.

    memory: weights once + every sequence's KV cache (+ recurrent state)
    compute: 2 FLOPs per param per token + attention (QK^T and PV: 4 * ctx * heads * head_dim per full layer)
    The step takes max(memory, compute): whichever unit is the bottleneck.
    """
    cfg = info["cfg"]
    mem = info["decode_bytes"] + batch * (ctx * info["kv_tok"] + info["state_seq"])
    attn = 4 * ctx * cfg["num_attention_heads"] * cfg["head_dim"] * n_full_attention(cfg)
    flops = batch * (2 * info["decode_params"] + attn)
    t_mem = mem / (gpu["bw_gbps"] * GB)
    t_comp = flops / (gpu["tflops_bf16"] * 1e12)
    t = max(t_mem, t_comp)
    return {"t_mem_ms": t_mem * 1e3, "t_comp_ms": t_comp * 1e3, "step_ms": t * 1e3,
            "tok_s": batch / t, "bound": "memory" if t_mem >= t_comp else "compute", "bytes": mem}


def prefill_ms(info: dict, gpu: dict, prompt: int) -> float:
    """Roofline (compute) time to prefill `prompt` tokens: linear layers + causal attention."""
    cfg = info["cfg"]
    attn = 2 * prompt * prompt * cfg["num_attention_heads"] * cfg["head_dim"] * n_full_attention(cfg)  # causal: half of 4*P^2*h*d
    flops = 2 * info["decode_params"] * prompt + attn
    return flops / (gpu["tflops_bf16"] * 1e12) * 1e3


# ---------------------------------------------------------------------------
# Serving client (stdlib only)
# ---------------------------------------------------------------------------

def base_url() -> str:
    url = os.environ.get("AGENT_BASE_URL")
    env = ROOT / ".env"
    if not url and env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("AGENT_BASE_URL="):
                url = line.split("=", 1)[1].strip().strip("'\"")
    return (url or "http://localhost:8000/v1").rstrip("/")


def served_model(timeout: float = 5.0) -> str | None:
    """Model id the server reports, or None if the server is not reachable."""
    try:
        with urllib.request.urlopen(base_url() + "/models", timeout=timeout) as r:
            return json.loads(r.read())["data"][0]["id"]
    except Exception:  # noqa: BLE001 - any failure means "no server", which lessons report as a skip
        return None


def stream_completion(model: str, prompt: str, max_tokens: int, timeout: float = 600) -> dict:
    """One streamed /completions request; returns TTFT, TPOT and token counts.

    TTFT  = request sent -> first token (queueing + prefill + first decode step)
    TPOT  = (last token - first token) / (n - 1): the steady decode pace
    e2e   = whole request
    Token count comes from the server's usage block, not from counting chunks:
    a chunk can carry 0 visible characters (partial UTF-8) or several tokens.
    """
    body = {"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0.0,
            "stream": True, "ignore_eos": True, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(base_url() + "/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"})
    t0 = time.perf_counter()
    first = last = None
    usage: dict = {}
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            obj = json.loads(data)
            if obj.get("usage"):
                usage = obj["usage"]
            if obj.get("choices"):
                now = time.perf_counter()
                first = first or now
                last = now
    end = time.perf_counter()
    n = usage.get("completion_tokens", 0)
    return {"ttft": (first or end) - t0, "tpot": (last - first) / (n - 1) if first and n > 1 else float("nan"),
            "e2e": end - t0, "out_tokens": n, "prompt_tokens": usage.get("prompt_tokens", 0)}


def unique_prompt(n_words: int) -> str:
    """~n_words tokens of filler with a random prefix, so prefix caching cannot reuse it."""
    return f"[{random.getrandbits(64):016x}] " + " ".join(["hello"] * n_words)


def pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, round(q * (len(xs) - 1)))]
