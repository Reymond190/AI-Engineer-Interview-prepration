"""
00 - GPU memory math: what fits in 16 GB, and why
=================================================

Interview topics
----------------
- Training memory = weights + gradients + optimizer states + activations
  (+ temporary buffers, + CUDA context / allocator fragmentation).
- Bytes per parameter, mixed-precision AdamW ("the 16 bytes rule"):
      fp32 master weights 4 + fp32 grads 4 + Adam m 4 + Adam v 4 = 16 B/param
  (with bf16 weights + fp32 states it's 2 + 2 + 8 (+4 master) = 12-16 B/param).
  => full fine-tuning a 9B model needs ~144 GB before activations. Not on 16 GB.
- Inference needs only the weights (+ KV cache): 2 B/param bf16, ~0.5 B/param int4.
- LoRA: frozen base in bf16 (2 B/param) + tiny trainable adapter with its own
  grads/states. QLoRA: base in 4-bit NF4 (~0.5 B/param + quant constants).
- Activations scale with batch x seq_len x hidden x layers. Rough per-layer
  rule (Korthikanti et al. 2022): ~34*s*b*h bytes + 5*a*s^2*b for attention
  scores; FlashAttention/SDPA removes the s^2 term (never materialised).
- The LOGITS are a hidden monster for big vocabularies: b*s*V floats. Qwen3's
  V=151,936 => one 8x512 batch of fp32 logits = 2.5 GB. Fused/chunked
  cross-entropy (lesson 14) avoids it.
- Levers, in order of usual impact: LoRA/QLoRA, gradient checkpointing
  (recompute activations: ~sqrt memory for ~30% more compute), smaller micro-
  batch + gradient accumulation, 8-bit/paged optimizers, fused CE.
- torch.cuda.max_memory_allocated (tensors) < memory_reserved (allocator
  cache) < nvidia-smi (also CUDA context ~0.5 GB). Quote the right one.

Run:  python finetune_course/00_memory_math.py [--quick]
"""

from __future__ import annotations

import torch

from common import BASE_SMALL, load_model, parse_args, peak_gb, reset_peak, section, table

GB = 1024**3


def estimate(n_params: float, n_trainable: float, weight_bytes: float, trainable_bytes: int = 16) -> float:
    """Static memory (no activations): frozen weights + trainable weights/grads/Adam states."""
    return (n_params * weight_bytes + n_trainable * trainable_bytes) / GB


def main() -> None:
    args = parse_args(__doc__)

    section("1. Static memory estimates (no activations)")
    rows = []
    for name, n in [("Qwen3-0.6B", 0.6e9), ("Qwen3-4B", 4.0e9), ("Qwen3.5-9B", 9.0e9), ("Llama-70B", 70e9)]:
        lora = n * 0.005  # r=16 on all linear layers is ~0.3-1% of params
        rows.append({
            "model": name,
            "infer bf16": f"{n * 2 / GB:.1f} GB",
            "infer int4": f"{n * 0.55 / GB:.1f} GB",
            "full FT (16B/p)": f"{estimate(0, n, 0):.0f} GB",
            "LoRA bf16": f"{estimate(n, lora, 2):.1f} GB",
            "QLoRA nf4": f"{estimate(n, lora, 0.55):.1f} GB",
        })
    table(rows, list(rows[0]))
    print("\n=> On 16 GB: full FT only for <~0.7B models, LoRA up to ~5B, QLoRA up to ~14B (short sequences).")

    section("2. Measure a real full fine-tune step of Qwen3-0.6B (fp32 master weights + AdamW)")
    reset_peak()
    model, tok = load_model(BASE_SMALL, dtype=torch.float32)
    model.config.use_cache = False
    n = sum(p.numel() for p in model.parameters())
    w = torch.cuda.memory_allocated() / GB
    print(f"params={n / 1e9:.3f}B  weights fp32: estimate {n * 4 / GB:.2f} GB, measured {w:.2f} GB")

    b, s = (4, 256) if args.quick else (8, 512)
    x = torch.randint(0, tok.vocab_size, (b, s), device="cuda")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)

    reset_peak()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(input_ids=x, labels=x)
    after_fwd = torch.cuda.memory_allocated() / GB
    out.loss.backward()
    after_bwd = torch.cuda.memory_allocated() / GB
    peak_fwd_bwd = peak_gb()
    opt.step()
    after_step = torch.cuda.memory_allocated() / GB
    cfg = model.config
    logits_gb = b * s * cfg.vocab_size * 4 / GB
    act_est = cfg.num_hidden_layers * 34 * s * b * cfg.hidden_size / GB + logits_gb

    print(f"batch={b}x{s} tokens")
    print(f"after forward   : {after_fwd:.2f} GB  (weights + saved activations; est. activations "
          f"{act_est:.2f} GB, of which fp32 logits {logits_gb:.2f} GB)")
    print(f"after backward  : {after_bwd:.2f} GB  (activations freed, + grads {n * 4 / GB:.2f} GB)")
    print(f"after opt.step(): {after_step:.2f} GB  (+ Adam m,v {n * 8 / GB:.2f} GB)")
    print(f"peak fwd+bwd    : {peak_fwd_bwd:.2f} GB")
    est_static = n * 16 / GB
    print(f"[memory] static estimate 16 B/param = {est_static:.2f} GB, measured after step = {after_step:.2f} GB, "
          f"ratio={after_step / est_static:.2f}")
    print("\nTakeaway: the 16 B/param rule predicts steady-state memory almost exactly;")
    print("the peak is higher because activations + logits live at the same time as weights.")


if __name__ == "__main__":
    main()
