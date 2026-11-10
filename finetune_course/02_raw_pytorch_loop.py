"""
02 - The training loop by hand (what Trainer/SFTTrainer hide)
=============================================================

Interview topics
----------------
- Causal LM objective: predict token t+1 from tokens <= t. Logits at
  position i are scored against labels at i+1 => SHIFT by one:
      loss = CE(logits[:, :-1], labels[:, 1:], ignore_index=-100)
  HF models do this shift internally when you pass `labels=`.
- Loss averaging: mean over *trained* tokens in the batch. With gradient
  accumulation, naive per-micro-batch means over-weight short examples;
  correct is sum(loss)/total_trained_tokens across the accumulated batch
  (transformers fixed this in 2024: "gradient accumulation bug").
- Perplexity = exp(mean token CE). Only comparable with the same tokenizer.
- AdamW: Adam + decoupled weight decay. Usual SFT lr: ~1e-5 full FT,
  ~1e-4..2e-4 LoRA. Warmup (avoid early large updates while Adam's second
  moment is still noisy) then cosine/linear decay.
- Gradient accumulation: effective batch = micro_batch x accum_steps x GPUs.
  Same math as a big batch, less memory, same wall-clock per sample.
- Mixed precision: fp32 master weights + bf16 autocast for matmuls. bf16 has
  fp32's exponent range => no loss scaling needed (fp16 needs GradScaler).
- Gradient clipping (max_norm=1.0) caps rare huge updates (loss spikes).
- model.train() vs model.eval(): dropout (Qwen3 uses 0) and use_cache.

Run:  python finetune_course/02_raw_pytorch_loop.py [--quick]
"""

from __future__ import annotations

import math
import random
import time

import torch
import torch.nn.functional as F

from common import BASE_SMALL, encode_all, eval_tickets, load_model, pad_collate, parse_args, peak_gb, reset_peak, section
from data import ticket_splits


def lm_loss_sum(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Sum of token CE over trained positions, and how many there were."""
    logits = logits[:, :-1].float()
    labels = labels[:, 1:]
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100, reduction="sum")
    return loss, int((labels != -100).sum())


@torch.no_grad()
def eval_loss(model, batches) -> float:
    model.eval()
    total, n = 0.0, 0
    for b in batches:
        b = {k: v.cuda() for k, v in b.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits
        s, k = lm_loss_sum(logits, b["labels"])
        total, n = total + s.item(), n + k
    return total / n


def main() -> None:
    args = parse_args(__doc__)
    splits = ticket_splits()
    model, tok = load_model(BASE_SMALL, dtype=torch.float32)  # fp32 master weights
    model.config.use_cache = False
    collate = pad_collate(tok.pad_token_id)

    train = encode_all(tok, splits["train"])
    val_batches = [collate(encode_all(tok, splits["val"][i:i + 16])) for i in range(0, len(splits["val"]), 16)]
    test = splits["test"][:60] if args.quick else splits["test"]

    micro, accum = 4, 4
    steps = 40 if args.quick else 125  # 125 x 16 = one epoch of 2000
    warmup, lr_max = max(1, steps // 10), 1e-5

    section("1. Sanity: our loss == HF's built-in loss")
    b = {k: v.cuda() for k, v in collate(train[:4]).items()}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(**b)
    s, k = lm_loss_sum(out.logits, b["labels"])
    print(f"manual={s.item() / k:.4f}  hf={out.loss.item():.4f}")

    section("2. Before training")
    v0 = eval_loss(model, val_batches)
    print(f"[before] val_loss={v0:.3f} ppl={math.exp(v0):.1f}")
    before = eval_tickets(model, tok, test, "before")

    section(f"3. Train: {steps} steps, micro={micro} x accum={accum}, lr={lr_max} warmup={warmup} cosine")
    opt = torch.optim.AdamW(model.parameters(), lr=lr_max, weight_decay=0.0, betas=(0.9, 0.95))

    def lr_at(step: int) -> float:
        if step < warmup:
            return lr_max * (step + 1) / warmup
        return lr_max * 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, steps - warmup)))

    rng = random.Random(0)
    order = list(range(len(train)))
    rng.shuffle(order)
    pos = 0
    reset_peak()
    t0 = time.time()
    for step in range(steps):
        model.train()
        # Draw the whole accumulated batch first, so the loss is normalised by
        # the TOTAL number of trained tokens (the correct accumulation).
        micro_batches = []
        for _ in range(accum):
            if pos + micro > len(order):
                rng.shuffle(order)
                pos = 0
            micro_batches.append(collate([train[i] for i in order[pos:pos + micro]]))
            pos += micro
        n_tok = sum(int((mb["labels"][:, 1:] != -100).sum()) for mb in micro_batches)
        total = 0.0
        for mb in micro_batches:
            mb = {k: v.cuda() for k, v in mb.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids=mb["input_ids"], attention_mask=mb["attention_mask"]).logits
            s, _ = lm_loss_sum(logits, mb["labels"])
            (s / n_tok).backward()
            total += s.item()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % 10 == 0 or step == steps - 1:
            print(f"step {step:4d}  loss={total / n_tok:.4f}  grad_norm={gnorm:.2f}  lr={lr_at(step):.2e}")
    print(f"train time {time.time() - t0:.0f}s, peak {peak_gb():.1f} GB")

    section("4. After training")
    v1 = eval_loss(model, val_batches)
    print(f"[after] val_loss={v1:.3f} ppl={math.exp(v1):.2f}")
    after = eval_tickets(model, tok, test, "after", show=2)
    print(f"[result] accuracy {before['exact']:.3f} -> {after['exact']:.3f}, val_loss {v0:.3f} -> {v1:.3f}")


if __name__ == "__main__":
    main()
