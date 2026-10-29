"""
03 - Full fine-tuning with the HF Trainer, and what overfitting looks like
==========================================================================

Interview topics
----------------
- Trainer = lesson 02's loop + mixed precision, grad accumulation (with the
  correct token-count normalisation), LR schedules, logging, evaluation,
  checkpointing/resume, distributed (DDP/FSDP/DeepSpeed via accelerate).
- Key TrainingArguments: per_device_train_batch_size,
  gradient_accumulation_steps, learning_rate, lr_scheduler_type,
  warmup_steps (a float <1 = ratio in transformers 5), num_train_epochs / max_steps, bf16, eval_strategy,
  save_strategy, gradient_checkpointing, optim, weight_decay, max_grad_norm.
- `remove_unused_columns=False` when you pass pre-tokenised dicts with a
  custom collator (otherwise Trainer drops your `labels`).
- Full FT: every weight moves. Best quality ceiling, highest memory
  (16 B/param), one full checkpoint per experiment (1.2 GB here, 18 GB for 9B),
  and the most catastrophic forgetting (lesson 07).
- Overfitting signature: train loss -> 0 while VAL loss bottoms out then RISES.
  Fixes: fewer epochs (1-3 is typical for SFT), more/diverse data, lower LR,
  early stopping on val loss / task metric, LoRA (fewer free parameters).
- Val loss is a proxy. Always also measure the TASK metric (here exact-match
  JSON); they can disagree (lower loss, worse format).

Run:  python finetune_course/03_full_finetune_trainer.py [--quick]
Out:  finetune_course/outputs/03_full_ft  (used by lesson 07)
"""

from __future__ import annotations

import torch
from datasets import Dataset
from transformers import Trainer, TrainingArguments

from common import (BASE_SMALL, encode_all, eval_tickets, free_gpu, load_model, out_dir, pad_collate,
                    parse_args, peak_gb, reset_peak, section)
from data import ticket_splits


def train(model, tok, train_rows, val_rows, *, name: str, max_steps: int, lr: float, eval_steps: int) -> Trainer:
    args = TrainingArguments(
        output_dir=str(out_dir(f"_tmp_{name}")),
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_steps=0.1,
        max_steps=max_steps,
        bf16=True,                    # fp32 master weights + bf16 autocast
        logging_steps=eval_steps,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="no",           # we save the final model ourselves
        report_to="none",
        remove_unused_columns=False,
        seed=0,
    )
    trainer = Trainer(model=model, args=args, train_dataset=Dataset.from_list(train_rows),
                      eval_dataset=Dataset.from_list(val_rows), data_collator=pad_collate(tok.pad_token_id))
    trainer.train()
    return trainer


def curve(trainer: Trainer) -> list[tuple[int, float | None, float | None]]:
    """(step, train_loss, eval_loss) from the trainer's log history."""
    by_step: dict[int, dict] = {}
    for h in trainer.state.log_history:
        by_step.setdefault(h["step"], {}).update(h)
    return [(s, d.get("loss"), d.get("eval_loss")) for s, d in sorted(by_step.items()) if "eval_loss" in d]


def main() -> None:
    args = parse_args(__doc__)
    splits = ticket_splits()
    test = splits["test"][:60] if args.quick else splits["test"]

    section("1. Full fine-tune Qwen3-0.6B on 2000 tickets")
    model, tok = load_model(BASE_SMALL, dtype=torch.float32)
    model.config.use_cache = False
    train_rows, val_rows = encode_all(tok, splits["train"]), encode_all(tok, splits["val"])
    before = eval_tickets(model, tok, test, "before")
    reset_peak()
    steps = 40 if args.quick else 250
    trainer = train(model, tok, train_rows, val_rows, name="full", max_steps=steps, lr=1e-5, eval_steps=max(10, steps // 5))
    print(f"[train] peak={peak_gb():.1f} GB")
    for s, tr, ev in curve(trainer):
        print(f"  step {s:4d}  train_loss={tr if tr is None else f'{tr:.4f}'}  val_loss={ev:.4f}")
    after = eval_tickets(model, tok, test, "after full FT", show=2)
    print(f"[result] accuracy {before['exact']:.3f} -> {after['exact']:.3f}")

    save = out_dir("03_full_ft")
    model.to(torch.bfloat16).save_pretrained(save)
    tok.save_pretrained(save)
    print(f"saved full model to {save} (bf16)")
    del model, trainer
    free_gpu()

    section("2. Overfitting on purpose: 32 examples, many epochs")
    model, tok = load_model(BASE_SMALL, dtype=torch.float32)
    model.config.use_cache = False
    small = encode_all(tok, splits["train"][:32])
    trainer = train(model, tok, small, val_rows[:100], name="overfit", max_steps=60, lr=5e-5, eval_steps=6)
    pts = curve(trainer)
    for s, tr, ev in pts:
        print(f"  step {s:3d}  train_loss={tr if tr is None else f'{tr:.4f}'}  val_loss={ev:.4f}")
    best = min(pts, key=lambda p: p[2])
    print(f"[overfit] best val_loss {best[2]:.4f} at step {best[0]}, final {pts[-1][2]:.4f} "
          f"(final train_loss {pts[-1][1]})")
    print("Early stopping would have kept the checkpoint at the minimum (load_best_model_at_end=True).")


if __name__ == "__main__":
    main()
