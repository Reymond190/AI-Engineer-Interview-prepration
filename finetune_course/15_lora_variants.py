"""
15 - LoRA variants: rsLoRA, DoRA, PiSSA, OLoRA, LoRA+, (IA)^3
=============================================================

Same data, same steps, same lr; only the adapter changes. Reports accuracy,
trainable params, time and memory so you can say what each one BUYS.

Interview topics
----------------
- rsLoRA (Kalajievski 2023): scale alpha/sqrt(r) instead of alpha/r. With
  alpha/r, large ranks get a vanishing update and learn no better than
  small ones; rank-stabilised scaling makes higher r actually help.
  `use_rslora=True`.
- DoRA (Liu et al. 2024): decompose W = m * V/||V|| into magnitude m
  (trainable vector) and direction V (LoRA-updated). Closer to full-FT
  learning dynamics, often +1-2 points at low rank; ~20-40% slower
  training (norm computation), no inference cost after merge. `use_dora=True`.
- PiSSA (Meng et al. 2024): init A,B from the TOP singular vectors of W
  (and subtract them from the frozen residual), so the adapter trains the
  principal directions. Converges faster. OLoRA: QR-based orthonormal init.
  Both change the base weights => convert before sharing the adapter.
- LoRA+ (Hayou et al. 2024): B gets a higher lr than A (ratio ~16). Theory:
  for wide models the equal-lr choice is sub-optimal. Faster convergence,
  free. `peft.optimizers.create_loraplus_optimizer`.
- LoftQ: quantisation-aware init for QLoRA (A,B initialised to cancel the
  4-bit quantisation error). EVA, CorDA: data-driven inits from
  activations. LoRA-FA: freeze A, train B (less activation memory).
- Non-LoRA PEFT for contrast: (IA)^3 scales activations with learned
  vectors (~0.01% params, very cheap, weaker). Prompt/prefix tuning:
  learn virtual tokens (weakest on small models).
- Practical defaults today: all-linear targets, r=16-64, alpha=r or 2r
  (or rsLoRA), lr 1e-4..2e-4; try DoRA/PiSSA if you need the last points.

Run:  python finetune_course/15_lora_variants.py [--quick]
"""

from __future__ import annotations

import time

import torch
from datasets import Dataset
from peft import IA3Config, LoraConfig, get_peft_model
from peft.optimizers import create_loraplus_optimizer
from trl import SFTConfig, SFTTrainer

from common import (BASE_SMALL, LORA_TARGETS, count_params, eval_tickets, free_gpu, load_model, load_tokenizer, out_dir,
                    parse_args, peak_gb, reset_peak, section, sft_records, table)
from data import ticket_splits


def run(name: str, peft_cfg, records, test, steps: int, lr: float = 2e-4, loraplus_ratio: float | None = None) -> dict:
    model, tok = load_model(BASE_SMALL)
    model.config.use_cache = False
    t0 = time.time()
    pm = get_peft_model(model, peft_cfg)  # PiSSA/OLoRA do their SVD/QR here
    init_s = time.time() - t0
    trainable, _ = count_params(pm)
    cfg = SFTConfig(output_dir=str(out_dir("_tmp_15")), per_device_train_batch_size=8, gradient_accumulation_steps=2,
                    learning_rate=lr, lr_scheduler_type="cosine", warmup_steps=0.1, max_steps=steps, bf16=True,
                    max_length=512, logging_steps=steps, save_strategy="no", report_to="none", seed=0)
    opts = (None, None)
    if loraplus_ratio:
        opts = (create_loraplus_optimizer(pm, optimizer_cls=torch.optim.AdamW, lr=lr, loraplus_lr_ratio=loraplus_ratio), None)
    trainer = SFTTrainer(model=pm, args=cfg, train_dataset=Dataset.from_list(records), processing_class=tok,
                         optimizers=opts)
    reset_peak()
    out = trainer.train()
    pm.config.use_cache = True
    acc = eval_tickets(pm, tok, test, name)["exact"]
    row = {"variant": name, "trainable": f"{trainable / 1e6:.2f}M", "accuracy": f"{acc:.3f}",
           "train loss": f"{out.training_loss:.4f}", "train s": f"{out.metrics['train_runtime']:.0f}",
           "init s": f"{init_s:.1f}", "peak GB": f"{peak_gb():.2f}"}
    print(f"[variant] {row}")
    del trainer, pm, model
    free_gpu()
    return row


def main() -> None:
    args = parse_args(__doc__)
    splits = ticket_splits()
    test = splits["test"][:60] if args.quick else splits["test"]
    steps = 30 if args.quick else 100
    records = sft_records(load_tokenizer(BASE_SMALL), splits["train"])

    def lora(**kw) -> LoraConfig:
        base = dict(r=16, lora_alpha=32, lora_dropout=0.0, target_modules=LORA_TARGETS, task_type="CAUSAL_LM")
        return LoraConfig(**{**base, **kw})

    section(f"LoRA variants, {steps} steps each")
    rows = [
        run("LoRA r=16", lora(), records, test, steps),
        run("LoRA attention-only", lora(target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]), records, test, steps),
        run("LoRA r=64 (alpha/r)", lora(r=64, lora_alpha=32), records, test, steps),
        run("rsLoRA r=64 (alpha/sqrt r)", lora(r=64, lora_alpha=32, use_rslora=True), records, test, steps),
        run("DoRA r=16", lora(use_dora=True), records, test, steps),
        run("PiSSA r=16", lora(init_lora_weights="pissa_niter_4"), records, test, steps, lr=5e-5),
        run("OLoRA r=16", lora(init_lora_weights="olora"), records, test, steps, lr=5e-5),
        run("LoRA+ r=16 (ratio 16)", lora(), records, test, steps, lr=5e-5, loraplus_ratio=16),
        run("(IA)^3", IA3Config(target_modules=["k_proj", "v_proj", "down_proj"], feedforward_modules=["down_proj"],
                                task_type="CAUSAL_LM"), records, test, steps, lr=3e-3),
    ]
    table(rows, ["variant", "trainable", "accuracy", "train loss", "train s", "init s", "peak GB"])
    print("\nNotes: PiSSA uses a lower lr (it starts from the principal subspace, big steps hurt);")
    print("(IA)^3 needs a much higher lr (it learns scaling vectors). With few steps the")
    print("convergence-speed variants (PiSSA, LoRA+) show their advantage most.")
    print("LoftQ is a QLoRA-time init: LoraConfig(init_lora_weights='loftq', loftq_config=LoftQConfig(loftq_bits=4)).")


if __name__ == "__main__":
    main()
