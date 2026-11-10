"""
16 - Beyond LoRA: full-parameter training in less memory, FP8, NEFTune
======================================================================

Interview topics
----------------
- LoRA restricts the UPDATE to low rank. The alternatives keep full-rank
  weight updates but shrink what dominates memory, the OPTIMIZER STATE:
    * GaLore (Zhao et al. 2024): project each gradient matrix onto its top-r
      singular subspace (refreshed every T steps), keep Adam m/v in that
      r-dim space, project the update back. Full-parameter learning with
      LoRA-like optimizer memory. Cost: periodic SVDs, slower steps.
    * APOLLO (2025): even cheaper: random projections to estimate
      channel-wise learning-rate scaling; SGD-like memory with AdamW-like
      results. (optional dep `apollo-torch`)
- Spectrum (Hartford et al. 2024): rank layers by signal-to-noise ratio of
  their singular values (random-matrix theory: singular values below the
  Marchenko-Pastur / Gavish-Donoho threshold are "noise"), fully train only
  the top ~25% highest-SNR layers, freeze the rest. Competitive with full FT
  at a fraction of the memory, no adapter to merge.
- FP8 training (torchao `convert_to_float8_training`): matmuls in float8
  (e4m3 forward, e5m2 grads) with dynamic per-tensor scaling, master weights
  in bf16/fp32. Up to ~1.3-1.5x on big matmuls (H100/Blackwell tensor cores),
  but small models are memory/launch bound and may see NO gain: measure.
  Blackwell adds FP4 (NVFP4/MXFP4) tensor cores; FP4 training is research-stage.
- NEFTune (Jain et al. 2023): add uniform noise to input embeddings during
  SFT (alpha/sqrt(L*d)). A regulariser: helps generalisation of chat
  quality; one flag `neftune_noise_alpha=5`. Here we test it on HELD-OUT
  phrasings, which is exactly what a regulariser should help.

Run:  python finetune_course/16_memory_frontier.py [--quick]
"""

from __future__ import annotations

import importlib.util
import re
import time

import torch
from torch import nn
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import Trainer, TrainingArguments

from common import (BASE_SMALL, LORA_TARGETS, count_params, encode_all, eval_tickets, free_gpu, load_model, load_tokenizer,
                    out_dir, pad_collate, parse_args, peak_gb, reset_peak, section, table)
from data import ticket_splits


def train_eval(name: str, model, tok, rows, test, *, steps: int, lr: float, **targs) -> dict:
    model.config.use_cache = False
    trainable, total = count_params(model)
    args = TrainingArguments(output_dir=str(out_dir("_tmp_16")), per_device_train_batch_size=4,
                             gradient_accumulation_steps=4, learning_rate=lr, lr_scheduler_type="cosine",
                             warmup_steps=0.1, max_steps=steps, bf16=True, logging_steps=steps, save_strategy="no",
                             report_to="none", remove_unused_columns=False, seed=0, **targs)
    row: dict = {"method": name, "trainable": f"{100 * trainable / total:.1f}%"}
    try:
        trainer = Trainer(model=model, args=args, train_dataset=Dataset.from_list(rows),
                          data_collator=pad_collate(tok.pad_token_id, multiple=16))
        reset_peak()
        t0 = time.time()
        out = trainer.train()
        row.update({"peak GB": f"{peak_gb():.2f}", "train s": f"{time.time() - t0:.0f}",
                    "train loss": f"{out.training_loss:.4f}"})
        model.config.use_cache = True
        row["accuracy"] = f"{eval_tickets(model, tok, test, name)['exact']:.3f}"
        del trainer
    except Exception as e:  # noqa: BLE001  unsupported technique on this stack = a reported result
        row.update({"peak GB": "-", "train loss": f"FAILED: {type(e).__name__}: {str(e)[:70]}"})
    print(f"[frontier] {row}")
    return row


def gavish_donoho_snr(w: torch.Tensor) -> float:
    """Spectrum-style SNR: singular-value mass above the optimal hard threshold / mass below it."""
    s = torch.linalg.svdvals(w.float())
    m, n = w.shape
    beta = min(m, n) / max(m, n)
    omega = 0.56 * beta**3 - 0.95 * beta**2 + 1.82 * beta + 1.43  # Gavish & Donoho 2014, unknown noise
    tau = omega * s.median()
    return (s[s > tau].sum() / s[s <= tau].sum().clamp_min(1e-8)).item()


def spectrum_unfreeze(model, top_frac: float = 0.25) -> list[str]:
    """Freeze everything, then unfreeze the top-SNR fraction of each projection type."""
    for p in model.parameters():
        p.requires_grad_(False)
    by_type: dict[str, list[tuple[float, str, nn.Linear]]] = {}
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and name.split(".")[-1] in LORA_TARGETS:
            by_type.setdefault(name.split(".")[-1], []).append((gavish_donoho_snr(mod.weight), name, mod))
    chosen = []
    for items in by_type.values():
        items.sort(key=lambda x: -x[0])
        for _, name, mod in items[: max(1, int(len(items) * top_frac))]:
            mod.weight.requires_grad_(True)
            chosen.append(name)
    return chosen


def main() -> None:
    args = parse_args(__doc__)
    splits = ticket_splits()
    test = splits["test"][:60] if args.quick else splits["test"]
    steps = 30 if args.quick else 100
    rows_enc = encode_all(load_tokenizer(BASE_SMALL), splits["train"])
    results = []

    section("1. Full fine-tune baseline (fp32 master weights + AdamW)")
    model, tok = load_model(BASE_SMALL, dtype=torch.float32)
    results.append(train_eval("full FT AdamW", model, tok, rows_enc, test, steps=steps, lr=1e-5))
    del model
    free_gpu()

    section("2. GaLore: full-parameter, low-rank optimizer state")
    if importlib.util.find_spec("galore_torch"):
        model, tok = load_model(BASE_SMALL, dtype=torch.float32)
        results.append(train_eval("GaLore r=128", model, tok, rows_enc, test, steps=steps, lr=1e-4,
                                  optim="galore_adamw", optim_target_modules=[r".*\.self_attn\..*", r".*\.mlp\..*"],
                                  optim_args="rank=128, update_proj_gap=50, scale=0.25"))
        del model
        free_gpu()
    else:
        print("(galore-torch not installed)")

    section("3. APOLLO")
    if importlib.util.find_spec("apollo_torch"):
        model, tok = load_model(BASE_SMALL, dtype=torch.float32)
        results.append(train_eval("APOLLO r=128", model, tok, rows_enc, test, steps=steps, lr=1e-4,
                                  optim="apollo_adamw", optim_target_modules=[r".*\.self_attn\..*", r".*\.mlp\..*"],
                                  optim_args="proj=random, rank=128, scale=1, update_proj_gap=50"))
        del model
        free_gpu()
    else:
        print("(apollo-torch not installed: `pip install apollo-torch` to include it)")

    section("4. Spectrum: train only the top-25% SNR layers")
    model, tok = load_model(BASE_SMALL, dtype=torch.float32)
    t0 = time.time()
    chosen = spectrum_unfreeze(model, 0.25)
    layers = sorted({int(m.group(1)) for c in chosen if (m := re.search(r"layers\.(\d+)\.", c))})
    print(f"SNR scan {time.time() - t0:.1f}s; unfroze {len(chosen)} matrices; layers touched: {layers}")
    results.append(train_eval("Spectrum 25%", model, tok, rows_enc, test, steps=steps, lr=2e-5))
    del model
    free_gpu()

    section("5. LoRA reference point")
    model, tok = load_model(BASE_SMALL)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, target_modules=LORA_TARGETS, task_type="CAUSAL_LM"))
    results.append(train_eval("LoRA r=16", model, tok, rows_enc, test, steps=steps, lr=2e-4))
    del model
    free_gpu()

    section("6. FP8 matmuls with torchao (bf16 weights, full FT) vs plain bf16")
    for label, fp8 in [("bf16 full FT", False), ("fp8 (torchao) full FT", True)]:
        model, tok = load_model(BASE_SMALL, dtype=torch.bfloat16)
        if fp8:
            try:
                from torchao.float8 import convert_to_float8_training
                convert_to_float8_training(model, module_filter_fn=lambda m, fqn: isinstance(m, nn.Linear)
                                           and "lm_head" not in fqn and m.in_features % 16 == 0
                                           and m.out_features % 16 == 0)
            except Exception as e:  # noqa: BLE001  report and skip
                print(f"(fp8 conversion failed: {e})")
                del model
                free_gpu()
                continue
        results.append(train_eval(label, model, tok, rows_enc, test, steps=steps, lr=1e-5))
        del model
        free_gpu()

    section("7. NEFTune noise (LoRA SFT, alpha=5) - evaluated on held-out phrasings")
    for alpha in (None, 5.0):
        model, tok = load_model(BASE_SMALL)
        model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, target_modules=LORA_TARGETS, task_type="CAUSAL_LM"))
        kw = {"neftune_noise_alpha": alpha} if alpha else {}
        results.append(train_eval(f"LoRA + NEFTune {alpha}" if alpha else "LoRA (no NEFTune)", model, tok, rows_enc,
                                  test, steps=steps, lr=2e-4, **kw))
        del model
        free_gpu()

    section("Summary")
    table(results, ["method", "trainable", "peak GB", "train s", "train loss", "accuracy"])
    print("\nCompare 'full FT AdamW' (fp32 master weights) with 'bf16 full FT' (weights AND updates in bf16):")
    print("at lr 1e-5 most updates are smaller than bf16's precision near each weight, so they round to zero.")
    print("That is why mixed precision keeps fp32 master weights (or uses stochastic rounding / Kahan summation).")


if __name__ == "__main__":
    main()
