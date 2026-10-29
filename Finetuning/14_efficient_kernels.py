"""
14 - Making training fast and small: kernels, packing, checkpointing, optimizers
================================================================================

Every row below trains the SAME LoRA SFT on the SAME examples for one epoch,
changing ONE thing, and reports real (non-padding) tokens/s and peak VRAM.
Numbers are from this GPU; the trade-offs are what interviewers want.

Interview topics
----------------
- Attention kernels: eager (materialises the s x s score matrix) vs SDPA
  (PyTorch picks a FlashAttention/memory-efficient kernel) vs FlashAttention-2/3
  (tiling in SRAM, never writes scores to HBM: O(s) memory, 2-4x faster).
  FA3 targets Hopper; on Blackwell consumer cards SDPA/FA2 are what you get.
- Liger kernels (LinkedIn, 2024): Triton fused RMSNorm, RoPE, SwiGLU and,
  most importantly, FUSED LINEAR CROSS-ENTROPY: computes lm_head + CE in
  chunks so the b*s*V logits tensor never exists. With a 152k vocab the
  logits are often the single largest activation. One flag:
  `use_liger_kernel=True`. Similar idea: Apple's Cut Cross-Entropy, Unsloth.
- Packing / padding-free: concatenate examples into long rows, reset
  position_ids per example, and use varlen attention (FA2 cu_seqlens) or a
  block-diagonal mask (SDPA) so examples don't see each other. Removes
  padding compute; biggest win on variable-length chat data. TRL's "bfd"
  (best-fit decreasing) packing avoids truncating examples.
- Gradient checkpointing: recompute activations in backward. Trades ~20-35%
  speed for a big activation saving. Activation OFFLOADING moves saved
  activations to CPU RAM instead (overlapped with compute via streams).
- 8-bit optimizers (bitsandbytes): Adam m/v stored in 8-bit block-wise
  quantisation, 8 B/param -> 2 B/param. Matters for full FT; with LoRA the
  states are already tiny. Paged = spill to CPU instead of OOM.
- torch.compile: graph capture + fused Triton kernels. First steps are
  slow (compilation); pays off on long runs with static shapes; dynamic
  sequence lengths cause recompiles (pad_to_multiple_of helps).
- What to reach for first on one 16 GB GPU: LoRA/QLoRA -> Liger -> packing
  -> grad checkpointing only if still OOM -> 8-bit optim for full FT.

Run:  python finetune_course/14_efficient_kernels.py [--quick]
"""

from __future__ import annotations

import importlib.util
import time

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import Trainer, TrainingArguments
from trl import SFTConfig, SFTTrainer

from common import (BASE_SMALL, LORA_TARGETS, encode_all, eval_tickets, free_gpu, load_model, load_tokenizer, out_dir,
                    pad_collate, parse_args, peak_gb, reset_peak, section, sft_records, table)
from data import ticket_splits


def run_sft(name: str, records: list[dict], n_tokens: int, test, *, attn: str = "sdpa", evaluate: bool = False,
            **overrides) -> dict:
    model, tok = load_model(BASE_SMALL, attn=attn)
    model.config.use_cache = False
    kw = dict(output_dir=str(out_dir("_tmp_14")), per_device_train_batch_size=32, gradient_accumulation_steps=1,
              learning_rate=2e-4, lr_scheduler_type="cosine", warmup_steps=0.1, num_train_epochs=1, bf16=True,
              max_length=512, logging_steps=50, save_strategy="no", report_to="none", seed=0)
    kw.update(overrides)
    lora = LoraConfig(r=16, lora_alpha=32, target_modules=LORA_TARGETS, task_type="CAUSAL_LM")
    row: dict = {"config": name}
    try:
        trainer = SFTTrainer(model=model, args=SFTConfig(**kw), train_dataset=Dataset.from_list(records),
                             processing_class=tok, peft_config=lora)
        reset_peak()
        t0 = time.time()
        out = trainer.train()
        dt = time.time() - t0
        row.update({"tokens/s": f"{n_tokens / dt:,.0f}", "peak GB": f"{peak_gb():.2f}",
                    "train loss": f"{out.training_loss:.4f}", "steps": trainer.state.global_step, "_tps": n_tokens / dt})
        if evaluate:
            trainer.model.config.use_cache = True
            row["accuracy"] = f"{eval_tickets(trainer.model, tok, test, name)['exact']:.3f}"
        del trainer
    except Exception as e:  # noqa: BLE001  a technique unsupported on this GPU/stack is a result, not a crash
        row.update({"tokens/s": "-", "peak GB": "-", "train loss": f"FAILED: {type(e).__name__}: {str(e)[:60]}"})
    del model
    free_gpu()
    print(f"[ablation] {row}")
    return row


def run_trainer_logits(name: str, rows_enc: list[dict], n_tokens: int, liger: bool) -> dict:
    """Plain HF Trainer + PEFT: the model's forward materialises the full b*s*V logits (unless Liger)."""
    from peft import get_peft_model
    model, tok = load_model(BASE_SMALL)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, target_modules=LORA_TARGETS, task_type="CAUSAL_LM"))
    args = TrainingArguments(output_dir=str(out_dir("_tmp_14")), per_device_train_batch_size=32, num_train_epochs=1,
                             learning_rate=2e-4, bf16=True, use_liger_kernel=liger, save_strategy="no",
                             report_to="none", logging_steps=1000, remove_unused_columns=False, seed=0)
    trainer = Trainer(model=model, args=args, train_dataset=Dataset.from_list(rows_enc),
                      data_collator=pad_collate(tok.pad_token_id))
    reset_peak()
    t0 = time.time()
    trainer.train()
    row = {"config": name, "tokens/s": f"{n_tokens / (time.time() - t0):,.0f}", "peak GB": f"{peak_gb():.2f}"}
    del trainer, model
    free_gpu()
    print(f"[logits] {row}")
    return row


def run_full_ft_optim(optim: str, rows_enc: list[dict], steps: int) -> dict:
    model, tok = load_model(BASE_SMALL, dtype=torch.float32)
    model.config.use_cache = False
    args = TrainingArguments(output_dir=str(out_dir("_tmp_14")), per_device_train_batch_size=4, max_steps=steps,
                             learning_rate=1e-5, bf16=True, optim=optim, save_strategy="no", report_to="none",
                             logging_steps=steps, remove_unused_columns=False, seed=0)
    row: dict = {"optimizer (full FT 0.6B)": optim}
    try:
        trainer = Trainer(model=model, args=args, train_dataset=Dataset.from_list(rows_enc),
                          data_collator=pad_collate(tok.pad_token_id))
        reset_peak()
        out = trainer.train()
        row.update({"peak GB": f"{peak_gb():.2f}", "train loss": f"{out.training_loss:.4f}"})
        del trainer
    except Exception as e:  # noqa: BLE001  report and continue
        row.update({"peak GB": "-", "train loss": f"FAILED: {type(e).__name__}"})
    del model
    free_gpu()
    return row


def main() -> None:
    args = parse_args(__doc__)
    splits = ticket_splits()
    n = 1200 if args.quick else 2000  # runs must last long enough that timing isn't noise
    test = splits["test"][:40]
    tok = load_tokenizer(BASE_SMALL)
    records = sft_records(tok, splits["train"][:n])
    n_tokens = sum(len(tok(r["prompt"] + r["completion"])["input_ids"]) for r in records)
    print(f"{n} examples, {n_tokens:,} real tokens per epoch")
    print("warm-up run (discarded: first CUDA run pays kernel compilation / allocator growth)")
    run_sft("warm-up", records[:64], n_tokens, test)

    section("1. The logits problem: plain Trainer vs Liger vs TRL's fused LM head")
    enc = encode_all(tok, splits["train"][:n])
    lrows = [run_trainer_logits("Trainer (full logits)", enc, n_tokens, liger=False),
             run_trainer_logits("Trainer + Liger (fused linear CE)", enc, n_tokens, liger=True)]
    table(lrows, ["config", "tokens/s", "peak GB"])
    print(f"[liger] peak {lrows[0]['peak GB']} -> {lrows[1]['peak GB']} GB")
    print("TRL >= 1.x SFTTrainer ALREADY computes the loss with a chunked 'fused LM head' (it never builds the")
    print("logits) and disables Liger's fused CE. So below, Liger only adds the RMSNorm/RoPE/SwiGLU fusions.")
    print("Read your trainer's source: the 'free' optimisation may already be on.")

    section("2. One change at a time (SFTTrainer, LoRA r=16, batch 32, 1 epoch)")
    rows = [run_sft("baseline: sdpa, padded", records, n_tokens, test, evaluate=True),
            run_sft("eager attention", records, n_tokens, test, attn="eager"),
            run_sft("+ liger kernels", records, n_tokens, test, use_liger_kernel=True),
            run_sft("+ packing (bfd, padding-free)", records, n_tokens, test, packing=True, max_length=4096,
                    per_device_train_batch_size=1),
            run_sft("+ gradient checkpointing", records, n_tokens, test, gradient_checkpointing=True,
                    gradient_checkpointing_kwargs={"use_reentrant": False}),
            run_sft("+ activation offloading", records, n_tokens, test, activation_offloading=True),
            run_sft("+ torch.compile", records, n_tokens, test, torch_compile=True, pad_to_multiple_of=64)]
    if importlib.util.find_spec("flash_attn"):
        rows.append(run_sft("flash_attention_2", records, n_tokens, test, attn="flash_attention_2"))
    else:
        print("(flash-attn not installed: no Blackwell wheel; SDPA already uses a flash-style kernel)")
    rows.append(run_sft("ALL: liger + packing", records, n_tokens, test, evaluate=True, use_liger_kernel=True,
                        packing=True, max_length=4096, per_device_train_batch_size=1))
    table(rows, ["config", "tokens/s", "peak GB", "train loss", "steps", "accuracy"])
    base = rows[0]
    for r in rows[1:]:
        if "_tps" in r and "_tps" in base:
            print(f"  {r['config']:32s} speed x{r['_tps'] / base['_tps']:.2f}  "
                  f"memory {float(r['peak GB']) - float(base['peak GB']):+.2f} GB")

    section("3. Optimizer state memory, full fine-tune (where it actually matters)")
    orows = [run_full_ft_optim(o, enc[:200], 10) for o in ("adamw_torch", "adamw_bnb_8bit", "paged_adamw_8bit")]
    table(orows, ["optimizer (full FT 0.6B)", "peak GB", "train loss"])
    print("Expected: 8-bit states save ~6 B/param (0.6B -> ~3.3 GB) at near-identical loss.")


if __name__ == "__main__":
    main()
