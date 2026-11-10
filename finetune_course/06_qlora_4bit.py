"""
06 - QLoRA: fine-tune a 4B model in 4-bit on a 16 GB card
=========================================================

Interview topics
----------------
- QLoRA (Dettmers et al. 2023) = frozen base quantized to 4-bit + LoRA
  adapters in bf16, gradients flow THROUGH the dequantised weights into the
  adapters. Matched 16-bit full FT on their benchmarks.
- Three ideas:
    1. NF4 (NormalFloat4): 16 quantile levels of a standard normal, since
       pretrained weights are ~normally distributed => lower error than
       uniform int4/fp4 at the same 4 bits. Block-wise (64 weights) absmax
       scaling.
    2. Double quantization: the per-block fp32 scales are themselves
       quantized to 8-bit (saves ~0.37 bit/param, ~3 GB on 65B).
    3. Paged optimizers: optimizer states in CUDA unified memory, paged to
       CPU RAM on spikes instead of OOM.
- Compute dtype bf16: weights are stored in 4-bit but dequantised to bf16
  per matmul. That is why QLoRA is ~20-40% SLOWER than bf16 LoRA. You trade
  speed for memory.
- prepare_model_for_kbit_training: casts norms/embeddings to fp32 for
  stability, enables gradient checkpointing, and makes input embeddings
  require grad (needed so checkpointed blocks have a grad path).
- Gradient checkpointing: keep only block inputs, recompute the rest in
  backward. Activations O(layers) -> O(sqrt) memory for ~30% more compute.
  `use_cache=False` is required while training with it.
- Merging a QLoRA adapter: merge into the bf16 (de-quantised) base, not
  into 4-bit weights, then re-quantise for serving (AWQ/GPTQ) if needed.

Run:  python finetune_course/06_qlora_4bit.py [--quick]
Out:  finetune_course/outputs/06_qlora_4b  (teacher in lesson 12)
"""

from __future__ import annotations

import torch
from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer

from common import (BASE_QLORA, LORA_TARGETS, eval_tickets, free_gpu, load_model, out_dir, parse_args, peak_gb,
                    reset_peak, section, sft_records, table)
from data import ticket_splits


def quant_error_demo() -> None:
    """NF4 vs FP4 reconstruction error on a normally distributed 'weight' matrix."""
    import bitsandbytes.functional as bnbf
    w = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16) * 0.02
    for qt in ("nf4", "fp4"):
        q, state = bnbf.quantize_4bit(w, quant_type=qt, blocksize=64)
        err = (bnbf.dequantize_4bit(q, state) - w).float().pow(2).mean().sqrt().item()
        print(f"  {qt}: rmse={err:.2e}  (relative {err / 0.02:.3f})")
    q8, s8 = bnbf.quantize_blockwise(w.float())
    err8 = (bnbf.dequantize_blockwise(q8, s8) - w.float()).pow(2).mean().sqrt().item()
    print(f"  int8 blockwise: rmse={err8:.2e}  (relative {err8 / 0.02:.3f})")


def main() -> None:
    args = parse_args(__doc__)
    splits = ticket_splits()
    test = splits["test"][:60] if args.quick else splits["test"]

    section("1. Quantisation error: why NF4")
    quant_error_demo()

    section("2. Weight memory: bf16 vs NF4 (Qwen3-4B)")
    rows = []
    for label, q4 in [("bf16", False), ("nf4 + double quant", True)]:
        reset_peak()
        m, _ = load_model(BASE_QLORA, quant4=q4)
        rows.append({"weights": label, "GB": f"{torch.cuda.memory_allocated() / 1024**3:.2f}",
                     "footprint (HF)": f"{m.get_memory_footprint() / 1024**3:.2f}"})
        del m
        free_gpu()
    table(rows, ["weights", "GB", "footprint (HF)"])

    section("3. QLoRA fine-tune")
    model, tok = load_model(BASE_QLORA, quant4=True)
    base = eval_tickets(model, tok, test, "4B base")
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.config.use_cache = False
    cfg = SFTConfig(
        output_dir=str(out_dir("_tmp_06")),
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=0.1,
        max_steps=40 if args.quick else 150,
        bf16=True,
        max_length=512,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        seed=0,
    )
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=LORA_TARGETS, task_type="CAUSAL_LM")
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=Dataset.from_list(sft_records(tok, splits["train"])),
                         processing_class=tok, peft_config=lora)
    trainer.model.print_trainable_parameters()
    reset_peak()
    out = trainer.train()
    print(f"[qlora] peak={peak_gb():.1f} GB  time={out.metrics['train_runtime']:.0f}s  "
          f"({out.metrics['train_samples_per_second']:.1f} samples/s)")
    trainer.model.config.use_cache = True
    m = eval_tickets(trainer.model, tok, test, "4B QLoRA", show=2)
    print(f"[result] accuracy {base['exact']:.3f} -> {m['exact']:.3f}")
    save = out_dir("06_qlora_4b")
    trainer.model.save_pretrained(save)
    tok.save_pretrained(save)
    print(f"saved adapter to {save}")


if __name__ == "__main__":
    main()
