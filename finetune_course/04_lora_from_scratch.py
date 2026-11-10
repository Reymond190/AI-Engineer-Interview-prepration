"""
04 - LoRA from scratch, then the same thing with PEFT
=====================================================

Interview topics
----------------
- LoRA (Hu et al. 2021): freeze W (d_out x d_in); learn a low-rank update
      W' = W + (alpha / r) * B @ A,   A: r x d_in,  B: d_out x r,  r << d
  Hypothesis: the fine-tuning update has low "intrinsic rank".
- Init: A random (Kaiming/Gaussian), B = 0 => B@A = 0 => at step 0 the
  model is EXACTLY the base model (training starts from the pretrained
  function, not a perturbed one).
- Scaling alpha/r: lets you change r without retuning lr. Common: alpha = r
  or 2r. rsLoRA (lesson 15) uses alpha/sqrt(r), which keeps big ranks stable.
- Trainable params per layer: r * (d_in + d_out) instead of d_in * d_out.
  For a 1024x1024 projection with r=16: 32k vs 1M (3%).
- Memory win comes from no grads/optimizer states for W. Activations are
  NOT reduced (forward still runs through full W), so long sequences still
  need gradient checkpointing.
- Merge for inference: W <- W + scale*B@A: zero added latency. Unmerged
  adapters cost one extra small matmul, but allow hot-swapping and
  multi-LoRA serving (lesson 13).
- target_modules: attention-only (original paper) vs all linear layers
  (QLoRA paper: needed to match full FT). More targets > higher rank,
  usually.
- lora_dropout regularises the adapter input; 0.0-0.1.

Run:  python finetune_course/04_lora_from_scratch.py [--quick]
"""

from __future__ import annotations

import math

import torch
from torch import nn
from datasets import Dataset
from transformers import Trainer, TrainingArguments

from common import (BASE_SMALL, LORA_TARGETS, count_params, encode_all, eval_tickets, free_gpu, load_model,
                    out_dir, pad_collate, parse_args, peak_gb, print_trainable, reset_peak, section, table)
from data import ticket_splits


class LoRALinear(nn.Module):
    """y = W x + (alpha/r) * B(A(dropout(x))), with W frozen."""

    def __init__(self, base: nn.Linear, r: int = 16, alpha: int = 32, dropout: float = 0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.scale = alpha / r
        self.A = nn.Parameter(torch.empty(r, base.in_features, dtype=torch.float32, device=base.weight.device))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, dtype=torch.float32, device=base.weight.device))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))  # same as nn.Linear / PEFT default
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Adapter in fp32 (cheap: it's tiny), base in bf16.
        delta = (self.drop(x).to(self.A.dtype) @ self.A.T @ self.B.T) * self.scale
        return self.base(x) + delta.to(x.dtype)

    @torch.no_grad()
    def merged(self) -> nn.Linear:
        lin = nn.Linear(self.base.in_features, self.base.out_features, bias=self.base.bias is not None,
                        device=self.base.weight.device, dtype=self.base.weight.dtype)
        lin.weight.copy_(self.base.weight + (self.scale * self.B @ self.A).to(lin.weight.dtype))
        if self.base.bias is not None:
            lin.bias.copy_(self.base.bias)
        return lin


def inject(model: nn.Module, targets: list[str], **kw) -> int:
    """Replace every nn.Linear whose name ends in a target with LoRALinear. Returns count."""
    for p in model.parameters():
        p.requires_grad_(False)
    hits = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear) and n.split(".")[-1] in targets]
    for name, mod in hits:
        parent, child = name.rsplit(".", 1)
        setattr(model.get_submodule(parent), child, LoRALinear(mod, **kw))
    return len(hits)


def merge_all(model: nn.Module) -> None:
    for name, mod in list(model.named_modules()):
        if isinstance(mod, LoRALinear):
            parent, child = name.rsplit(".", 1)
            setattr(model.get_submodule(parent), child, mod.merged())


def main() -> None:
    args = parse_args(__doc__)
    splits = ticket_splits()
    test = splits["test"][:60] if args.quick else splits["test"]

    section("1. Inject hand-written LoRA (r=16, alpha=32, all linear layers)")
    model, tok = load_model(BASE_SMALL)
    model.config.use_cache = False
    probe = tok(["The quick brown fox"], return_tensors="pt").to("cuda")
    with torch.no_grad():
        ref = model(**probe).logits
    n = inject(model, LORA_TARGETS, r=16, alpha=32)
    with torch.no_grad():
        same = (model(**probe).logits - ref).abs().max().item()
    print(f"wrapped {n} Linear layers; max |logit diff| vs base at init = {same:.2e} (B=0 => identical)")
    print_trainable(model, "hand LoRA")
    ours, _ = count_params(model)

    section("2. Train it (plain HF Trainer: it doesn't care that LoRA is hand-made)")
    targs = TrainingArguments(output_dir=str(out_dir("_tmp_04")), per_device_train_batch_size=8,
                              gradient_accumulation_steps=2, learning_rate=2e-4, lr_scheduler_type="cosine",
                              warmup_steps=0.1, max_steps=40 if args.quick else 150, bf16=True, logging_steps=10,
                              save_strategy="no", report_to="none", remove_unused_columns=False, seed=0)
    reset_peak()
    Trainer(model=model, args=targs, train_dataset=Dataset.from_list(encode_all(tok, splits["train"])),
            data_collator=pad_collate(tok.pad_token_id)).train()
    print(f"[train] peak={peak_gb():.1f} GB (vs ~10+ GB for full FT in lesson 02/03)")
    lora_m = eval_tickets(model, tok, test, "hand LoRA")

    section("3. Merge W <- W + (alpha/r) B A and check outputs match")
    with torch.no_grad():
        before_merge = model(**probe).logits
        merge_all(model)
        diff = (model(**probe).logits - before_merge).abs().max().item()
    print(f"max |logit diff| merged vs adapter = {diff:.3f}")
    print("  Not zero: W + delta is rounded to bf16 (8 mantissa bits), and small LoRA updates partly round away.")
    print("  If parity matters, merge in fp32 and cast once at the end, or serve the unmerged adapter.")
    merged_m = eval_tickets(model, tok, test, "hand LoRA merged")
    print(f"[merge] accuracy adapter={lora_m['exact']:.3f} merged={merged_m['exact']:.3f}")
    del model
    free_gpu()

    section("4. The PEFT equivalent")
    from peft import LoraConfig, get_peft_model
    model, _ = load_model(BASE_SMALL)
    cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules=LORA_TARGETS, task_type="CAUSAL_LM")
    pm = get_peft_model(model, cfg)
    peft_n, _ = count_params(pm)
    print(f"[params] hand={ours:,} peft={peft_n:,} match={ours == peft_n}")
    del pm, model
    free_gpu()

    section("5. How targets and rank change the budget (Qwen3-0.6B)")
    from transformers import AutoConfig
    c = AutoConfig.from_pretrained(BASE_SMALL)
    h, i, L = c.hidden_size, c.intermediate_size, c.num_hidden_layers
    q, kv = c.num_attention_heads * c.head_dim, c.num_key_value_heads * c.head_dim
    shapes = {"q_proj": (h, q), "k_proj": (h, kv), "v_proj": (h, kv), "o_proj": (q, h),
              "gate_proj": (h, i), "up_proj": (h, i), "down_proj": (i, h)}
    rows = []
    for tname, tg in [("q,v (orig paper)", ["q_proj", "v_proj"]), ("attention", ["q_proj", "k_proj", "v_proj", "o_proj"]),
                      ("all linear", LORA_TARGETS)]:
        for r in (8, 16, 64):
            p = L * sum(r * (a + b) for a, b in (shapes[t] for t in tg))
            rows.append({"targets": tname, "r": r, "trainable": f"{p / 1e6:.2f}M", "% of 0.6B": f"{100 * p / 6e8:.2f}%"})
    table(rows, ["targets", "r", "trainable", "% of 0.6B"])


if __name__ == "__main__":
    main()
