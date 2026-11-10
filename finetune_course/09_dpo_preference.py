"""
09 - Preference tuning with DPO (on top of the SFT model)
=========================================================

Interview topics
----------------
- Pipeline: pretrain -> SFT (imitate good answers) -> preference tuning
  (prefer better over worse). SFT first: DPO only re-ranks what the policy
  can already produce. Doing DPO on a base model rarely works.
- RLHF (PPO): train a reward model on human comparisons, then PPO maximises
  reward - beta*KL(policy || reference). 4 models in memory (policy, ref,
  reward, value), unstable, expensive.
- DPO (Rafailov et al. 2023): the optimal KL-constrained policy has a closed
  form, so the reward model can be eliminated. Loss on (prompt, chosen y_w,
  rejected y_l):
      -log sigmoid( beta * [ (log pi(y_w) - log ref(y_w)) - (log pi(y_l) - log ref(y_l)) ] )
  "Implicit reward" r(y) = beta * log(pi(y)/ref(y)). Just 2 models
  (policy + frozen reference), supervised-style training.
- beta: KL strength. Small (0.05-0.1) = move further from the reference.
  LR is ~10x lower than SFT (5e-7..5e-6 full, ~5e-6..5e-5 LoRA).
- Metrics: rewards/chosen, rewards/rejected, rewards/margins,
  rewards/accuracies (fraction of pairs with chosen > rejected). Watch out:
  chosen log-prob can DROP while the margin grows (both go down, rejected faster).
- With PEFT the reference is free: the same weights with the adapter
  disabled. Here we MERGE the SFT adapter first so that "adapter off" means
  "SFT model", i.e. the reference is the SFT policy, as it should be.
- On-policy negatives: rejected = what THIS model actually gets wrong.
  Off-policy pairs (someone else's mistakes) teach less.
- Variants: IPO (fixes DPO overfitting to deterministic prefs), KTO (no
  pairs, just good/bad labels), ORPO (SFT + odds-ratio preference in one
  stage, no reference), SimPO (length-normalised, no reference), online/
  iterative DPO (regenerate pairs with the current policy each round).

Run:  python finetune_course/09_dpo_preference.py [--quick]
Needs: outputs/05_sft_lora; uses outputs/08_synth/pairs.jsonl if present
"""

from __future__ import annotations

import json
import random

from datasets import Dataset
from peft import LoraConfig, PeftModel
from trl import DPOConfig, DPOTrainer

from common import (BASE_SMALL, LORA_TARGETS, OUT, eval_tickets, generate, load_model, out_dir, parse_args,
                    peak_gb, render, render_prompt, require, reset_peak, section)
from data import corrupt_ticket, grade_ticket, ticket_splits


def pair(tok, messages: list[dict], chosen: str, rejected: str) -> dict[str, str]:
    """DPO record as strings rendered with the model's own template."""
    prompt, c = render(tok, messages, chosen)
    _, r = render(tok, messages, rejected)
    return {"prompt": prompt, "chosen": c, "rejected": r}


def main() -> None:
    args = parse_args(__doc__)
    splits = ticket_splits()
    test = splits["test"][:60] if args.quick else splits["test"]
    rng = random.Random(0)

    section("1. Start from the SFT model (merged, so it can be the DPO reference)")
    model, tok = load_model(BASE_SMALL)
    model = PeftModel.from_pretrained(model, str(require(OUT / "05_sft_lora", "05"))).merge_and_unload()
    sft = eval_tickets(model, tok, test, "SFT")

    section("2. Build preference pairs: on-policy mistakes first")
    pool = splits["train"][: 600 if args.quick else 2000]
    outs = generate(model, tok, [render_prompt(tok, ex["messages"]) for ex in pool])
    records, n_onpolicy = [], 0
    for ex, o in zip(pool, outs):
        if not grade_ticket(o, ex["label"])["exact"]:
            records.append(pair(tok, ex["messages"], ex["target"], o))
            n_onpolicy += 1
    synth = OUT / "08_synth" / "pairs.jsonl"
    n_08 = 0
    if synth.exists():
        for line in synth.read_text(encoding="utf-8").splitlines():
            p = json.loads(line)
            records.append(pair(tok, p["messages"], p["chosen"], p["rejected"]))
            n_08 += 1
    target = 200 if args.quick else 600
    fill = 0
    while len(records) < target:
        ex = rng.choice(splits["train"])
        records.append(pair(tok, ex["messages"], ex["target"], corrupt_ticket(ex["label"], rng)))
        fill += 1
    rng.shuffle(records)
    print(f"[pairs] on-policy={n_onpolicy} from-lesson-08={n_08} synthetic-corruptions={fill} total={len(records)}")
    if n_onpolicy:
        r0 = next(r for r in records if r["rejected"])
        print(f"  chosen  : {r0['chosen']!r}\n  rejected: {r0['rejected']!r}")

    section("3. DPO with a fresh LoRA on top (reference = SFT model with adapter disabled)")
    cfg = DPOConfig(
        output_dir=str(out_dir("_tmp_09")),
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        warmup_steps=0.1,
        max_steps=30 if args.quick else 100,
        beta=0.1,
        max_length=512,
        bf16=True,
        logging_steps=5,
        save_strategy="no",
        report_to="none",
        seed=0,
    )
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules=LORA_TARGETS, task_type="CAUSAL_LM")
    trainer = DPOTrainer(model=model, args=cfg, train_dataset=Dataset.from_list(records),
                         processing_class=tok, peft_config=lora)
    reset_peak()
    trainer.train()
    print(f"[train] peak={peak_gb():.1f} GB")
    logs = [h for h in trainer.state.log_history if "rewards/accuracies" in h]
    for h in logs[:: max(1, len(logs) // 6)] + logs[-1:]:
        print(f"  step {h['step']:3d}  loss={h.get('loss', float('nan')):.3f}  reward_acc={h['rewards/accuracies']:.2f}  "
              f"margin={h['rewards/margins']:.2f}  chosen={h['rewards/chosen']:.2f}  rejected={h['rewards/rejected']:.2f}")
    last = logs[-1]
    print(f"[dpo] reward_acc={last['rewards/accuracies']:.2f} margin={last['rewards/margins']:.2f}")

    section("4. Did preferences move the task metric?")
    dpo = eval_tickets(trainer.model, tok, test, "SFT+DPO", show=1)
    print(f"[result] accuracy SFT={sft['exact']:.3f} -> SFT+DPO={dpo['exact']:.3f}")
    print("If SFT is already near-perfect, DPO has little to fix: the pairs come from its few mistakes.")


if __name__ == "__main__":
    main()
