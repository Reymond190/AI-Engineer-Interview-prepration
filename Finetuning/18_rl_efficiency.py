"""
18 - Modern RL fine-tuning practice: GRPO variants and rollout speed
====================================================================

Same task, rewards, steps and LoRA as lesson 10; only the RL objective /
rollout engine changes. Reports reward curve, completion length, wall time
and held-out accuracy per variant.

Interview topics
----------------
- Original GRPO (DeepSeek-Math): per-token PPO-clip loss averaged per
  SEQUENCE, then across the group; advantages divided by group std; KL to a
  reference. Known biases (Dr. GRPO, Liu et al. 2025):
    * per-sequence length normalisation => long WRONG answers are penalised
      less per token => responses grow over training ("length bias");
    * std normalisation over-weights groups that are almost all right/wrong
      (question-difficulty bias).
  Dr. GRPO: normalise by a constant (not each length), no std scaling.
- DAPO (ByteDance, 2025): token-level loss over the whole batch (long
  sequences count per token), "clip-higher" (epsilon_high 0.28 > epsilon 0.2:
  lets low-probability tokens grow => keeps entropy/exploration alive),
  dynamic sampling (drop groups with all-equal rewards: zero advantage),
  overlong filtering/shaping (`mask_truncated_completions`). TRL's default
  loss is now "dapo" and default beta is 0 (no KL, no reference model).
- GSPO (Qwen team, 2025): importance ratio and clipping at the SEQUENCE
  level (geometric mean of token ratios), matching the sequence-level
  reward. More stable, especially for MoE models. Also CISPO (MiniMax):
  clip the IS weight, not the update, so no token's gradient is dropped.
- Ratios only matter when data is reused: with one optimizer step per batch
  of rollouts the policy IS the rollout policy (ratio = 1). Here
  steps_per_generation=2 so the second step is slightly off-policy and
  clipping actually does something.
- Rollouts dominate RL wall-clock. Fixes: vLLM generation (TRL
  `use_vllm=True`; "colocate" shares the GPU with training and sleeps
  between phases, "server" uses separate GPUs), paged/continuous batching
  in transformers (`use_transformers_paged=True`), shorter max lengths,
  async/off-policy pipelines with importance-sampling correction.

Run:  python finetune_course/18_rl_efficiency.py [--quick]
"""

from __future__ import annotations

import importlib
import importlib.util
import time

from datasets import Dataset
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

from common import (BASE_SMALL, LORA_TARGETS, eval_math, free_gpu, load_model, out_dir, parse_args, peak_gb,
                    render_prompt, reset_peak, section, table)
from data import math_splits

L10 = importlib.import_module("10_grpo_verifiable_rewards")

VARIANTS: dict[str, dict] = {
    "GRPO (original)": dict(loss_type="grpo", beta=0.04, scale_rewards="group"),
    "Dr. GRPO": dict(loss_type="dr_grpo", beta=0.0, scale_rewards="none"),
    "DAPO": dict(loss_type="dapo", beta=0.0, epsilon_high=0.28, mask_truncated_completions=True),
    "GSPO": dict(loss_type="grpo", beta=0.0, importance_sampling_level="sequence", epsilon=3e-4, epsilon_high=4e-4),
}


def run(name: str, overrides: dict, train, test, steps: int) -> dict:
    model, tok = load_model(BASE_SMALL)
    cfg = GRPOConfig(output_dir=str(out_dir("_tmp_18")), per_device_train_batch_size=8, num_generations=4,
                     steps_per_generation=2, max_completion_length=256, temperature=1.0, learning_rate=2e-5,
                     lr_scheduler_type="constant", max_steps=steps, bf16=True, logging_steps=1, save_strategy="no",
                     report_to="none", seed=0, **overrides)
    lora = LoraConfig(r=16, lora_alpha=32, target_modules=LORA_TARGETS, task_type="CAUSAL_LM")
    row: dict = {"variant": name}
    try:
        trainer = GRPOTrainer(model=model, reward_funcs=[L10.correctness_reward, L10.format_reward], args=cfg,
                              train_dataset=train, processing_class=tok, peft_config=lora)
        reset_peak()
        t0 = time.time()
        trainer.train()
        pts = L10.reward_curve(trainer)
        k = max(1, len(pts) // 4)
        first, last = sum(p[1] for p in pts[:k]) / k, sum(p[1] for p in pts[-k:]) / k
        length = sum(p[2] for p in pts[-k:]) / k
        acc = eval_math(trainer.model, tok, test, name)["acc"]
        row.update({"reward first": f"{first:.3f}", "reward last": f"{last:.3f}", "len last": f"{length:.0f}",
                    "train s": f"{time.time() - t0:.0f}", "peak GB": f"{peak_gb():.1f}", "math acc": f"{acc:.3f}"})
        del trainer
    except Exception as e:  # noqa: BLE001  config unsupported by this TRL version = reported, not fatal
        row["reward first"] = f"FAILED: {type(e).__name__}: {str(e)[:80]}"
    del model
    free_gpu()
    print(f"[rl-variant {name}] {row}")
    return row


def main() -> None:
    args = parse_args(__doc__)
    splits = math_splits()
    test = splits["test"][: 50 if args.quick else 100]
    steps = 16 if args.quick else 80

    model, tok = load_model(BASE_SMALL)
    base = eval_math(model, tok, test, "base")["acc"]
    train = Dataset.from_list([{"prompt": render_prompt(tok, ex["messages"]), "answer": ex["answer"]}
                               for ex in splits["train"]])
    del model
    free_gpu()

    section(f"1. Objective variants ({steps} steps each, LoRA r=16, G=4)")
    rows = [run(name, kw, train, test, steps) for name, kw in VARIANTS.items()]

    section("2. Rollout engine")
    rows.append(run("DAPO + paged generation", {**VARIANTS["DAPO"], "use_transformers_paged": True}, train, test, steps))
    if importlib.util.find_spec("vllm"):
        rows.append(run("DAPO + vLLM colocate", {**VARIANTS["DAPO"], "use_vllm": True, "vllm_mode": "colocate",
                                                 "vllm_gpu_memory_utilization": 0.3}, train, test, steps))
    else:
        print("(vllm is not installed in the `ft` env on purpose: it pins torch. In a dedicated RL env,")
        print(" use_vllm=True + vllm_mode='colocate' typically gives several-x faster rollouts.)")

    section("Summary")
    print(f"base math acc = {base:.3f}")
    table(rows, ["variant", "reward first", "reward last", "len last", "train s", "peak GB", "math acc"])
    print("\nWith this few steps differences are noisy: read the length column (length bias) and stability")
    print("more than final accuracy. Re-run with --seed to see the variance; RL needs repeated runs to compare.")


if __name__ == "__main__":
    main()
