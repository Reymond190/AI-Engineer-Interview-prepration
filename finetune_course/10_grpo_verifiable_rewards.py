"""
10 - RL with verifiable rewards: GRPO
=====================================

Interview topics
----------------
- RLVR (RL from verifiable rewards): when correctness can be CHECKED by code
  (math answer, unit tests pass, JSON validates, tool call executes), use
  that as the reward. No reward model, no human labels. This is how
  DeepSeek-R1 / o1-style reasoning models are trained.
- GRPO (DeepSeek-Math 2024): for each prompt sample a GROUP of G
  completions, reward each, advantage = (r - mean(group)) / std(group).
  The group mean replaces PPO's learned value function (critic), so
  only policy (+ optional reference) in memory.
- Objective: PPO-style clipped ratio pi/pi_old * A, per token, minus
  beta * KL(pi || ref). Modern practice often sets beta=0 (no reference
  model at all, TRL's default) — lesson 18 compares the variants.
- Groups where every completion gets the same reward have zero advantage:
  no learning signal. Tasks must be "hard but possible" for the current
  model (~10-90% pass rate). Curriculum matters.
- Reward design is the whole game. Reward hacking: the policy finds any
  loophole (prints many numbers, repeats the format line, games length).
  Make rewards strict, check samples by hand, and combine correctness with
  format rewards carefully (format alone must not dominate).
- RL vs SFT: SFT imitates fixed answers (off-policy); RL learns from its OWN
  samples (on-policy) and can exceed the demonstrations. RL mostly
  sharpens what the base model can already do sometimes (pass@k -> pass@1).
- Cost: generation dominates (it's sampling, not a forward pass). That's
  why real setups use vLLM for rollouts (lesson 18).

Run:  python finetune_course/10_grpo_verifiable_rewards.py [--quick]
"""

from __future__ import annotations

import re

from datasets import Dataset
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

from common import (BASE_SMALL, LORA_TARGETS, eval_math, load_model, out_dir, parse_args, peak_gb, render_prompt,
                    reset_peak, section)
from data import extract_answer, math_splits


def correctness_reward(completions: list[str], answer: list[int], **_) -> list[float]:
    return [1.0 if extract_answer(c) == a else 0.0 for c, a in zip(completions, answer)]


def format_reward(completions: list[str], **_) -> list[float]:
    """Small bonus for ending with exactly one 'Answer: N' line. Kept small so it can't dominate."""
    return [0.2 if extract_answer(c) is not None and c.count("Answer:") == 1 else 0.0 for c in completions]


def naive_reward(completions: list[str], answer: list[int], **_) -> list[float]:
    """A HACKABLE reward: 'the right number appears somewhere'."""
    return [1.0 if str(a) in re.findall(r"-?\d+", c) else 0.0 for c, a in zip(completions, answer)]


def reward_hacking_demo() -> None:
    hack = " ".join(str(i) for i in range(200))
    cases = {"honest correct": "5 + 3 = 8.\nAnswer: 8", "honest wrong": "5 + 3 = 9.\nAnswer: 9",
             "hack: list every number": hack, "hack: two answers": "Answer: 7\nAnswer: 8"}
    for name, c in cases.items():
        print(f"  {name:26s} naive={naive_reward([c], [8])[0]:.1f}  strict={correctness_reward([c], [8])[0]:.1f}"
              f"  format={format_reward([c])[0]:.1f}")
    print("  => the naive reward pays 1.0 for spraying numbers; RL WILL find that.")


def reward_curve(trainer) -> list[tuple[int, float, float]]:
    return [(h["step"], h["reward"], h.get("completions/mean_length", 0.0))
            for h in trainer.state.log_history if "reward" in h]


def make_trainer(model, tok, train, cfg_kwargs: dict, name: str, max_steps: int) -> GRPOTrainer:
    cfg = GRPOConfig(
        output_dir=str(out_dir(f"_tmp_{name}")),
        per_device_train_batch_size=8,
        num_generations=4,                 # group size G: 2 prompts x 4 samples per step
        max_completion_length=256,
        temperature=1.0,
        learning_rate=2e-5,
        lr_scheduler_type="constant",
        max_steps=max_steps,
        bf16=True,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        seed=0,
        **cfg_kwargs,
    )
    lora = LoraConfig(r=16, lora_alpha=32, target_modules=LORA_TARGETS, task_type="CAUSAL_LM")
    return GRPOTrainer(model=model, reward_funcs=[correctness_reward, format_reward], args=cfg,
                       train_dataset=train, processing_class=tok, peft_config=lora)


def main() -> None:
    args = parse_args(__doc__)
    splits = math_splits()
    test = splits["test"][: 50 if args.quick else 100]

    section("1. Reward design: strict vs hackable")
    reward_hacking_demo()

    section("2. Baseline")
    model, tok = load_model(BASE_SMALL)
    before = eval_math(model, tok, test, "base")
    train = Dataset.from_list([{"prompt": render_prompt(tok, ex["messages"]), "answer": ex["answer"]}
                               for ex in splits["train"]])

    section("3. GRPO (classic: token-level loss averaged per sequence, KL beta=0.04)")
    steps = 25 if args.quick else 120
    trainer = make_trainer(model, tok, train, {"loss_type": "grpo", "beta": 0.04, "scale_rewards": "group"}, "10", steps)
    reset_peak()
    trainer.train()
    print(f"[train] peak={peak_gb():.1f} GB")
    pts = reward_curve(trainer)
    for s, r, length in pts[:: max(1, len(pts) // 8)]:
        print(f"  step {s:3d}  mean_reward={r:.3f}  mean_len={length:.0f}")
    k = max(1, len(pts) // 4)
    first, last = sum(p[1] for p in pts[:k]) / k, sum(p[1] for p in pts[-k:]) / k
    print(f"[grpo] reward first={first:.3f} last={last:.3f}")

    section("4. After GRPO")
    after = eval_math(trainer.model, tok, test, "GRPO")
    print(f"[result] math_acc {before['acc']:.3f} -> {after['acc']:.3f}, format {before['format']:.2f} -> {after['format']:.2f}")


if __name__ == "__main__":
    main()
