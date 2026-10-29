"""
07 - Evaluating a fine-tune: task gain, generalisation, forgetting
==================================================================

Interview topics
----------------
- Evaluate on THREE axes, every time:
    1. Target task, held-out test split (never seen, decontaminated).
    2. Generalisation: inputs a bit outside the training distribution
       (here: product names that never appear in training). Separates
       learning the rule from memorising strings.
    3. Regression / catastrophic forgetting: general skills the base model
       had (here: arithmetic word problems + perplexity on generic text).
- "LoRA learns less and forgets less" (Biderman et al. 2024): full FT
  reaches higher target scores on hard domains but drifts further from the
  base; LoRA's low-rank update stays closer. Measure it, don't assume it.
- Mitigating forgetting: LoRA / lower lr / fewer epochs, mix 5-20% general
  instruction data into SFT ("replay"), regularise toward the base (KL),
  merge back with the base (lesson 17 / model soups).
- Perplexity is a cheap regression alarm, NOT a quality metric. A model can
  get worse ppl and better answers (and vice versa).
- Deterministic graders > LLM-as-judge when the output is checkable; use
  judges (with a rubric, position swapping, a stronger model) for open
  text, and spot-check judges against humans.
- Report variance: n=200 exact-match has a ~±3-7% 95% CI. Small deltas
  need more test data or repeated seeds.

Run:  python finetune_course/07_evaluation_forgetting.py [--quick]
Needs: outputs/05_sft_lora (lesson 05), outputs/03_full_ft (lesson 03, optional)
"""

from __future__ import annotations

import copy
import json
import math

import torch

from common import (BASE_SMALL, OUT, eval_math, eval_tickets, free_gpu, load_model, parse_args, require, section,
                    table)
from data import PRODUCTS, math_splits, ticket_splits

GENERIC_TEXT = [
    "The water cycle describes how water evaporates from oceans, condenses into clouds, and returns to the surface as precipitation.",
    "In 1969, Apollo 11 landed on the Moon, and Neil Armstrong became the first person to walk on its surface.",
    "Photosynthesis converts light energy into chemical energy, producing glucose and releasing oxygen as a by-product.",
    "A balanced diet includes vegetables, fruits, whole grains, proteins, and healthy fats in appropriate portions.",
    "The French Revolution began in 1789 and led to profound changes in the political structure of Europe.",
    "Binary search finds an item in a sorted list by repeatedly halving the portion of the list that could contain it.",
    "Mount Everest, on the border between Nepal and China, is the highest mountain above sea level.",
    "Interest rates set by central banks influence borrowing costs, consumer spending, and inflation.",
]

NEW_PRODUCTS = ["QuantumMouse", "OrbitCam", "SolarKettle", "ByteBand"]


def unseen_products(examples: list[dict]) -> list[dict]:
    """Same tickets, product names never seen in training."""
    out = []
    for i, ex in enumerate(examples):
        old = ex["label"]["product"]
        new = NEW_PRODUCTS[i % len(NEW_PRODUCTS)]
        e = copy.deepcopy(ex)
        e["messages"][1]["content"] = e["messages"][1]["content"].replace(old, new)
        e["label"]["product"] = new
        e["target"] = json.dumps(e["label"])
        out.append(e)
    assert not set(NEW_PRODUCTS) & set(PRODUCTS)
    return out


@torch.no_grad()
def generic_ppl(model, tok) -> float:
    model.eval()
    total, n = 0.0, 0
    for t in GENERIC_TEXT:
        enc = tok(t, return_tensors="pt").to(model.device)
        out = model(**enc, labels=enc["input_ids"])
        k = enc["input_ids"].shape[1] - 1
        total, n = total + out.loss.item() * k, n + k
    return math.exp(total / n)


def evaluate(name: str, model, tok, test, ood, mth) -> dict:
    t = eval_tickets(model, tok, test, f"{name} test")
    o = eval_tickets(model, tok, ood, f"{name} unseen-products")
    m = eval_math(model, tok, mth, f"{name} math")
    ppl = generic_ppl(model, tok)
    print(f"[forgetting {name}] generic_ppl={ppl:.2f} math_acc={m['acc']:.3f}")
    return {"model": name, "task acc": f"{t['exact']:.3f}", "unseen-product acc": f"{o['exact']:.3f}",
            "math acc": f"{m['acc']:.3f}", "generic ppl": f"{ppl:.2f}"}


def main() -> None:
    args = parse_args(__doc__)
    n = 60 if args.quick else 200
    test = ticket_splits()["test"][:n]
    ood = unseen_products(test)
    mth = math_splits()["test"][: 40 if args.quick else 100]
    lora_path = require(OUT / "05_sft_lora", "05")

    rows = []
    section("Base model")
    model, tok = load_model(BASE_SMALL)
    rows.append(evaluate("base", model, tok, test, ood, mth))

    section("LoRA adapter (lesson 05)")
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, str(lora_path))
    rows.append(evaluate("lora", model, tok, test, ood, mth))

    section("LoRA merged into the base")
    model = model.merge_and_unload()
    rows.append(evaluate("lora-merged", model, tok, test, ood, mth))
    del model
    free_gpu()

    full = OUT / "03_full_ft"
    if full.exists():
        section("Full fine-tune (lesson 03)")
        model, tok = load_model(str(full))
        rows.append(evaluate("full-ft", model, tok, test, ood, mth))
        del model
        free_gpu()
    else:
        print("(skipping full FT: run lesson 03 to include it)")

    section("Summary")
    table(rows, list(rows[0]))
    print("\nRead it like an interviewer: did the task improve, does it transfer to unseen names,")
    print("and what did it cost on skills we did NOT train (math, generic perplexity)?")


if __name__ == "__main__":
    main()
