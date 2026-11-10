"""
17 - Model merging: combine skills without more training
========================================================

Interview topics
----------------
- Task arithmetic (Ilharco et al. 2023): a fine-tune's "task vector" is
  tau = W_ft - W_base. Add task vectors to combine skills, subtract to
  remove behaviours. LoRA adapters ARE task vectors (B@A).
- Interference: two task vectors can push the same weights in opposite
  directions or overwrite each other. Fixes:
    * TIES (Yadav et al. 2023): Trim (keep top-k% magnitude per vector),
      Elect sign (majority sign per parameter), merge only agreeing values.
    * DARE (Yu et al. 2024): randomly Drop p% of delta entries And REscale
      the rest by 1/(1-p); deltas are very redundant, so 90% can go.
      DARE+TIES is common.
- Linear vs concatenation for LoRA: averaging A's and B's separately is NOT
  the average of B@A (cross terms). "cat" stacks the ranks, giving exactly
  sum_i w_i * B_i@A_i (rank grows). PEFT's `add_weighted_adapter` does both.
- Full-model merging (mergekit): linear "model soups" (average fine-tunes
  of the same base, often beats each), SLERP (spherical interpolation of two
  models, preserves weight norm), TIES/DARE on full deltas, "frankenmerges"
  (stacking layers), evolutionary merge search (Sakana AI, 2024). Many top
  open-leaderboard models were merges.
- Merging only works for models fine-tuned from the SAME base (same
  architecture and starting point: linear mode connectivity).
- Alternative to merging: keep adapters separate and route per request
  (multi-LoRA serving, lesson 13), or train one multi-task adapter.

Run:  python finetune_course/17_model_merging.py [--quick]
Needs: outputs/05_sft_lora (tickets), outputs/11_tool_lora (tool calls)
"""

from __future__ import annotations

from peft import PeftModel

from common import (BASE_SMALL, OUT, eval_tickets, eval_toolcalls, load_model, parse_args, require, section, table)
from data import ticket_splits, toolcall_splits


def main() -> None:
    args = parse_args(__doc__)
    n = 60 if args.quick else 200
    tickets = ticket_splits()["test"][:n]
    tools = toolcall_splits()["test"][: min(n, 120)]

    model, tok = load_model(BASE_SMALL)
    pm = PeftModel.from_pretrained(model, str(require(OUT / "05_sft_lora", "05")), adapter_name="tickets")
    pm.load_adapter(str(require(OUT / "11_tool_lora", "11")), adapter_name="tools")

    merges = {
        "linear (avg A, avg B)": dict(combination_type="linear", weights=[0.5, 0.5]),
        "cat (exact sum of BA)": dict(combination_type="cat", weights=[1.0, 1.0]),
        "TIES density=0.5": dict(combination_type="ties", weights=[1.0, 1.0], density=0.5),
        "DARE-TIES density=0.3": dict(combination_type="dare_ties", weights=[1.0, 1.0], density=0.3),
    }
    for i, (label, kw) in enumerate(merges.items()):
        pm.add_weighted_adapter(["tickets", "tools"], adapter_name=f"m{i}", **kw)

    rows = []
    for name, label in [("tickets", "tickets adapter only"), ("tools", "tools adapter only"),
                        *[(f"m{i}", lab) for i, lab in enumerate(merges)]]:
        section(label)
        pm.set_adapter(name)
        t = eval_tickets(pm, tok, tickets, f"{label} | tickets")["exact"]
        c = eval_toolcalls(pm, tok, tools, f"{label} | tools")["exact"]
        print(f"[merge {label}] tickets={t:.3f} tools={c:.3f}")
        rows.append({"adapter": label, "ticket acc": f"{t:.3f}", "tool exact": f"{c:.3f}", "mean": f"{(t + c) / 2:.3f}"})

    section("Base model (no adapter)")
    with pm.disable_adapter():
        t = eval_tickets(pm, tok, tickets, "base | tickets")["exact"]
        c = eval_toolcalls(pm, tok, tools, "base | tools")["exact"]
    rows.insert(0, {"adapter": "base", "ticket acc": f"{t:.3f}", "tool exact": f"{c:.3f}", "mean": f"{(t + c) / 2:.3f}"})

    section("Summary")
    table(rows, ["adapter", "ticket acc", "tool exact", "mean"])
    print("\nA single adapter is best at its own task and near-base on the other. A good merge keeps most")
    print("of BOTH without any training. Note how 'linear' (averaging A and B separately) differs from 'cat'.")


if __name__ == "__main__":
    main()
