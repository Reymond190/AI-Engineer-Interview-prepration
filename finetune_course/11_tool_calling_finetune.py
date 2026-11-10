"""
11 - Fine-tuning a small model for tool calling (agents)
========================================================

Interview topics
----------------
- Function calling is a FORMAT the model learned in post-training: tool
  JSON schemas go into the system prompt via the chat template, the model
  emits a structured call (Qwen: <tool_call>{"name":..., "arguments":...}</tool_call>),
  the server's tool parser (vLLM --tool-call-parser hermes / qwen3_coder)
  turns that text back into OpenAI `tool_calls`.
- Why fine-tune for it: small models pick the wrong tool, invent
  arguments, call tools when they should just answer, or break the JSON.
  A few hundred to a few thousand examples of YOUR tools fixes most of it
  (cheaper and faster at inference than a bigger model).
- Data must cover: each tool, argument extraction/normalisation (ISO
  currency codes), multiple phrasings, AND negatives (no tool needed) —
  otherwise the model learns "always call a tool".
- Train with the SAME template + tool schemas you serve with; the schema
  text is part of the prompt (masked out of the loss, but it costs ~250
  tokens per example here).
- Evaluate structurally: valid call rate, right tool, exact arguments,
  and false-positive calls on "no tool" prompts. Then do end-to-end agent
  evals (lesson 21 of langgraph_course): a perfect call format still needs
  guardrails in code (validation, retries) at run time.
- Multi-turn agent traces (call -> tool result -> answer) need the loss on
  every assistant turn, not just the last one.

Run:  python finetune_course/11_tool_calling_finetune.py [--quick]
Out:  finetune_course/outputs/11_tool_lora  (used by 13, 17)
"""

from __future__ import annotations

from datasets import Dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

from common import (BASE_SMALL, LORA_TARGETS, eval_toolcalls, load_model, out_dir, parse_args, peak_gb, reset_peak,
                    section, sft_records)
from data import toolcall_splits


def main() -> None:
    args = parse_args(__doc__)
    splits = toolcall_splits()
    test = splits["test"][:60] if args.quick else splits["test"]

    section("1. Base model with tool schemas in the prompt")
    model, tok = load_model(BASE_SMALL)
    before = eval_toolcalls(model, tok, test, "base", show=3)

    section("2. SFT on 1200 tool-use examples (25% of them: no tool needed)")
    recs = sft_records(tok, splits["train"])
    print(f"example completion: {recs[0]['completion']!r}")
    cfg = SFTConfig(
        output_dir=str(out_dir("_tmp_11")),
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=0.1,
        max_steps=40 if args.quick else 150,
        bf16=True,
        max_length=768,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        seed=0,
    )
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=LORA_TARGETS, task_type="CAUSAL_LM")
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=Dataset.from_list(recs), processing_class=tok,
                         peft_config=lora)
    reset_peak()
    trainer.train()
    print(f"[train] peak={peak_gb():.1f} GB")

    section("3. After")
    after = eval_toolcalls(trainer.model, tok, test, "tool SFT", show=3)
    print(f"[result] tool_exact {before['exact']:.3f} -> {after['exact']:.3f}")
    save = out_dir("11_tool_lora")
    trainer.model.save_pretrained(save)
    tok.save_pretrained(save)
    print(f"saved adapter to {save}")


if __name__ == "__main__":
    main()
