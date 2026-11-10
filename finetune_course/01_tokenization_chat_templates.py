"""
01 - Tokenization, chat templates, label masking, padding vs packing
====================================================================

Interview topics
----------------
- A chat model never sees "messages", only one token sequence produced by its
  CHAT TEMPLATE (Jinja in tokenizer_config.json). Qwen uses ChatML:
      <|im_start|>system\\n...<|im_end|>\\n<|im_start|>user\\n...<|im_end|>\\n<|im_start|>assistant\\n
  Fine-tune with a different format than you serve with => silent quality loss.
  The #1 practical SFT bug.
- Hybrid reasoning models: Qwen3 non-thinking mode inserts an EMPTY
  "<think>\\n\\n</think>\\n\\n" block. Train data must contain it too.
- Label masking: labels = input_ids with prompt positions set to -100
  (PyTorch CE ignore_index). Loss only on the assistant answer =
  "completion-only" / "train on responses". Training on prompts wastes
  capacity and teaches the model to generate user text.
- The answer must end with the end-of-turn token (<|im_end|>) and it must be
  in the labels, or the model never learns to stop (endless generation).
- Tokenise prompt and completion separately: BPE can merge characters across
  the join, shifting the boundary by a token.
- Padding side: RIGHT for training (labels align), LEFT for batched
  generation (new tokens must follow each row's real last token).
- Padding wastes compute on short/long mixes. PACKING concatenates examples
  into full-length rows; done right (padding-free, position_ids reset per
  example + varlen attention) examples cannot attend to each other.
  Naive packing lets example 2 attend to example 1: cross-contamination.
- Vocab size matters for memory (embeddings + logits) and for speed on
  multilingual text (tokens per word, "fertility").

Run:  python finetune_course/01_tokenization_chat_templates.py
"""

from __future__ import annotations

import statistics

from common import BASE_SMALL, encode, example_texts, load_tokenizer, parse_args, render_prompt, section
from data import ticket_splits, toolcall_splits


def main() -> None:
    parse_args(__doc__)
    tok = load_tokenizer(BASE_SMALL)
    ex = ticket_splits()["train"][0]

    section("1. Tokens")
    text = ex["messages"][1]["content"]
    ids = tok(text, add_special_tokens=False)["input_ids"]
    print(f"vocab_size={len(tok):,}  eos={tok.eos_token!r}  pad={tok.pad_token!r}")
    print(f"text   : {text}")
    print(f"tokens : {[tok.decode([i]) for i in ids]}")
    print(f"{len(text.split())} words -> {len(ids)} tokens")

    section("2. The chat template (what the model actually reads)")
    print("--- thinking OFF (what we train and serve with) ---")
    print(render_prompt(tok, ex["messages"]))
    print("--- thinking ON ---")
    print(tok.apply_chat_template(ex["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=True))

    section("3. Prompt / completion split and label masking")
    prompt, completion = example_texts(tok, ex)
    print(f"completion text: {completion!r}")
    enc = encode(tok, prompt, completion)
    n_masked = sum(1 for lab in enc["labels"] if lab == -100)
    print(f"[labels] total={len(enc['labels'])} masked_prompt_tokens={n_masked} "
          f"trained_tokens={len(enc['labels']) - n_masked}")
    trained = [tok.decode([i]) for i, lab in zip(enc["input_ids"], enc["labels"]) if lab != -100]
    print(f"tokens the loss sees: {trained}")
    print(f"[eos in labels] {tok.convert_tokens_to_ids('<|im_end|>') in enc['labels']}")

    section("4. Why tokenise the two parts separately")
    joined = tok(prompt + completion, add_special_tokens=False)["input_ids"]
    separate = enc["input_ids"]
    print(f"joined == separate: {joined == separate}  (here the template ends in '\\n\\n', a clean boundary;")
    print("  a raw-text prompt ending in a space is the classic trap)")
    a, b = "Label: ", "positive"
    j, s = tok(a + b)["input_ids"], tok(a)["input_ids"] + tok(b)["input_ids"]
    print(f"  {a!r}+{b!r}: joined={[tok.decode([i]) for i in j]} separate={[tok.decode([i]) for i in s]}")
    print("  At inference the prompt ends in ' ' and the model must emit 'positive' (no space),")
    print("  a token sequence it never saw if training text was tokenised joined (' positive').")

    section("5. Tool calls are just more template")
    tex = next(e for e in toolcall_splits()["train"] if e["call"])
    p, c = example_texts(tok, tex)
    print(f"prompt has {len(tok(p)['input_ids'])} tokens (tool schemas live in the system prompt)")
    print(f"completion: {c!r}")

    section("6. Padding waste vs packing")
    lens = [len(encode(tok, *example_texts(tok, e))["input_ids"]) for e in ticket_splits()["train"][:512]]
    bs = 16
    padded = sum(max(lens[i:i + bs]) * len(lens[i:i + bs]) for i in range(0, len(lens), bs))
    print(f"lengths: min={min(lens)} median={statistics.median(lens):.0f} max={max(lens)}")
    print(f"[padding] real tokens={sum(lens):,}, padded tokens={padded:,}, "
          f"waste={100 * (1 - sum(lens) / padded):.1f}% (batch={bs})")
    print("Packing (lesson 14) fills rows to max_len with whole examples -> ~0% waste.")
    print("Our prompts are similar-length, so waste is small; on real chat data it is often 50%+.")


if __name__ == "__main__":
    main()
