"""
08 - Synthetic data with the local 9B model: generate, filter, dedup, decontaminate
===================================================================================

RUNS ON WINDOWS (.venv), with the vLLM server UP — the only lesson that does:
    bash server/serve.sh                                         # WSL
    .venv\\Scripts\\python.exe finetune_course\\08_synthetic_data_vllm.py [--quick]

Interview topics
----------------
- Most fine-tuning data today is synthetic or LLM-assisted (Self-Instruct,
  Alpaca, Evol-Instruct, distillation from a bigger "teacher"). Check the
  teacher's license/ToS before training on its outputs.
- LABEL-FIRST generation: sample the label (category, priority, ...) in
  code, ask the LLM to write an input that matches it. Labels are correct by
  construction; the LLM only supplies linguistic diversity. Much more
  reliable than asking an LLM to label raw text.
- The pipeline that matters more than the generator:
    1. validate  (schema / rule checks in code: mentions the product, refund
                  wording consistent with the label, length bounds, no leaked JSON)
    2. dedup     (exact + near-duplicate: character n-gram Jaccard / MinHash)
    3. decontaminate (drop anything n-gram-overlapping the TEST set,
                  otherwise your eval measures memorisation)
    4. balance   (class distribution; LLMs over-produce "typical" cases)
- Preference pairs for DPO (lesson 09): chosen = known-correct label,
  rejected = a real model's wrong answer to the same prompt. Real mistakes
  make better negatives than random corruption.
- Diversity knobs: temperature, persona/style seeds, varying constraints.
  Model collapse risk when training repeatedly on own outputs.

Out: finetune_course/outputs/08_synth/{tickets,pairs}.jsonl  (lesson 09 uses pairs)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from collections import Counter
from pathlib import Path

from pydantic import BaseModel, Field

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from data import PRODUCTS, PRIORITY_CUES, TICKET_SYSTEM, parse_json, ticket_splits  # noqa: E402

# Both courses have a common.py; load the LangGraph one (vLLM client) by path.
_spec = importlib.util.spec_from_file_location("lg_common", HERE.parent / "langgraph_course" / "common.py")
lg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lg)

STYLES = ["terse and annoyed", "polite and detailed", "non-native English speaker", "rambling, mentions an unrelated detail",
          "very short, lowercase, no punctuation", "formal business tone"]
PRIORITY_RULES = {
    "urgent": "a whole team blocked / production down, a security problem, or money being lost right now",
    "high": "needs fixing today, a deadline tomorrow, or a repeated complaint",
    "medium": "a real problem but with a workaround, or a 'this week' request",
    "low": "no rush, a question, or just an FYI",
}


class Ticket(BaseModel):
    ticket: str = Field(description="the customer's message, 1-3 sentences")


class Triage(BaseModel):
    category: str = Field(description="billing|technical|shipping|account")
    priority: str = Field(description="low|medium|high|urgent")
    product: str
    refund_requested: bool


def spec_prompt(label: dict, style: str) -> str:
    refund = "explicitly asks for a refund or money back" if label["refund_requested"] else \
        "does NOT mention refunds or money back at all"
    return (f"Write one customer support message about the product '{label['product']}'.\n"
            f"- Topic category: {label['category']}\n"
            f"- Urgency: {PRIORITY_RULES[label['priority']]} (make this clear from the text, don't name a priority level)\n"
            f"- The customer {refund}.\n- Writing style: {style}.\n"
            "Mention the product name exactly as given. Output only the message.")


def validate(text: str, label: dict) -> str | None:
    """Return a rejection reason, or None if OK. Cheap code checks catch most garbage."""
    low = text.lower()
    if not 20 <= len(text) <= 400:
        return "length"
    if label["product"].lower() not in low:
        return "product missing"
    if any(ch in text for ch in "{}") or "priority" in low:
        return "leaked label/format"
    mentions_refund = any(w in low for w in ("refund", "money back", "reimburse"))
    if mentions_refund != label["refund_requested"]:
        return "refund mismatch"
    return None


def shingles(text: str, n: int = 5) -> set[str]:
    t = " ".join(text.lower().split())
    return {t[i:i + n] for i in range(max(1, len(t) - n + 1))}


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / max(1, len(a | b))


def word_ngrams(text: str, n: int = 8) -> set[tuple[str, ...]]:
    w = text.lower().split()
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("-n", type=int, default=None, help="tickets to generate")
    args = ap.parse_args()
    n = args.n or (24 if args.quick else 200)
    rng = random.Random(0)
    out = HERE / "outputs" / "08_synth"
    out.mkdir(parents=True, exist_ok=True)
    print(f"model={lg.MODEL} at {lg.BASE_URL}; generating {n} tickets")

    labels = [{"category": rng.choice(["billing", "technical", "shipping", "account"]),
               "priority": rng.choice(list(PRIORITY_CUES)), "product": rng.choice(PRODUCTS),
               "refund_requested": rng.random() < 0.3} for _ in range(n)]
    gen = lg.structured(Ticket, temperature=1.0)
    prompts = [spec_prompt(lab, rng.choice(STYLES)) for lab in labels]
    results = gen.batch(prompts, config={"max_concurrency": 4}, return_exceptions=True)

    reasons: Counter[str] = Counter()
    kept: list[dict] = []
    test_ngrams = set().union(*(word_ngrams(ex["messages"][1]["content"]) for ex in ticket_splits()["test"]))
    seen: list[set[str]] = []
    for lab, res in zip(labels, results):
        if isinstance(res, Exception):
            reasons["llm error"] += 1
            continue
        text = res.ticket.strip()
        if (why := validate(text, lab)) is not None:
            reasons[why] += 1
            continue
        sh = shingles(text)
        if any(jaccard(sh, s) > 0.6 for s in seen):
            reasons["near-duplicate"] += 1
            continue
        if word_ngrams(text) & test_ngrams:
            reasons["test contamination"] += 1
            continue
        seen.append(sh)
        kept.append({"messages": [{"role": "system", "content": TICKET_SYSTEM}, {"role": "user", "content": text}],
                     "target": json.dumps(lab), "label": lab})
    print(f"[synth] generated={n} kept={len(kept)} rejected={dict(reasons)}")
    print(f"[synth] class balance: {dict(Counter(k['label']['category'] for k in kept))}")
    for k in kept[:3]:
        print(f"  {k['label']['priority']:6s} {k['messages'][1]['content']}")

    # Preference pairs: ask the 9B to triage WITHOUT the house rules; its
    # disagreements with the known label become (chosen, rejected) pairs.
    # structured() already prepends a system message (the schema hint), and the
    # Qwen3.5 template rejects a second one, so the instructions go in the user turn.
    triage = lg.structured(Triage)
    answers = triage.batch([f"{TICKET_SYSTEM}\n\nTicket: {k['messages'][1]['content']}" for k in kept],
                           config={"max_concurrency": 4}, return_exceptions=True)
    pairs = []
    agree = errors = 0
    for k, a in zip(kept, answers):
        if isinstance(a, Exception):
            if not errors:
                print(f"  (triage error: {type(a).__name__}: {str(a)[:150]})")
            errors += 1
            continue
        pred = a.model_dump()
        if pred == k["label"]:
            agree += 1
        else:
            pairs.append({"messages": k["messages"], "chosen": k["target"], "rejected": json.dumps(pred)})
    print(f"[pairs] 9B agreed with label on {agree}/{len(kept) - errors} (errors={errors}); "
          f"{len(pairs)} preference pairs from its mistakes")
    assert parse_json(pairs[0]["rejected"]) if pairs else True

    with open(out / "tickets.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(k) + "\n" for k in kept)
    with open(out / "pairs.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(p) + "\n" for p in pairs)
    print(f"wrote {out / 'tickets.jsonl'} and {out / 'pairs.jsonl'}")


if __name__ == "__main__":
    main()
