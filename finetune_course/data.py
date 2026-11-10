"""
data.py
-------
Deterministic synthetic datasets used across the course.

Why synthetic instead of a public dataset?
  * Grading is exact and done in code (no LLM judge), so "did fine-tuning help?"
    is one number you can trust, and run_all.py can regex-check it.
  * Labels follow HIDDEN CONVENTIONS a base model cannot guess from the prompt
    (e.g. "site is down" => priority "urgent", "charged twice" => billing +
    refund). That is exactly the situation where fine-tuning beats prompting:
    teaching a format + house rules, not new world knowledge.
  * Seeded => every run sees the same split, so lessons are comparable.

Three tasks:
  tickets    support ticket -> JSON {category, priority, product, refund_requested}
             (the running task, lessons 01-09, 12-17)
  arithmetic short multi-step word problems with a verifiable integer answer
             (RL with verifiable rewards, lessons 10 and 18)
  toolcalls  user request -> the right tool call (or a direct answer)
             (agent-style fine-tuning, lessons 11 and 17)

Each example is {"messages": [system, user], "target": <assistant text>, ...}.
`common.py` turns that into tokens with the model's chat template.
"""

from __future__ import annotations

import json
import random
from typing import Any

# ---------------------------------------------------------------------------
# Task 1: support tickets -> JSON
# ---------------------------------------------------------------------------

TICKET_SYSTEM = (
    "You triage customer support tickets. Reply with ONLY a JSON object with keys "
    '"category" (billing|technical|shipping|account), "priority" (low|medium|high|urgent), '
    '"product" (string), "refund_requested" (true|false).'
)

PRODUCTS = ["CloudSync", "PixelPad", "AeroDesk", "VoltCharge", "NimbusDB", "Lumen Router", "TrailWatch", "EchoBuds"]

# (category, text template, refund?) — {p} is the product.
# Every 4th template is HELD OUT: val/test use phrasings never seen in
# training, so the score measures learning the rule, not memorising strings.
ISSUES: list[tuple[str, str, bool]] = [
    ("billing", "I was charged twice for {p} this month.", True),
    ("billing", "My invoice for {p} shows the wrong plan.", False),
    ("billing", "Please cancel my {p} subscription and give me my money back.", True),
    ("billing", "The {p} renewal price went up without notice.", False),
    ("billing", "I want a refund for {p}, it is not what was advertised.", True),
    ("billing", "Why does my card statement show two {p} payments? Refund the extra one.", True),
    ("billing", "The {p} discount code wasn't applied at checkout.", False),
    ("billing", "I was billed for {p} after I cancelled, I expect a refund.", True),
    ("technical", "{p} crashes every time I open the settings page.", False),
    ("technical", "{p} won't connect to wifi after the last update.", False),
    ("technical", "Sync in {p} has been stuck at 99% for hours.", False),
    ("technical", "The {p} firmware update bricked my device, I want a refund.", True),
    ("technical", "{p} keeps logging errors about an expired certificate.", False),
    ("technical", "The {p} app drains my battery in an hour.", False),
    ("technical", "{p} is so slow it's unusable, I'd like my money back.", True),
    ("technical", "After a reboot {p} forgets all its settings.", False),
    ("shipping", "My {p} order hasn't arrived and tracking hasn't moved in a week.", False),
    ("shipping", "The {p} box arrived crushed and the unit is damaged, refund please.", True),
    ("shipping", "I received the wrong color of {p}.", False),
    ("shipping", "Can I change the delivery address for my {p} order?", False),
    ("shipping", "The courier says my {p} was delivered but it never came.", False),
    ("shipping", "Half of my {p} order is missing from the parcel, please refund that part.", True),
    ("shipping", "Customs is holding my {p} package.", False),
    ("shipping", "My {p} came without the charger in the box.", False),
    ("account", "I can't log in to my {p} account, the reset email never comes.", False),
    ("account", "Please delete my {p} account and all my data.", False),
    ("account", "Someone changed the email on my {p} account without me.", False),
    ("account", "How do I add a second user to my {p} account?", False),
    ("account", "My {p} account got locked after two-factor stopped working.", False),
    ("account", "I want to merge my two {p} accounts.", False),
    ("account", "My {p} profile shows someone else's name.", False),
    ("account", "I need to transfer my {p} account to a colleague.", False),
]

# Priority is a house rule keyed on cues. If a ticket has two cues, the
# HIGHER priority wins (a compositional rule the model has to learn).
# Every 3rd cue is held out for val/test.
PRIORITY_CUES: dict[str, list[str]] = {
    "urgent": ["Our whole team is blocked, production is down.", "This is a security issue, please act now.",
               "We lose money every minute this continues.", "Customers can't check out at all right now.",
               "I think our data is leaking.", "All of our stores are offline."],
    "high": ["I need this fixed today.", "This is the third time I'm writing about this.",
             "Our launch is tomorrow.", "My manager is asking about this by end of day.",
             "I have a demo to a client this afternoon.", "Please escalate this."],
    "medium": ["Please look into it this week.", "It's annoying but I have a workaround.", "",
               "It happens a few times a day.", "Not critical, but it needs fixing.", "I can live with it for now."],
    "low": ["No rush, just letting you know.", "Whenever you get a chance.", "Just a question, not a problem.",
            "Only curious.", "Low priority on my side.", "FYI for your team."],
}
PRIORITY_ORDER = ["low", "medium", "high", "urgent"]


def _pool(items: list, heldout: bool, every: int) -> list:
    return [x for i, x in enumerate(items) if (i % every == every - 1) == heldout]

OPENERS = ["", "Hi, ", "Hello support, ", "Hey team - ", "Good morning. ", "Ugh. "]
CLOSERS = ["", " Thanks.", " Thanks in advance!", " - Sam", " Regards, Priya", " Cheers"]


def _ticket(rng: random.Random, heldout: bool = False) -> dict[str, Any]:
    cat, tmpl, refund = rng.choice(_pool(ISSUES, heldout, 4))
    product = rng.choice(PRODUCTS)
    priority = rng.choice(PRIORITY_ORDER)
    cues = [rng.choice(_pool(PRIORITY_CUES[priority], heldout, 3))]
    if cues[0] and rng.random() < 0.3:  # second, LOWER-priority cue; the higher one must win
        lower = PRIORITY_ORDER[:PRIORITY_ORDER.index(priority)]
        if lower:
            cues.insert(rng.randint(0, 1), rng.choice([c for c in _pool(PRIORITY_CUES[rng.choice(lower)], heldout, 3) if c]))
    issue = tmpl.format(p=product)
    parts = [issue, *[c for c in cues if c]] if rng.random() < 0.7 else [*[c for c in cues if c], issue]
    text = f"{rng.choice(OPENERS)}{' '.join(parts)}{rng.choice(CLOSERS)}".strip()
    text = text[0].upper() + text[1:]
    label = {"category": cat, "priority": priority, "product": product, "refund_requested": refund}
    return {
        "messages": [{"role": "system", "content": TICKET_SYSTEM}, {"role": "user", "content": text}],
        "target": json.dumps(label),
        "label": label,
    }


def _split(make, n_train: int, n_val: int, n_test: int, seed: int, key) -> dict[str, list[dict]]:
    """Generate unique examples, then split. Dedup BEFORE splitting = no test leakage."""
    rng = random.Random(seed)
    seen: set[str] = set()
    out: list[dict] = []
    total = n_train + n_val + n_test
    tries = 0
    while len(out) < total and tries < total * 50:
        tries += 1
        ex = make(rng)
        k = key(ex)
        if k not in seen:
            seen.add(k)
            out.append(ex)
    return {"train": out[:n_train], "val": out[n_train:n_train + n_val], "test": out[n_train + n_val:total]}


def ticket_splits(n_train: int = 2000, n_val: int = 200, n_test: int = 200, seed: int = 0) -> dict[str, list[dict]]:
    """train: seen phrasings. val/test: HELD-OUT issue templates and priority cues."""
    key = lambda ex: ex["messages"][1]["content"]  # noqa: E731
    train = _split(_ticket, n_train, 0, 0, seed, key)["train"]
    held = _split(lambda r: _ticket(r, heldout=True), n_val + n_test, 0, 0, seed + 100, key)["train"]
    return {"train": train, "val": held[:n_val], "test": held[n_val:]}


def parse_json(text: str) -> dict | None:
    """Lenient: take the first {...} block. A base model often wraps JSON in prose or ```."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def grade_ticket(output: str, label: dict) -> dict[str, bool]:
    """Per-field exact match + 'exact' (all fields) + 'valid' (parseable JSON)."""
    pred = parse_json(output)
    if pred is None:
        return {"valid": False, "exact": False, **{k: False for k in label}}
    fields = {k: pred.get(k) == v for k, v in label.items()}
    return {"valid": True, "exact": all(fields.values()), **fields}


# Wrong-but-plausible answers, for DPO "rejected" fillers (lesson 09).
def corrupt_ticket(label: dict, rng: random.Random) -> str:
    bad = dict(label)
    kind = rng.choice(["priority", "category", "refund", "prose"])
    if kind == "priority":
        bad["priority"] = rng.choice([p for p in PRIORITY_ORDER if p != label["priority"]])
    elif kind == "category":
        bad["category"] = rng.choice([c for c in ("billing", "technical", "shipping", "account") if c != label["category"]])
    elif kind == "refund":
        bad["refund_requested"] = not label["refund_requested"]
    else:
        return "Sure! Here is the triage:\n```json\n" + json.dumps(bad, indent=2) + "\n```"
    return json.dumps(bad)


# ---------------------------------------------------------------------------
# Task 2: arithmetic word problems (verifiable reward)
# ---------------------------------------------------------------------------

MATH_SYSTEM = (
    "Solve the problem step by step. On the last line write the word 'Answer:' followed by the final number."
)
# Prompt-design note: earlier versions said "Answer: <integer>", "Answer: 42" and "Answer: N".
# Qwen3-0.6B copied each placeholder literally ("Answer: <57>", "Answer: 42", "Answer: N").
# Small models copy examples; describe the format instead of showing a fillable template.

_NAMES = ["Ava", "Ben", "Chen", "Dara", "Eli", "Fatima", "Gus", "Hana"]
_ITEMS = ["apples", "stickers", "marbles", "books", "coins", "cards"]


def _math(rng: random.Random) -> dict[str, Any]:
    n, it = rng.choice(_NAMES), rng.choice(_ITEMS)
    # Sized so the base model is "hard but possible" (~30-60%): GRPO needs groups with mixed rewards.
    a, b, c, k = rng.randint(40, 400), rng.randint(12, 99), rng.randint(7, 60), rng.randint(3, 13)
    form = rng.randint(0, 3)
    if form == 3:
        q = f"{n} starts with {a + k * b} {it}, sells {k} boxes of {b} {it} each, then buys {c} more. How many {it} now?"
        return {"messages": [{"role": "system", "content": MATH_SYSTEM}, {"role": "user", "content": q}],
                "target": f"Answer: {a + c}", "answer": a + c}
    if form == 0:
        q, ans = f"{n} has {a} {it}, buys {b} more, then gives away {c}. How many {it} does {n} have?", a + b - c
    elif form == 1:
        q, ans = f"{n} buys {k} packs of {it} with {b} in each pack and already had {a}. How many {it} now?", k * b + a
    else:
        q, ans = (f"{n} has {a * k} {it} and shares them equally among {k} friends, then each friend gets "
                  f"{c} more. How many {it} does each friend have?"), a + c
    return {"messages": [{"role": "system", "content": MATH_SYSTEM}, {"role": "user", "content": q}],
            "target": f"Answer: {ans}", "answer": ans}


def math_splits(n_train: int = 1000, n_test: int = 100, seed: int = 1) -> dict[str, list[dict]]:
    return _split(_math, n_train, 0, n_test, seed, key=lambda ex: ex["messages"][1]["content"])


def extract_answer(text: str) -> int | None:
    """Only an 'Answer: N' line counts (trailing units ok). Where you draw this line IS reward design."""
    import re
    m = re.findall(r"^\W*Answer:\W*(-?\d+)", text.strip(), flags=re.MULTILINE)
    return int(m[-1]) if m else None


# ---------------------------------------------------------------------------
# Task 3: tool calling
# ---------------------------------------------------------------------------

TOOLS: list[dict] = [
    {"type": "function", "function": {
        "name": "get_order_status", "description": "Look up an order by id.",
        "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}}},
    {"type": "function", "function": {
        "name": "convert_currency", "description": "Convert an amount between currencies (ISO codes).",
        "parameters": {"type": "object", "properties": {
            "amount": {"type": "number"}, "from_ccy": {"type": "string"}, "to_ccy": {"type": "string"}},
            "required": ["amount", "from_ccy", "to_ccy"]}}},
    {"type": "function", "function": {
        "name": "create_ticket", "description": "Open a support ticket.",
        "parameters": {"type": "object", "properties": {
            "product": {"type": "string"}, "summary": {"type": "string"}}, "required": ["product", "summary"]}}},
]

TOOL_SYSTEM = "You are a support agent. Call a tool when one is needed; otherwise answer directly and briefly."

_CCY = [("dollars", "USD"), ("euros", "EUR"), ("pounds", "GBP"), ("yen", "JPY")]


def _toolcall(rng: random.Random, negative: bool = False) -> dict[str, Any]:
    kind = 3 if negative else rng.randint(0, 2)
    if kind == 0:
        oid = f"A{rng.randint(10000, 99999)}"
        q = rng.choice([f"Where is my order {oid}?", f"Status of order {oid} please", f"Has {oid} shipped yet?"])
        call = {"name": "get_order_status", "arguments": {"order_id": oid}}
    elif kind == 1:
        amt = rng.choice([5, 20, 49.5, 100, 250, 1200])
        (fw, fc), (tw, tc) = rng.sample(_CCY, 2)
        q = rng.choice([f"How much is {amt} {fw} in {tw}?", f"Convert {amt} {fc} to {tc}"])
        call = {"name": "convert_currency", "arguments": {"amount": amt, "from_ccy": fc, "to_ccy": tc}}
    elif kind == 2:
        p = rng.choice(PRODUCTS)
        issue = rng.choice(["keeps crashing", "won't turn on", "lost all my settings"])
        q = f"My {p} {issue}. Please open a ticket."
        call = {"name": "create_ticket", "arguments": {"product": p, "summary": f"{p} {issue}"}}
    else:
        p = rng.choice(PRODUCTS)
        q = rng.choice([f"Is {p} available on Linux?", f"Does {p} come with a warranty?", f"Thanks, {p} works now!",
                        f"Do you ship {p} to {rng.choice(['Canada', 'Japan', 'Brazil', 'Kenya'])}?",
                        f"What colors does {p} come in?", f"Hi, quick question about {p}: is there a manual online?",
                        "What are your support hours?", "Hi there", "Thanks, that's all!"])
        q = rng.choice(["", "Hi, ", "Quick question: ", "Hello! ", "Hey - "]) + q
        call = None
    ex: dict[str, Any] = {"messages": [{"role": "system", "content": TOOL_SYSTEM}, {"role": "user", "content": q}],
                          "tools": TOOLS, "call": call}
    if call is None:
        ex["assistant"] = {"role": "assistant", "content": "Happy to help! Our team can answer that - "
                           "support is available 9am-6pm on weekdays."}
    else:
        ex["assistant"] = {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": call}]}
    return ex


def toolcall_splits(n_train: int = 1200, n_test: int = 120, seed: int = 2, neg_frac: float = 0.25) -> dict[str, list[dict]]:
    """Tool calls + ~25% "no tool needed" negatives, split proportionally (both pools deduped separately)."""
    key = lambda ex: ex["messages"][1]["content"]  # noqa: E731
    n_neg_tr, n_neg_te = int(n_train * neg_frac), int(n_test * neg_frac)
    pos = _split(_toolcall, n_train - n_neg_tr, 0, n_test - n_neg_te, seed, key)
    neg = _split(lambda r: _toolcall(r, negative=True), n_neg_tr, 0, n_neg_te, seed + 1, key)
    out = {}
    for k in ("train", "test"):
        out[k] = pos[k] + neg[k]
        random.Random(seed).shuffle(out[k])
    return {"train": out["train"], "val": [], "test": out["test"]}


def parse_tool_call(text: str) -> dict | None:
    """Qwen format: <tool_call>{"name": ..., "arguments": {...}}</tool_call>."""
    s, e = text.find("<tool_call>"), text.find("</tool_call>")
    if s < 0 or e < 0:
        return None
    try:
        obj = json.loads(text[s + len("<tool_call>"):e])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) and "name" in obj else None


def grade_toolcall(output: str, call: dict | None) -> dict[str, bool]:
    pred = parse_tool_call(output)
    if call is None:  # should answer directly, NOT call a tool
        ok = pred is None and "<tool_call>" not in output
        return {"valid": ok, "right_tool": ok, "exact": ok}
    valid = pred is not None
    right = valid and pred.get("name") == call["name"]
    exact = right and pred.get("arguments") == call["arguments"]
    return {"valid": valid, "right_tool": right, "exact": exact}


if __name__ == "__main__":
    for name, splits in [("tickets", ticket_splits()), ("math", math_splits()), ("toolcalls", toolcall_splits())]:
        print(f"{name}: " + ", ".join(f"{k}={len(v)}" for k, v in splits.items()))
        ex = splits["train"][0]
        print("  user  :", ex["messages"][1]["content"])
        print("  target:", ex.get("target") or ex.get("assistant"))
