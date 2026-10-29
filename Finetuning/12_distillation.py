"""
12 - Knowledge distillation: a 4B teacher -> 0.6B student
=========================================================

Interview topics
----------------
- Distillation trains a small student to imitate a big teacher. Why:
  serve 0.6B at 4B quality (latency/cost), or use a teacher to label data
  you don't have labels for.
- Two flavours:
    * Logit (token-level) KD, Hinton et al.: match the teacher's whole
      next-token distribution, not just the argmax:
          L = alpha * T^2 * KL(softmax(z_t/T) || softmax(z_s/T)) + (1-alpha) * CE(y)
      "Dark knowledge": the teacher's probabilities on wrong tokens carry
      information (which mistakes are close). T>1 softens; T^2 keeps grad
      scale comparable. Needs the SAME tokenizer/vocab (Qwen3 family: yes).
    * Sequence-level KD: the teacher GENERATES outputs, the student does
      plain SFT on them. Works across tokenizers and with API-only teachers
      (that's what most "distilled" open models are). Lets you use
      UNLABELED inputs.
- On-policy distillation (GKD, Agarwal et al. 2024; TRL GKDTrainer /
  DistillationTrainer): the student generates, the teacher scores the
  student's own tokens => fixes train/inference mismatch (exposure bias).
  Reverse KL is mode-seeking (student picks one good mode) vs forward KL
  mean-seeking.
- DeepSeek-R1-Distill models (one is in server/serve.sh) are sequence-level
  distillation of R1's reasoning traces into Qwen/Llama students: SFT only,
  no RL, and they beat RL-trained models of the same size.
- Memory: the teacher only needs a forward pass (no grads/states), and can
  be 4-bit. Or precompute teacher logits (top-k only: full vocab logits are
  ~600 KB per token in fp32).

Run:  python finetune_course/12_distillation.py [--quick]
Needs: outputs/06_qlora_4b (lesson 06)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import Trainer, TrainingArguments

from common import (BASE_QLORA, BASE_SMALL, LORA_TARGETS, OUT, encode_all, eval_tickets, free_gpu, generate,
                    load_model, out_dir, pad_collate, parse_args, render_prompt, require, section, table)
from data import parse_json, ticket_splits


class KDTrainer(Trainer):
    """Trainer whose loss mixes CE on gold tokens with KL to a frozen teacher's distribution."""

    def __init__(self, *a, teacher=None, alpha: float = 0.5, temperature: float = 2.0, **kw):
        super().__init__(*a, **kw)
        self.teacher, self.alpha, self.T = teacher, alpha, temperature

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        out = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        with torch.no_grad():
            t_logits = self.teacher(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]).logits
        labels = inputs["labels"][:, 1:]
        mask = labels != -100
        # Only the completion positions; slicing first keeps fp32 softmax memory small.
        V = min(out.logits.size(-1), t_logits.size(-1))
        s = out.logits[:, :-1][mask][:, :V].float()
        t = t_logits[:, :-1][mask][:, :V].float()
        kd = F.kl_div(F.log_softmax(s / self.T, -1), F.log_softmax(t / self.T, -1), log_target=True,
                      reduction="batchmean") * self.T**2
        ce = F.cross_entropy(s, labels[mask])
        loss = self.alpha * kd + (1 - self.alpha) * ce
        return (loss, out) if return_outputs else loss


def student_run(name: str, rows: list[dict], test, steps: int, teacher=None) -> dict:
    model, tok = load_model(BASE_SMALL)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, target_modules=LORA_TARGETS, task_type="CAUSAL_LM"))
    targs = TrainingArguments(output_dir=str(out_dir("_tmp_12")), per_device_train_batch_size=8,
                              gradient_accumulation_steps=2, learning_rate=2e-4, lr_scheduler_type="cosine",
                              warmup_steps=0.1, max_steps=steps, bf16=True, logging_steps=10, save_strategy="no",
                              report_to="none", remove_unused_columns=False, seed=0)
    kw = dict(model=model, args=targs, train_dataset=Dataset.from_list(rows), data_collator=pad_collate(tok.pad_token_id))
    trainer = KDTrainer(**kw, teacher=teacher) if teacher is not None else Trainer(**kw)
    trainer.train()
    m = eval_tickets(model, tok, test, name)
    del trainer, model
    free_gpu()
    return {"student": name, "train examples": len(rows), "accuracy": f"{m['exact']:.3f}", "_acc": m["exact"]}


def main() -> None:
    args = parse_args(__doc__)
    splits = ticket_splits()
    test = splits["test"][:60] if args.quick else splits["test"]
    n_gold, n_unlabeled = 200, (600 if args.quick else 1500)
    gold = splits["train"][:n_gold]                          # small labelled set
    unlabeled = splits["train"][n_gold:n_gold + n_unlabeled]  # labels ignored
    steps = 40 if args.quick else 120

    section("1. Teacher: Qwen3-4B (4-bit) + QLoRA adapter from lesson 06")
    teacher, tok = load_model(BASE_QLORA, quant4=True, adapter=require(OUT / "06_qlora_4b", "06"))
    teacher.eval()
    t = eval_tickets(teacher, tok, test, "teacher 4B")

    section("2. Sequence-level KD data: teacher labels the unlabeled tickets")
    outs = generate(teacher, tok, [render_prompt(tok, ex["messages"]) for ex in unlabeled])
    pseudo = []
    for ex, o in zip(unlabeled, outs):
        if (p := parse_json(o)) is not None:  # keep only well-formed teacher outputs
            pseudo.append({**ex, "target": o, "label": p})
    agree = sum(p["label"] == ex["label"] for p, ex in zip(pseudo, unlabeled)) / max(1, len(pseudo))
    print(f"[pseudo-labels] kept {len(pseudo)}/{len(unlabeled)}; teacher agrees with hidden gold {agree:.2f}")

    student_tok = tok  # same tokenizer family
    gold_rows = encode_all(student_tok, gold)
    rows = []

    section("3a. Student: SFT on 200 gold labels (hard targets)")
    rows.append(student_run("sft-gold", gold_rows, test, steps))

    section("3b. Student: logit KD on the same 200 (alpha=0.5, T=2)")
    rows.append(student_run("logit-KD", gold_rows, test, steps, teacher=teacher))

    section("3c. Student: SFT on 200 gold + teacher-labelled unlabeled data (sequence-level KD)")
    rows.append(student_run("seq-KD", gold_rows + encode_all(student_tok, pseudo), test, steps))

    section("Summary")
    rows.insert(0, {"student": "teacher 4B (QLoRA)", "train examples": "-", "accuracy": f"{t['exact']:.3f}"})
    table(rows, ["student", "train examples", "accuracy"])
    best = max(rows[1:], key=lambda r: r["_acc"])
    print(f"[distill] teacher={t['exact']:.3f} best_student={best['student']} acc={best['_acc']:.3f}")
    if t["exact"] <= best["_acc"]:
        print("Note: the teacher is no better than the students here (in --quick, lesson 06 trains it for only 40")
        print("steps). Distillation can only transfer what the teacher knows; seq-KD also copies its mistakes.")
        print("Run lesson 06 without --quick for a stronger teacher, then re-run this lesson.")
    del teacher
    free_gpu()


if __name__ == "__main__":
    main()
