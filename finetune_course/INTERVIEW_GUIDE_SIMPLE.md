# LLM fine-tuning: the simple version

A plain-language companion to [INTERVIEW_GUIDE.md](INTERVIEW_GUIDE.md). Each
answer is the short idea first, then the one number worth remembering. When
an interviewer pushes for detail, switch to the full guide. Numbers in
parentheses are lesson numbers. "Measured" means it came from a run on this
machine (RTX 5080, Qwen3-0.6B/4B).

---

## 1. The big picture

**Pretraining, fine-tuning, alignment: what's the difference?**

- **Pretraining:** read the internet and guess the next word, trillions of
  times. The model learns facts and skills.
- **Fine-tuning (SFT):** same "guess the next word" game, but only on
  examples of good answers. The model learns *how to answer*: format, style,
  tone.
- **Alignment (RLHF, DPO, GRPO):** teach the model which of its possible
  answers is *better*.

Think of it as school (pretraining), then job training (SFT), then a
performance review (alignment).

**When should I fine-tune instead of prompting or RAG?**

- **Try prompting first.** It's cheap and easy to change.
- **Use RAG for facts.** Facts change and need sources. Fine-tuning facts
  in makes the model confidently wrong and the facts go stale.
- **Fine-tune for behaviour.** Strict formats, house rules, tone, tool
  syntax, or making a small cheap model do a big model's job.
  - Measured (05): house rules 0.15 → 0.68. (11): tool calls 0.28 → 1.00.

Rule of thumb: **RAG changes what the model knows; fine-tuning changes how it
acts.**

**Full fine-tuning vs LoRA (PEFT)?**

- **Full:** change every weight. Best quality ceiling, but needs huge memory,
  huge checkpoint files, and forgets more.
- **LoRA:** freeze the model and train a tiny add-on (~1–2% of params).
  Small memory, megabyte-sized files, forgets less.
- On small tasks LoRA matched full fine-tuning here. On big hard new domains
  (lots of code or math), full still wins. "LoRA learns less and forgets
  less."

**What does fine-tuning actually change?**
Mostly *which existing skills the model uses and how it formats them*. It
mostly doesn't teach brand-new abilities. That's why ~1,000 great examples
can be enough (LIMA), and why **data quality matters most**.

---

## 2. Memory

**How much GPU memory does training need?**
Rule: **full fine-tuning ≈ 16 bytes per parameter** (weights + gradients +
two Adam numbers per weight), plus activations.

- 0.6B model: ~9 GB (measured 9.24 vs 8.88 estimated).
- 9B model: ~144 GB for full. ~18 GB for LoRA. ~5–6 GB for QLoRA.
- Just *running* a model (inference): ~2 bytes/param in bf16, ~0.5 in 4-bit.

**What's the biggest memory hog people forget?**
**The logits.** For every token the model scores all ~152k vocabulary words.
Batch 8 × 512 tokens in fp32 = 2.5 GB for one tensor. Fused cross-entropy
(Liger) avoids ever storing it. Measured (14): 18.2 → 8.6 GB.

**Gradient checkpointing?**
Throw away most in-between results and recompute them when needed. Less
memory, ~20–35% slower. Use it when you're still out of memory.

**bf16 vs fp16?**

- **bf16:** big range, low precision. No special tricks needed. Use it on
  modern GPUs.
- **fp16:** more precision, small range. Small gradients round to zero, so it
  needs a "loss scaler".
- Keep the *master copy* of weights in fp32. Measured (16): pure bf16 0.27
  vs fp32 master 0.60. Tiny updates got rounded away.

**FlashAttention?**
Same attention math, done in small tiles so the giant score table is never
written to memory. Faster and uses much less memory.

**Gradient accumulation?**
Can't fit batch 32? Run 4 batches of 8 and add up the gradients before
stepping. Catch: divide by the *total* number of tokens, not per mini-batch.

**Windows gotcha (measured):** a full GPU doesn't crash on Windows; it
silently spills into system RAM and gets 10–50× slower. Always free models
between runs.

---

## 3. Data

**Chat template?**
The exact text format the model sees (`<|im_start|>user ... <|im_end|>`). If
training format ≠ serving format, quality silently drops. **Train text must
match serve text exactly.**

**Label masking?**
Only grade the model on the *answer*, not the question: set prompt labels to
-100 so they're ignored. Keep the end-of-turn token in, or the model never
learns to stop talking.

**Padding side?**
Pad on the right for training, on the left for generation (new tokens must
come right after the real text).

**Packing?**
Glue short examples together to avoid wasted padding. Do it properly
(position resets + attention that can't see across examples), or examples
leak into each other.

**How much data?**
Hundreds to a few thousand *good, varied* examples. Include negatives
("no tool needed"), or the model always calls a tool. Test on **new
phrasings**, not just new rows, so you know it learned the rule and didn't
memorise strings.

**Synthetic data recipe (08)?**
Pick the label first in code, then ask the LLM to write an input for it, so
labels are right by construction. Then: validate, deduplicate, remove
anything too close to the test set, balance classes.

**Contamination?**
Test questions sneaking into training data. It makes scores fake. Check
n-gram overlap.

---

## 4. LoRA and friends

**Explain LoRA in one breath.**
Freeze the big weight matrix W. Learn a small correction made of two skinny
matrices: ΔW = B·A. B starts at zero, so training starts exactly at the
original model. For Qwen3-0.6B, r=16 is 10M trainable params (1.7%).

**What does LoRA save, and what doesn't it?**
It saves gradients and optimizer memory for the frozen weights. It does
**not** save activation memory, because data still flows through the full
model. Measured: 5.6 GB vs ~15 GB for full.

**Rank (r) and alpha?**
r = how big the correction can be (8–64 is typical). alpha/r = how loud it
is. **Training all linear layers matters more than a big rank.** Learning
rate is the first thing to tune.

**QLoRA?**
Store the frozen model in 4-bit (NF4), train LoRA on top in bf16. Lets a
model 4× bigger fit. Slower, because weights get unpacked every step.
Measured: 4B model 7.5 GB → 2.5 GB.

**Merging?**
Add B·A into W: one normal model, no extra cost at inference. Or keep
adapters separate so one base model serves many tasks.

**Combining two adapters (17)?**
Two unrelated tasks (tickets, tool calls) could just be added together:
both kept their scores (0.72 and 1.00). Pruning too aggressively (DARE 30%)
hurt one of them.

**LoRA variants, one line each:**

- **DoRA:** splits size and direction; a point or two better, slower.
- **PiSSA / OLoRA:** smarter starting point; converges faster.
- **LoRA+:** B gets a bigger learning rate than A.
- **rsLoRA:** scaling that keeps big ranks useful.
- **(IA)³ / prompt tuning:** tiny and weak.

Lesson (15): a variant that changes the step size needs its own learning
rate. LoRA+ at the normal lr blew up (0.02).

---

## 5. Training behaviour

**Learning rates to remember:** full FT ~1e-5, LoRA ~2e-4, DPO ~5e-7 to
5e-6, RL ~1e-6. Warm up, then decay. Clip gradients at 1.0.

**Epochs?** 1–3. More just memorises.

**Overfitting?** Train loss → 0 while val loss goes back up. Measured (03):
best val at step 12, then worse. Fix: early stopping, more varied data,
lower lr, fewer steps.

**Is low val loss enough?** No. Measure the real task. Measured (02): val
loss 0.06 but only 50% accuracy.

**Catastrophic forgetting?**
Training on one thing breaks other things. Measured (07): a ticket LoRA took
arithmetic from **0.675 to 0.00**, but perplexity barely moved. So perplexity
isn't enough; test behaviour. Fixes: mix in 5–20% general data, train less,
or only send in-domain requests to the adapter.

**Loss spikes?** Lower lr, longer warmup, clip, check for broken data, use bf16.

---

## 6. Alignment

**RLHF (PPO)?**
Train a "judge" model on human preferences, then train the model to please
the judge without drifting too far from the original. Needs 4 models in
memory. Powerful but fiddly.

**DPO?**
Skip the judge. Show "good answer vs bad answer" pairs and push the model
toward the good one directly. Only 2 models (the policy and a frozen copy).
Do SFT first: DPO only re-ranks answers the model can already give.
Measured (09): 0.68 → 0.78.

**Reference model with LoRA?** Just turn the adapter off. Merge the SFT
adapter first, so "off" = the SFT model.

**DPO variants:** KTO (thumbs up/down instead of pairs), ORPO and SimPO (no
reference model), online DPO (regenerate pairs each round).

**GRPO?**
For each question, sample several answers, score them with a checker (is
the math right? do the tests pass?), and push toward the above-average ones.
The group average replaces a critic model. If all answers score the same,
nothing is learned, so tasks must be hard but possible. Measured (10): math
0.72 → 0.86.

**Funny lesson (10):** a 0.6B model copies whatever template you show it.
"Answer: <integer>" gave "Answer: <57>". Describe the format; don't show a
fill-in-the-blank.

**Reward hacking?**
The model games the reward. A loose "is the right number anywhere?" reward
paid off for printing 0 to 199. Make rewards strict and read samples.

**GRPO fixes (18):** Dr. GRPO removes length and difficulty biases. DAPO
drops useless groups and keeps exploration up. GSPO works per sequence,
not per token. Honest answer: "one seed and 16 steps is noise; I'd run
several seeds."

**RL vs SFT?** SFT copies fixed answers. RL learns from its own attempts and
mostly turns "sometimes right" into "usually right". RL is much more
expensive, because generating answers is slow.

---

## 7. Distillation

**What is it?** A big teacher model trains a small student.

- **Logit KD:** the student copies the teacher's full probability spread,
  including which wrong answers are "almost right". Needs the same
  tokenizer.
- **Sequence KD:** the teacher writes answers and the student trains on them.
  Works with any teacher, even an API. This is how most "distilled" models
  are made.
- **On-policy (GKD):** the student writes, the teacher grades.

**Measured (12):** logit KD 0.67 beat plain SFT 0.60. Adding teacher-written
labels *hurt* (0.48), because the teacher's mistakes came along. **A student
can't beat its teacher's blind spots.**

---

## 8. Speed and memory tricks: what to grab first on one 16 GB GPU

1. **LoRA** (QLoRA if it doesn't fit).
2. **Fused cross-entropy** (Liger): huge memory win. Measured 18 → 8.6 GB.
3. **Packing**, but only with FlashAttention-2. Without it, it was 0.37× the
   speed here.
4. **Gradient checkpointing** for long sequences.
5. **8-bit / paged Adam** for full FT. Measured 11.2 → 5.8 GB, same loss.
6. **Spectrum / GaLore / APOLLO** if you need full-parameter training on a
   small GPU.
7. **Keep fp32 master weights.** FP8 only helps on big matmuls (3× *slower*
   on a 0.6B here).

Other names to recognise: **NEFTune** (noise on embeddings, no effect on
narrow tasks), **TIES/DARE** (merge models without training), **Unsloth**
(faster LoRA kernels), **Muon** (new optimizer), **QAT** (train while
simulating quantisation).

---

## 9. More than one GPU

- **DDP:** every GPU has a full copy, and they average gradients.
- **ZeRO / FSDP:** split optimizer state, then gradients, then weights
  across GPUs. Memory per GPU ≈ 1/N.
- **Tensor parallel:** split each matrix across GPUs (needs fast links).
- **Pipeline parallel:** split the layers across GPUs.
- **70B fine-tune?** QLoRA on one 48–80 GB GPU, or LoRA + FSDP on a node.

---

## 10. Evaluation and serving

**How do you evaluate?**

1. The real task, on a clean held-out set.
2. New kinds of inputs (unseen phrasings).
3. Regressions: did other skills break?
4. Error bars: n=200 is about ±7%.

**LLM-as-judge?** For open-ended answers. Use a rubric, swap answer order,
watch for "longer = better" bias, and check the judge against humans.

**Serving?** Merge and serve as a normal model, or serve one base with many
LoRA adapters in vLLM. **Check that served output matches PyTorch output.**

**Quantising after?** Merge in bf16, then quantise (AWQ/GPTQ) with task data,
then re-run the eval.

---

## 11. Your story in 30 seconds

> I fine-tuned Qwen3-0.6B to turn support tickets into strict JSON with house
> rules. The base model scored 0.15 on unseen phrasings. I built label-first
> synthetic data, tested on phrasings it had never seen, and trained LoRA
> r=16 on all linear layers. It reached 0.68 in 60 steps. Then I tried DPO,
> distillation from a 4B, and LoRA variants, checked it didn't break other
> skills, and served it on vLLM next to a tool-calling adapter. Biggest
> lessons: training text must match serving text exactly, test on new
> phrasings, and validate output in code.
