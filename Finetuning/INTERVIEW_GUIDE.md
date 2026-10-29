# LLM fine-tuning: interview guide

Short answers to know from memory. Numbers in parentheses point to the
lesson that demonstrates the topic. Where a claim was **measured here**
(RTX 5080 16 GB, Qwen3-0.6B/4B, `--quick` runs), the number comes from
`outputs/logs/`. Everything else is from the papers and is tagged *read about*.

---

## 1. Fundamentals

**Pretraining vs fine-tuning vs alignment?**
Pretraining is next-token prediction on trillions of tokens, which gives
knowledge and general skills. Supervised fine-tuning (SFT) is the same loss on
curated (prompt, ideal answer) pairs, and teaches format, style, task
conventions and tool syntax. Preference tuning (RLHF / DPO) and RL with
verifiable rewards (GRPO) then shape *which* of the model's possible answers it
prefers. Modern post-training is SFT, then preference tuning and/or RL, often
over several rounds.

**When do you fine-tune instead of prompting or using RAG?**
- Prompting first: it's cheapest, and the instructions are visible and easy to change.
- RAG when the problem is *knowledge* (facts that change, private documents,
  citations needed). Fine-tuning is a poor way to add facts: models hallucinate
  around facts they learned this way, and the facts go stale.
- Fine-tune when the problem is *behaviour*: a strict output format, house
  rules a prompt can't fully specify (05: base 0.15, LoRA 0.68 on held-out
  phrasings, **measured here**), a domain tone, a tool schema (11: exact tool
calls 0.28 → 1.00 after 40 steps), or cost
  and latency (a fine-tuned 0.6B instead of a prompted 9B, as in 12).
- They combine: a fine-tuned model can be better at *using* retrieved context.

**Full fine-tuning vs PEFT?**
Full FT updates every weight. It has the highest quality ceiling, but costs
about 16 B/param of memory, produces one full checkpoint per experiment, and
forgets the most. PEFT (LoRA and friends) trains well under 2% of the
parameters: memory is a few GB, adapters are megabytes, it forgets less, and
many adapters can be served on one base. In quick runs here, LoRA matched or
beat full FT on the target task (03 vs 05). On hard new domains (code, math
with lots of data), full FT still wins: "LoRA learns less and forgets less",
Biderman et al. 2024.

**What does a fine-tune actually change?**
Mostly the output distribution over *existing* capabilities: style, format,
and which behaviours get selected. A useful mental model is the "superficial
alignment hypothesis" (LIMA): about 1k curated examples were enough for chat
behaviour. This is also why data quality dominates.

---

## 2. Memory and compute (00, 06, 14)

**How much memory does training need?**
Weights + gradients + optimizer states + activations + temporary buffers.
Mixed-precision AdamW needs ~16 B/param: fp32 master weights (4) + grads (4) +
Adam m and v (4 + 4). **Measured here (00):** 0.6B full FT after `opt.step()`
was 9.24 GB against an estimate of 8.88 GB (ratio 1.04).
- 9B full FT ≈ 144 GB before activations. LoRA bf16 9B ≈ 18 GB. QLoRA 9B ≈ 5–6 GB.
- Inference needs only weights + KV cache: 2 B/param in bf16, about 0.5 B/param in int4.

**What dominates activation memory?**
Activations scale with batch × seq × hidden × layers (~34·s·b·h bytes per layer).
Attention scores add 5·a·s²·b, but FlashAttention/SDPA never materialise them.
**The logits** (b × s × vocab) are often the largest single tensor: Qwen's
vocab is 151,936, so an 8×512 batch in fp32 is 2.5 GB. That is why fused
linear cross-entropy (Liger, Cut-CE) helps so much.

**Gradient checkpointing?**
Keep only some activations (block inputs) and recompute the rest during
backward. Activation memory drops a lot for ~20–35% more compute.
`use_cache=False` is required while it is on. Use it when you're still out of
memory after cheaper fixes (14).

**Mixed precision: bf16 vs fp16?**
bf16 has fp32's 8 exponent bits, so its dynamic range is the same and it needs
no loss scaling. It has only 8 mantissa bits, so it is imprecise: **measured
here (16)**, full FT with weights *and* updates in bf16 reached 0.27, against
0.60 with fp32 master weights. fp16 has more
mantissa but a small range, so it needs a GradScaler to avoid gradient
underflow. On Ampere and newer GPUs, use bf16. Keep master weights and
optimizer states in fp32 for stability. Side effect, **measured here (04):**
merging a LoRA into bf16 weights rounds part of the small update away (logits
moved by up to 0.75).

**FlashAttention?**
An exact attention kernel that tiles Q, K and V into on-chip SRAM and never
writes the s×s score matrix to HBM. Memory is O(s) and it runs 2–4× faster.
FA2 is the widely deployed version, FA3 targets Hopper, and PyTorch SDPA picks
a flash-style kernel automatically. Here flash-attn had no Blackwell wheel, so
SDPA was used.

**Gradient accumulation?**
Effective batch = micro-batch × accumulation steps × GPUs. It's the same
gradient as the big batch if the loss is normalised by the *total* number of
trained tokens, not averaged per micro-batch. Transformers fixed exactly this
bug in 2024, and 02 implements it by hand.

**Windows/WSL gotcha (measured here):** when VRAM is full, the NVIDIA driver
on Windows does not raise OOM. It silently spills to system RAM ("sysmem
fallback"), and training gets 10–50× slower. A leaked model reference turned a
30 s run into 653 s. Always `del model, trainer`, then `gc.collect()` and
`torch.cuda.empty_cache()`.

---

## 3. Data (01, 08)

**What is a chat template, and why does it matter?**
The Jinja template in `tokenizer_config.json` turns messages into the token
sequence the model reads (Qwen uses ChatML: `<|im_start|>role\n…<|im_end|>`).
If you train with a different format than you serve with, quality drops
silently. Qwen3's non-thinking mode inserts an empty `<think>\n\n</think>`
block, and training data must contain it too (01).

**What is label masking / completion-only loss?**
Labels are the input ids with prompt positions set to -100, which PyTorch
cross-entropy ignores, so the loss covers only the assistant answer. The
end-of-turn token must be in the labels, or the model never learns to stop.
Tokenise prompt and completion separately: BPE can merge across the join
(01: `"Label: "+"positive"` tokenised as one string makes `' positive'`, a
different token).

**Padding side?**
Right-pad for training. Left-pad for batched generation, because new tokens
must directly follow each row's last real token.

**Packing?**
Concatenate examples into full-length rows to remove padding. Correct packing
is "padding-free": position ids reset per example, plus varlen attention
(FA2) or a block-diagonal mask (SDPA), so examples can't attend to each other.
Naive packing leaks context between examples. TRL's `packing=True` uses
best-fit-decreasing (BFD) packing, padding-free.

**How much data, and what kind?**
Quality and diversity beat volume. A few hundred to a few thousand good
examples teach a format or behaviour; LIMA used 1k. Cover edge cases and
*negatives*: lesson 11 includes 25% "no tool needed" examples, or the model
learns to always call a tool. Hold out *phrasings*, not just rows: the test set
here uses issue templates and priority cues never seen in training, which
separates learning the rule from memorising strings.

**Synthetic data pipeline (08)?**
Generate label-first (sample the label in code, have the LLM write a matching
input), so labels are correct by construction. Then validate with code checks,
dedup (exact plus near-duplicate via n-gram Jaccard or MinHash), decontaminate
against the test set (n-gram overlap), and balance classes. Check the
teacher's license before training on its outputs.

**Contamination?**
Test items, or near-duplicates of them, in training data inflate scores. Remove
them with n-gram overlap (8- to 13-grams are common) or embedding similarity.
It matters for public benchmarks too: many are partly in pretraining data.

---

## 4. Parameter-efficient fine-tuning (04, 05, 06, 15)

**Explain LoRA.**
Freeze W and learn ΔW = (α/r)·B·A with A ∈ ℝ^{r×d_in} and B ∈ ℝ^{d_out×r}.
A starts random and B starts at zero, so training begins exactly at the base
model (04 checks this: logit diff 0.0). Trainable parameters per layer are
r·(d_in+d_out) instead of d_in·d_out. For Qwen3-0.6B, r=16 on all linear layers
is 10.1M params, 1.7% (04).

**Why does it save memory, and what doesn't it save?**
There are no gradients or Adam states for W. Activations are still full size,
because the forward pass still runs through W, so long sequences still need
checkpointing. **Measured here:** LoRA peak 5.6 GB vs ~15 GB for full FT (02/04).

**Rank and alpha?**
α/r scales the update, so changing r doesn't require retuning lr. Common
choices are α = r or 2r. rsLoRA uses α/√r, which keeps large ranks useful (15).
Typical r is 8–64. Targeting all linear layers usually matters more than rank
(QLoRA paper). Learning rate is the first thing to sweep (05).

**Which modules to target?**
The original paper targeted q and v. Current practice is all linear layers
(q, k, v, o, gate, up, down), which was needed to match full FT in QLoRA.
Embeddings and lm_head are trained only when adding tokens.

**QLoRA?**
The base is frozen in 4-bit NF4 and LoRA adapters are trained in bf16 on top.
NF4 places its 16 levels at the quantiles of a normal distribution, which
pretrained weights roughly follow, so error is lower than with uniform
int4/fp4 (06 measures NF4 < FP4 RMSE). It adds double quantization (the scales
are quantized too) and paged optimizers (spill to CPU instead of OOM). It is
slower than bf16 LoRA because weights are dequantised for every matmul.
**Measured here:** Qwen3-4B weights take 7.5 GB in bf16 and ~2.5 GB in NF4.

**Merging adapters?**
Merging (W += (α/r)BA) gives a plain model with zero inference overhead.
Merge into a bf16/fp32 base, never into 4-bit weights. Keeping adapters
unmerged lets many adapters share one base (multi-LoRA serving, 13).

**Combining two adapters without training (17), measured here:**

| | tickets | tool calls |
|---|---|---|
| tickets adapter | 0.68 | 0.28 |
| tools adapter | 0.13 | 1.00 |
| `cat` merge (exact Σ BA) | 0.72 | 1.00 |
| TIES (density 0.5) | 0.65 | 1.00 |
| DARE-TIES (density 0.3) | 0.40 | 1.00 |

These two tasks barely interfere (different prompts: tool schemas present or
not), so a plain sum works. Aggressive dropping (DARE at 30% density) hurt the
"weaker" adapter.

**LoRA variants (15)?**
- **DoRA:** a learned magnitude vector plus a LoRA-updated direction. Often +1–2 points; ~20–40% slower training; no cost after merge.
- **PiSSA / OLoRA:** initialise from the principal singular vectors or a QR of W. Faster convergence; they modify the base weights.
- **LoRA+:** a higher lr for B than for A (ratio ~16). Free speed-up.
- **rsLoRA:** α/√r scaling.
- **LoftQ:** a quantisation-aware init for QLoRA.
- **EVA / CorDA:** data-driven inits.
- **LoRA-FA:** A frozen.
- **(IA)³:** learned scaling vectors, ~0.01% of params, weaker.
- **Prompt/prefix tuning:** learned virtual tokens, the weakest option on small models.

**Measured here (15, 30 steps, same data; one seed, so ±~0.05 is noise):**
LoRA 0.53, attention-only 0.38 (targets matter), r=64 0.50, rsLoRA r=64 0.52,
DoRA 0.53 (30% slower), PiSSA 0.57 (lr 5e-5), OLoRA 0.63 (lr 5e-5), LoRA+ 0.53
(lr 5e-5), (IA)³ 0.32 with 0.14M params. The first run used LoRA+ at base lr
2e-4, which gave B an effective lr of 3.2e-3; it **diverged (0.02)**. Variants
that change the effective step size need their own lr.

---

## 5. Training dynamics (02, 03, 07)

**Learning rates?**
Full FT ~1e-5. LoRA ~1e-4 to 3e-4. DPO 5e-7 to 5e-6 (full) or ~5e-6 to 5e-5
(LoRA). RL ~1e-6 (full) or ~1e-5 (LoRA). Use warmup (Adam's second moment is
noisy early) and then cosine or linear decay. Clip gradients at 1.0.

**How many epochs?**
1–3 for SFT. More epochs on small data memorise.

**How do you recognise overfitting?**
Train loss goes to 0 while val loss bottoms out and rises. **Measured here
(03):** val loss was best (0.12) at step 12 and ended at 0.23, while train loss
fell to 3e-6. Fixes: early stopping (`load_best_model_at_end`), more and more
diverse data, lower lr, fewer epochs, LoRA, dropout, NEFTune.

**Val loss vs task metric?**
Loss is a proxy. Always also measure the task metric (exact match, pass@1,
judge score). In 02 the val loss reached ~0.06 while accuracy on held-out
phrasings was only 0.50.

**Catastrophic forgetting (07)?**
Fine-tuning degrades skills you didn't train. Measure it with a regression set
(here: arithmetic plus perplexity on generic text). **Measured here:** a
60-step ticket LoRA took arithmetic accuracy from **0.675 to 0.00**: the model
stopped reasoning and replied in ~27 tokens. Generic perplexity barely moved
(9.20 → 8.99). Perplexity is a weak alarm; you need a *behavioural*
regression suite. LoRA did not protect here either: a narrow dataset with one
fixed output format teaches "always answer this way". Mitigations: mix 5–20%
general data into training ("replay"), use lower lr and fewer steps, add a KL
penalty toward the base, merge or average with the base, or route only
in-domain requests to the adapter (multi-LoRA serving).

**Loss spikes or divergence?**
Lower the lr, lengthen warmup, clip gradients, check the data (very long
examples, broken labels, all -100 rows), and use bf16 rather than fp16.

---

## 6. Alignment: preferences and RL (09, 10, 18)

**RLHF with PPO?**
Train a reward model on human comparisons (a Bradley-Terry loss), then
maximise reward − β·KL(π‖π_ref) with PPO. That needs four models in memory
(policy, reference, reward, value/critic). It's powerful but unstable and
expensive.

**DPO?**
The KL-constrained optimum has a closed form, so the reward model can be
dropped. The loss is
`-log σ(β[(log π(y_w) − log π_ref(y_w)) − (log π(y_l) − log π_ref(y_l))])`.
Only the policy and a frozen reference are needed, and training looks like
supervised learning. The implicit reward is β·log(π/π_ref). Watch
`rewards/accuracies` and `rewards/margins`. A known quirk: chosen log-probs can
fall even while the margin grows. Do SFT first; DPO only re-ranks what the
policy can already produce.

**How do you get the reference model with PEFT?**
Disable the adapter. Merge the SFT adapter first so that "adapter off" means
the SFT model, not the base (09).

**DPO variants?**
- **IPO:** regularised, avoids overfitting deterministic preferences.
- **KTO:** learns from unpaired good/bad labels.
- **ORPO:** SFT plus an odds-ratio term in one stage, with no reference model.
- **SimPO:** length-normalised, with no reference model.
- **Online / iterative DPO:** regenerate pairs from the current policy every round.

Use on-policy negatives (the model's own mistakes) where possible (09).
**Measured here:** DPO on top of SFT went 0.68 → 0.78, with reward accuracy
0.96. The SFT model made only 4 mistakes on 600 *training-phrasing* prompts, so
almost all pairs were synthetic corruptions: on-policy mining finds nothing
when the policy has memorised the training distribution. Mine on held-out-style
prompts.

**GRPO?**
Sample G completions per prompt, score each with a reward, and set the
advantage to (r − group mean) / group std. The group baseline replaces the
critic. The loss is PPO-clip, with optional KL. It is used with
**verifiable rewards** (math answers, unit tests, schema validation): RLVR,
the recipe behind R1-style reasoning models. Groups where every sample gets the
same reward give no signal, so tasks must be "hard but possible".

**Measured here (10):** 25 GRPO steps took math accuracy 0.72 → 0.86 and
format compliance 0.88 → 1.00, while completions got *shorter* (111 → 84
tokens). It also showed, three times, that a 0.6B copies whatever placeholder
the prompt shows: `Answer: <integer>` → "Answer: <57>", `Answer: 42` →
"Answer: 42" for every problem, `Answer: N` → "Answer: N". Describe the format;
don't show a fillable template. Prompt bugs look like model failures.

**Reward hacking?**
The policy maximises the reward as written, not as intended. In 10, the naive
"right number appears somewhere" reward pays 1.0 for printing 0…199. Make
rewards strict, keep format bonuses small, inspect samples, and add length
controls.

**What is wrong with original GRPO (18)?**
- **Dr. GRPO:** per-sequence length normalisation creates a length bias (long
  wrong answers are penalised less per token), and std scaling adds a
  difficulty bias. Remove both.
- **DAPO:** token-level loss across the batch, clip-higher (ε_high 0.28, which
  keeps entropy up), dynamic sampling (drop zero-variance groups), and
  overlong masking/shaping.
- **GSPO:** sequence-level importance ratios and clipping. More stable,
  especially for MoE.
- **CISPO:** clip the importance weight instead of dropping token gradients.

TRL now defaults to the DAPO loss with β = 0 (no reference model).
**Measured here (18, 16 steps each, base 0.72):** GRPO 0.74, Dr. GRPO 0.94,
DAPO 0.86, GSPO 0.80. Paged generation cut wall time from 74 s to 52 s. At 16
steps and one seed this ranking is noise. The honest interview answer is
"RL comparisons need several seeds and longer runs; I'd watch length and
entropy curves, not one final number".

**RL vs SFT?**
SFT imitates fixed targets (off-policy). RL learns from the model's own samples
(on-policy) and can go beyond the demonstrations. In practice RL mostly turns
pass@k into pass@1, and it is far more expensive per step, mostly because of
generation.

---

## 7. Distillation and small models (12)

**Logit KD vs sequence-level KD?**
- **Logit (token) KD:** the student matches the teacher's full next-token
  distribution: α·T²·KL(p_t^T ‖ p_s^T) + (1−α)·CE. It needs the same
  vocabulary, and it transfers "dark knowledge" (which wrong answers are
  close).
- **Sequence-level KD:** the teacher generates outputs and the student does
  SFT on them. It works across tokenizers and with API-only teachers, and it
  can label unlabeled inputs. This is how most "distilled" open models were
  made, including DeepSeek-R1-Distill (SFT on R1 traces, no RL).

**On-policy distillation (GKD)?**
The student generates and the teacher scores the student's own tokens. This
fixes exposure bias. Reverse KL is mode-seeking.

**Why distil?**
Serve a 0.6B at near-4B quality for a task (12 compares 200 gold labels,
logit KD, and teacher-labelled extra data).

**Measured here (quick mode):** teacher (40-step QLoRA 4B) 0.57; student SFT
on 200 gold 0.60; logit KD on the same 200 0.67; SFT on 200 gold + 600
teacher-labelled 0.48. Soft targets helped even from a weak teacher. Adding
the teacher's pseudo-labels *hurt*, because they were on training phrasings
(the student already had those) and carried the teacher's 6% errors. KD can't
beat a teacher's blind spots.

---

## 8. Latest efficient techniques (2024–26)

Each entry gives what it is, why it helps, when to use it, and the trade-off.
*Measured* entries have numbers from 14–18 (see `outputs/logs/`).

| Technique | What / why | When | Trade-off | Status |
|---|---|---|---|---|
| **Liger kernels** | Triton-fused RMSNorm/RoPE/SwiGLU + **fused linear cross-entropy**: logits never materialised | Always worth trying; big vocab, long seq | Needs model support (most popular archs) | measured (14) |
| **Padding-free packing (BFD)** | Concatenate examples, reset position ids, varlen / block-diagonal attention | Variable-length data (chat, code) | Needs FA2 for full speed-up; with SDPA it's correct but uses a dense mask | measured (14) |
| **FlashAttention-2/3** | Tiled exact attention, O(s) memory | Long sequences | FA3 Hopper-only; build pain on new GPUs | read about (no Blackwell wheel here) |
| **Gradient checkpointing / activation offloading** | Recompute or move activations to CPU | Still OOM after other fixes | ~20-35 % slower / PCIe bound | measured (14) |
| **8-bit / paged AdamW** | Adam states in 8-bit block-wise quantisation | Full FT, big models | Negligible quality loss | measured (14) |
| **torch.compile** | Graph capture + fused kernels | Long runs, static shapes | Compile time, recompiles on dynamic shapes | measured (14) |
| **QLoRA (NF4)** | 4-bit frozen base + LoRA | Model too big for bf16 LoRA | ~20-40 % slower | measured (06) |
| **DoRA / PiSSA / LoRA+ / rsLoRA** | Better LoRA parametrisation / init / lr | Need the last few points or faster convergence | DoRA slower; PiSSA changes base weights | measured (15) |
| **GaLore / APOLLO** | Full-rank updates, low-rank (or random-projected) optimizer state | Want full-param learning on small GPU | Slower steps (SVD), newer, less tooling | measured (16) |
| **Spectrum** | Train only top-SNR layers (random-matrix theory) | Full-FT quality, less memory, no adapter | Choosing the fraction; still full checkpoints | measured (16) |
| **FP8 training (torchao)** | float8 matmuls with dynamic scaling | Large matmuls on H100/Blackwell | No gain for small models; numerics care | measured (16) |
| **NEFTune** | Noise on embeddings during SFT | Chat SFT, generalisation | Tiny effect on narrow tasks | measured (16) |
| **TIES / DARE merging** | Merge task vectors with less interference | Combine adapters/fine-tunes without training | Needs same base; tune density | measured (17) |
| **Dr. GRPO / DAPO / GSPO / CISPO** | Fix GRPO's length/difficulty biases, clipping | Any RLVR run | Many knobs; RL needs repeated seeds | measured (18) |
| **vLLM rollouts in RL** | Fast generation engine colocated with training | RL (generation dominates) | Extra env/pins; memory split | read about (18 prints the config) |
| **Unsloth** | Hand-written Triton kernels + manual backprop for LoRA/QLoRA | Single-GPU LoRA, 2x speed / less memory claims | Wraps HF API; less transparent | read about |
| **Cut Cross-Entropy (Apple)** | Like Liger's fused CE, computes CE over vocab blocks in SRAM | Huge vocab models | Kernel availability | read about |
| **Muon optimizer** | Orthogonalised momentum (Newton-Schulz) for 2-D weights | Pretraining / large FT, ~1.3-2x sample-efficiency claims | New; mixing with AdamW for embeddings | read about |
| **Sequence / context parallelism** (ring attention, Ulysses) | Split one long sequence across GPUs | 32k-1M token training | Multi-GPU only | read about |
| **FP4 (NVFP4/MXFP4)** | 4-bit float tensor cores on Blackwell | Inference now; training research-stage | Accuracy risk | read about |
| **QAT** | Simulate quantisation during fine-tuning | You'll deploy at int4/fp8 | Extra training cost | read about |
| **LongLoRA** | Shifted sparse attention during FT to extend context cheaply | Context extension | Approximation during training | read about |

**Measured here (quick runs, Qwen3-0.6B, RTX 5080):**

| Experiment | Result |
|---|---|
| Plain `Trainer`, batch 32: full logits vs **Liger** fused linear CE (14) | peak **18.2 → 8.6 GB**, 3.0k → 11.3k tok/s (the 18 GB run spilled past VRAM into system RAM) |
| TRL `SFTTrainer` baseline vs + Liger (14) | no change: TRL's own fused LM head is already on |
| Padding-free packing with SDPA (14) | **0.37×** speed: no varlen kernel, so a dense mask over a 4096-token row (quadratic). Needs FA2 to pay off; our data had only 5.5% padding anyway |
| Gradient checkpointing / activation offloading (14) | ≈0 / −0.2 GB: activations are tiny at 32×130 tokens. These matter for long sequences, not here |
| Full FT optimizer: AdamW vs 8-bit vs paged 8-bit (14) | **11.2 → 7.2 → 5.8 GB**, same loss (0.236 / 0.232 / 0.232) |
| Full FT fp32-master 12.4 GB, GaLore 10.4 GB, APOLLO 9.7 GB, Spectrum-25% 5.9 GB, LoRA 3.5 GB (16) | accuracy 0.60 / 0.42 / 0.50 (0.75 in an earlier run: seed noise) / 0.57 / 0.53 |
| Pure-bf16 full FT vs fp32 master weights (16) | **0.27 vs 0.60**: lr 1e-5 updates round away in bf16 |
| FP8 (torchao) vs bf16 full FT (16) | 3× *slower* (28 s vs 9 s), 0.20 vs 0.27: 0.6B matmuls are too small, and no torch.compile |
| NEFTune α=5 (16) | 0.52 vs 0.53: no effect on a narrow extraction task |
| torch.compile (14) | failed under this torch + PEFT combination (reported, not crashed) |

**What to reach for first on one 16 GB GPU** (in this order, based on these numbers):
1. LoRA (QLoRA if the bf16 base doesn't fit).
2. A fused cross-entropy (Liger, or your trainer's built-in fused head; check
   which is active).
3. Packing for variable-length data, but only with a varlen attention kernel (FA2).
4. Gradient checkpointing for long sequences, or if still OOM.
5. 8-bit / paged AdamW if doing full FT.
6. Spectrum or GaLore/APOLLO if you need full-parameter learning.
7. Keep fp32 master weights. FP8 only pays off on large matmuls.

---

## 9. Scaling beyond one GPU (read about)

**DDP?**
Every GPU holds a full replica, and gradients are all-reduced every step.

**ZeRO 1/2/3 (DeepSpeed) / FSDP (PyTorch)?**
They shard optimizer states (stage 1), plus gradients (stage 2), plus
parameters (stage 3, which FSDP's FULL_SHARD matches). Memory per GPU falls
roughly as 1/N at stage 3, at the cost of more communication. Offload variants
push state to CPU or NVMe.

**Tensor parallelism?**
Split each matmul across GPUs (Megatron). It needs fast interconnect (NVLink)
and is used inside a node.

**Pipeline parallelism?**
Split the layers into stages and feed micro-batches through them to fill the
"bubble". Used across nodes.

**3D/4D parallelism?**
DP × TP × PP (× context/sequence parallelism for long sequences, × expert
parallelism for MoE).

**What would you use to fine-tune 70B?**
QLoRA on one 48–80 GB GPU, or LoRA plus FSDP/ZeRO-3 across a node. Full FT
needs a cluster (~1.1 TB of states).

---

## 10. Evaluation and deployment (07, 13)

**How do you evaluate a fine-tune?**
1. Target task on a held-out, decontaminated test set with a deterministic
   grader where possible.
2. Generalisation to shifted inputs (07: unseen product names; the test set
   uses unseen phrasings).
3. Regressions: general benchmarks or a fixed regression suite, safety refusals.
4. Report confidence intervals: n=200 exact match is roughly ±7% at 95%.

**When do you use an LLM-as-judge?**
For open-ended output. Use a rubric and a stronger judge, swap positions for
pairwise comparisons (position bias), control for length bias, and validate
the judge against human labels.

**How do you serve a fine-tune?**
Either merge the adapter and serve it as a normal model (it can then be
quantised with AWQ/GPTQ/FP8), or serve one base with many adapters (vLLM
`--enable-lora --lora-modules name=path`, with per-request `model=name`).
Check parity first: the served output must match the PyTorch output, since
template and sampling drift are common (13).

**Quantising after fine-tuning?**
Fine-tune in bf16 (or QLoRA), merge in bf16, then quantise with calibration
data from your task (AWQ/GPTQ). Re-run the task eval after quantising.

---

## 11. "Tell me about a fine-tune you did" (story template)

Fill in the numbers from your own `run_all.py --full` run:

> **Problem:** triage support tickets into strict JSON with house rules
> (priority cues, "higher priority wins", refund detection) that a prompted
> small model got wrong (base Qwen3-0.6B: **0.15** exact match on held-out
> phrasings).
> **Approach:** label-first synthetic data, a held-out test set built from
> *unseen phrasings*, and LoRA r=16 on all linear layers at lr 2e-4 with
> completion-only loss and the exact serving chat template.
> **Result:** **0.68** after 60 steps (quick); compare with full FT and QLoRA-4B
> (06), and confirm no regression on arithmetic or generic perplexity (07).
> **Iterations:** priority on unseen phrasings was the hardest field, so I
> tried DPO on the model's own mistakes (09), distillation from the 4B (12),
> and LoRA variants (15).
> **Deployment:** one vLLM server, base plus two adapters (tickets, tool calls)
> routed per request, with parity checked against PyTorch (13).
> **Lessons:** train text must match serve text byte-for-byte, measure on
> held-out phrasings rather than held-out rows, and add guardrails in code
> (validate the JSON, retry).
