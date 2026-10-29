# LLM fine-tuning course (local, RTX 5080 16 GB)

> Copied from the `model_run` lab, where this folder is `finetune_course/`. Paths below
> use that layout: `setup/05_setup_finetune_env.sh` is `setup_finetune_env.sh` here.
> `server/serve.sh` (vLLM, used by lessons 08 and 13) and `langgraph_course/common.py`
> (used by lesson 08; see `../Langggraph/common.py`) live in the lab repo.

Runnable scripts from basics to current efficient techniques, built on the
HF stack (transformers + PEFT + TRL). Every script opens with an
**"Interview topics"** docstring: read that first, run it, then change
things and break them. `INTERVIEW_GUIDE.md` collects the short answers to
know from memory, with numbers measured on this machine.

## Setup (WSL, separate env)

Training runs in a **separate conda env `ft`** in WSL. vLLM pins torch in the
`deepseek` env, and training libraries move fast; separate envs keep either
one from breaking the other.

```bash
# WSL, project root, once:
bash setup/05_setup_finetune_env.sh

# STOP the vLLM server first: it reserves ~85% of the GPU.
conda activate ft
python finetune_course/00_memory_math.py
python finetune_course/run_all.py            # every lesson, --quick, outcome checks
```

Every lesson takes `--quick` (minutes) or runs full-length without it.
Outputs (adapters, merged models, logs) go to `finetune_course/outputs/`, which
is gitignored.

The exceptions:
- **08** runs on **Windows** (`.venv`) against the **running** vLLM server:
  `.venv\Scripts\python.exe finetune_course\08_synthetic_data_vllm.py`
- **13 `--serve-check`** needs `bash server/serve.sh ft` (Qwen3-0.6B + the course's LoRA adapters).

## The running task

`data.py` generates seeded synthetic support tickets → JSON
`{category, priority, product, refund_requested}`, graded by exact match in
code. Labels follow **house rules a base model can't guess** (priority cues,
"the higher priority wins", refund wording). **val/test use issue phrasings
and priority cues never seen in training**, so scores measure learning the
rule, not memorising strings. There are two more tasks: arithmetic word
problems with a verifiable answer (RL, lessons 10/18) and tool calling
(lessons 11/17).

Base models: Qwen3-0.6B (most lessons, fast), Qwen3-4B (QLoRA, teacher). All
Qwen3 sizes share one tokenizer and chat template.

## Lessons

### Basics → core techniques
| # | Script | Core idea |
|---|--------|-----------|
| 00 | `00_memory_math.py` | 16 B/param rule, estimate vs measured VRAM, why a 9B full FT can't fit |
| 01 | `01_tokenization_chat_templates.py` | Chat templates, `<think>` block, -100 label masking, EOS, BPE boundary trap, padding vs packing |
| 02 | `02_raw_pytorch_loop.py` | Causal-LM loss by hand, correct grad accumulation, warmup+cosine, bf16 autocast, clipping |
| 03 | `03_full_finetune_trainer.py` | HF `Trainer` full FT; overfitting curve on purpose |
| 04 | `04_lora_from_scratch.py` | LoRA layer written by hand, merge, PEFT parity, targets × rank budget |
| 05 | `05_sft_trl.py` | `SFTTrainer` + LoRA, completion-only loss, lr/rank sweep |
| 06 | `06_qlora_4bit.py` | NF4 vs FP4 error, double quant, paged AdamW, QLoRA on Qwen3-4B |
| 07 | `07_evaluation_forgetting.py` | Task vs generalisation vs forgetting; base / LoRA / merged / full FT |
| 08 | `08_synthetic_data_vllm.py` | Label-first synthetic data with the local 9B: validate, dedup, decontaminate, preference pairs |
| 09 | `09_dpo_preference.py` | DPO on top of SFT with on-policy negatives; reward margins/accuracies |
| 10 | `10_grpo_verifiable_rewards.py` | GRPO with verifiable rewards; reward hacking demo |
| 11 | `11_tool_calling_finetune.py` | Tool-call fine-tuning for agents (right tool, exact args, no false calls) |
| 12 | `12_distillation.py` | 4B teacher → 0.6B: hard labels vs logit KD vs sequence-level KD |
| 13 | `13_merge_export_serve.py` | Merge/export, parity check, multi-LoRA serving on vLLM |

### Advanced: current efficient techniques (each prints a comparison table)
| # | Script | Core idea |
|---|--------|-----------|
| 14 | `14_efficient_kernels.py` | SDPA vs eager, Liger (fused linear CE), padding-free packing, grad checkpointing, activation offloading, torch.compile, 8-bit/paged AdamW |
| 15 | `15_lora_variants.py` | LoRA vs rsLoRA vs DoRA vs PiSSA vs OLoRA vs LoRA+ vs (IA)³ |
| 16 | `16_memory_frontier.py` | GaLore, APOLLO, Spectrum (SNR layer selection), FP8 (torchao), NEFTune |
| 17 | `17_model_merging.py` | Task arithmetic: linear vs cat vs TIES vs DARE-TIES adapter merges |
| 18 | `18_rl_efficiency.py` | GRPO vs Dr. GRPO vs DAPO vs GSPO; paged / vLLM rollouts |

Dependencies between lessons: 05 → 07/09/13/17, 06 → 12, 11 → 13/17.
`run_all.py` runs them in order.

## Scoring

```bash
python finetune_course/run_all.py                       # all but 08, quick
python finetune_course/run_all.py 05 07                 # selected
python finetune_course/run_all.py --full --save results_ft_full.json
python finetune_course/run_all.py --compare a.json b.json
```

Checks are on **outcomes**: accuracy after > before, DPO reward accuracy > 0.5,
GRPO reward rises, Liger lowers peak VRAM, the merged adapter keeps both
skills. Full logs are written to `outputs/logs/<lesson>.log`.

## Gotchas found while building this

- **Windows/WSL sysmem fallback:** when VRAM fills up, the driver spills to
  system RAM instead of raising OOM, and training silently gets 10–50× slower.
  `common.load_model` warns if less than half the VRAM is free (vLLM still up,
  or a leaked model). Always `del model, trainer` and then `free_gpu()`.
- **transformers 5:** `warmup_ratio` is gone; `warmup_steps` accepts a float < 1 as a ratio.
- **The first test set was too easy:** with one shared phrasing pool, 40 steps
  reached 100%, which made every later comparison meaningless. Holding out
  phrasings fixed it.
- **Merging into bf16** rounds away part of the LoRA update (04 shows a logit
  diff of 0.75). Merge in fp32, or serve unmerged, if parity matters.
- **flash-attn** has no prebuilt Blackwell wheel for current torch; SDPA is used.
- **TRL 1.x SFTTrainer already uses a chunked "fused LM head"** (logits never
  materialised) and turns off Liger's fused cross-entropy. Liger's big memory
  win only shows with a plain `Trainer` (14, section 1).
- **Small models copy prompt placeholders:** `Answer: <integer>` produced
  "Answer: <57>"; `Answer: 42` produced "Answer: 42" for every problem. Describe
  the format instead of showing a template (`data.MATH_SYSTEM`).
- **Pure-bf16 full fine-tuning** (no fp32 master weights) at lr 1e-5 learned
  far less (16), because updates round to zero.
- **torch.compile** failed under this torch/PEFT combination; lesson 14 reports
  it as a result instead of crashing.
