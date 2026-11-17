"""
01_decode_math.py
-----------------
Pen-and-paper inference: predict decode speed, prefill time, KV-cache size and
max concurrency for the models in the local HF cache, from their exact
per-tensor byte counts (safetensors headers) and this GPU's measured
bandwidth/TFLOPs (outputs/gpu.json from lesson 00). No GPU needed; lesson 02
checks these predictions against the live vLLM server.

Run: WSL (reads ~/.cache/huggingface), any env with python 3.10+.
    python inference_course/01_decode_math.py

Interview topics
- Decode tok/s (batch 1) ceiling = bandwidth / bytes read per step. Bytes per step =
  all layer weights + lm_head (NOT the embedding table: a lookup reads one row).
- Why "params x bits/8" is wrong for quantized checkpoints: embeddings/lm_head often
  stay BF16. Qwen3.5-9B w4a16: the BF16 lm_head is ~25% of the bytes each token reads.
- KV cache per token = 2 (K,V) x layers x kv_heads x head_dim x bytes. GQA shrinks it by
  heads/kv_heads; hybrid linear-attention models (Qwen3.5: 1 full-attention layer in 4)
  shrink it again, and pay a fixed per-sequence recurrent state instead.
- Batched decode step = weights once + every sequence's KV. At long context the KV reads
  dominate the weight reads, so throughput stops scaling with batch (attention is memory-bound too).
- Max concurrency is a KV-memory question: (VRAM x util - weights - activations) / (ctx x kv/token).
- Prefill is compute-bound (AI ~ prompt length), decode is memory-bound (AI ~ batch):
  TTFT ~ 2 x params x prompt / FLOPs, TPOT ~ bytes / bandwidth. That asymmetry is the
  reason for chunked prefill and for prefill/decode disaggregation.
"""

from __future__ import annotations

from common import (
    GB,
    MODELS,
    GiB,
    kv,
    load_gpu,
    load_model_info,
    parse_args,
    prefill_ms,
    roofline_step,
    section,
    table,
)

args = parse_args(__doc__)
gpu = load_gpu()
print(f"GPU numbers ({'measured, lesson 00' if gpu['measured'] else 'SPEC FALLBACK'}): "
      f"{gpu['bw_gbps']:.0f} GB/s, {gpu['tflops_bf16']:.0f} TFLOPS bf16")

infos = {p: i for p in MODELS if (i := load_model_info(p))}
if not infos:
    raise SystemExit("no model of MODELS found in the HF cache; run in WSL where serve.sh downloaded them")
TAG = {"qwen35": "qwen35", "qwen3": "qwen3-8b", "deepseek": "deepseek14b", "qwen3-0.6b": "qwen3-06b"}

# ---------------------------------------------------------------------------
section("1. What one decode step reads (exact bytes from safetensors headers)")
# ---------------------------------------------------------------------------
rows = []
for p, i in infos.items():
    w = i["w"]
    rows.append({"profile": p, "layers_GB": w["layers"]["bytes"] / GB, "lm_head_GB": w["lm_head"]["bytes"] / GB,
                 "embed_GB(skip)": w.get("embed", {}).get("bytes", 0) / GB,
                 "other_GB(skip)": sum(w[c]["bytes"] for c in ("vision", "mtp") if c in w) / GB,
                 "read/step_GB": i["decode_bytes"] / GB, "params_B": i["decode_params"] / 1e9,
                 "lm_head_share": w["lm_head"]["bytes"] / i["decode_bytes"]})
table(rows)
print("\nembed is skipped: a lookup reads 1 row. vision (--language-model-only) and MTP heads (spec decoding only)")
print("are never read in normal decode. A tied model (Qwen3-0.6B) reads the embedding matrix as its lm_head.")

# ---------------------------------------------------------------------------
section("2. Batch-1 decode ceiling: tok/s = bandwidth / bytes per step")
# ---------------------------------------------------------------------------
rows = []
for p, i in infos.items():
    s = roofline_step(i, gpu, batch=1, ctx=1024)
    rows.append({"profile": p, "step_ms": s["step_ms"], "ceiling_tok_s": s["tok_s"], "t_mem_ms": s["t_mem_ms"],
                 "t_comp_ms": s["t_comp_ms"], "bound": s["bound"]})
    kv(f"decode-{TAG[p]}", read_gb=i["decode_bytes"] / GB, lm_head_share=i["w"]["lm_head"]["bytes"] / i["decode_bytes"],
       ceiling_tok_s=s["tok_s"], mem_over_comp=s["t_mem_ms"] / s["t_comp_ms"])
table(rows)
print("\nAt batch 1 the step is ~40-120x longer on memory than on compute: the tensor cores mostly wait.")
if "qwen35" in infos:
    i = infos["qwen35"]
    lm = i["w"]["lm_head"]["bytes"]
    s0 = roofline_step(i, gpu, 1, 1024)["tok_s"]
    i4 = dict(i, decode_bytes=i["decode_bytes"] - lm * 0.75)  # if lm_head were int4 too (+ ~scales, ignored)
    s4 = roofline_step(i4, gpu, 1, 1024)["tok_s"]
    print(f"qwen35: quantizing just the BF16 lm_head to 4-bit would raise the ceiling {s0:.0f} -> {s4:.0f} tok/s "
          f"(+{(s4 / s0 - 1) * 100:.0f}%): a Phase-2 experiment (quality cost unknown until measured).")
    kv("lmhead-int4-whatif", before=s0, after=s4, gain=s4 / s0)

# ---------------------------------------------------------------------------
section("3. KV cache: bytes per token and per sequence")
# ---------------------------------------------------------------------------
rows = []
for p, i in infos.items():
    c = i["cfg"]
    mha = 2 * c["num_hidden_layers"] * c["num_attention_heads"] * c["head_dim"] * 2   # if no GQA and no hybrid
    rows.append({"profile": p, "layers": c["num_hidden_layers"], "heads/kv": f"{c['num_attention_heads']}/{c['num_key_value_heads']}",
                 "head_dim": c["head_dim"], "KV_KiB/tok": i["kv_tok"] / 1024, "MHA_KiB/tok": mha / 1024,
                 "state_MiB/seq": i["state_seq"] / 2**20, "32k_seq_GiB": (32768 * i["kv_tok"] + i["state_seq"]) / GiB})
    kv(f"kv-{TAG[p]}", kib_per_tok=i["kv_tok"] / 1024, mha_kib_per_tok=mha / 1024, state_mib=i["state_seq"] / 2**20,
       gib_32k=(32768 * i["kv_tok"] + i["state_seq"]) / GiB)
table(rows)
print("\nGQA divides KV by heads/kv_heads. Qwen3.5 keeps KV in only 8 of 32 layers (Gated DeltaNet in the rest),")
print("and pays a fixed recurrent state per sequence instead (dtype assumed bf16 here).")

# ---------------------------------------------------------------------------
section("4. Batched decode: when do KV reads overtake weight reads?")
# ---------------------------------------------------------------------------
rows = []
for p in [p for p in ("qwen3", "qwen35") if p in infos]:
    i = infos[p]
    for ctx in (1024, 8192, 32768):
        for b in (1, 4, 16):
            s = roofline_step(i, gpu, b, ctx)
            kvb = b * (ctx * i["kv_tok"] + i["state_seq"])
            rows.append({"profile": p, "ctx": ctx, "batch": b, "kv/weights": kvb / i["decode_bytes"],
                         "step_ms": s["step_ms"], "tok_s": s["tok_s"], "bound": s["bound"]})
            if ctx in (1024, 32768) and b in (1, 4):
                kv(f"batch-{TAG[p]}-c{ctx}-b{b}", tok_s=s["tok_s"], kv_over_weights=kvb / i["decode_bytes"])
table(rows)
print("\nShort context: batch 4 gives ~4x tokens/s for free. 32k context on Qwen3-8B: the KV cache of 4 sequences")
print("weighs ~4x the model, so batching buys much less. (Ignores whether it even FITS - see next section.)")

# ---------------------------------------------------------------------------
section("5. Does it fit? KV budget under vLLM's --gpu-memory-utilization 0.85")
# ---------------------------------------------------------------------------
vram = 15.9 * GiB  # RTX 5080 usable
ACT = 1.5 * GiB    # rough: activations, CUDA graphs, sampler buffers (vLLM profiles this at startup)
rows = []
for p, i in infos.items():
    w = i["w"]
    resident = sum(w[c]["bytes"] for c in ("layers", "lm_head", "embed") if c in w)
    if i["cfg"].get("tie_word_embeddings"):
        resident -= w["lm_head"]["bytes"]  # same tensor counted twice
    budget = 0.85 * vram - resident - ACT
    tokens = budget / i["kv_tok"] if budget > 0 else 0
    rows.append({"profile": p, "weights_GiB": resident / GiB, "kv_budget_GiB": budget / GiB, "kv_tokens": int(tokens),
                 "seqs@32k": tokens / 32768, "seqs@8k": tokens / 8192})
table(rows)
print("\nCompare with the server log line 'GPU KV cache size: N tokens' when serve.sh starts.")
print("Hybrid models also reserve state slots per sequence, so the real number for qwen35 is lower.")

# ---------------------------------------------------------------------------
section("6. Prefill vs decode: TTFT is compute, TPOT is bandwidth")
# ---------------------------------------------------------------------------
rows = []
for p in [p for p in ("qwen35", "qwen3") if p in infos]:
    i = infos[p]
    tpot = roofline_step(i, gpu, 1, 1024)["step_ms"]
    for prompt in (128, 1024, 4096, 16384):
        ms = prefill_ms(i, gpu, prompt)
        rows.append({"profile": p, "prompt": prompt, "prefill_ms": ms, "per_token_us": ms * 1e3 / prompt,
                     "decode_step_ms": tpot, "prefill_tok_per_decode_step": prompt / ms * tpot})
    kv(f"prefill-{TAG[p]}", ms_1k=prefill_ms(i, gpu, 1024), ms_4k=prefill_ms(i, gpu, 4096), tpot_ms=tpot,
       tokens_per_decode_step_time=1024 / prefill_ms(i, gpu, 1024) * tpot)
table(rows)
print("\nIn the time one batch-1 decode step makes 1 token, prefill could push ~40-60 prompt tokens through the same")
print("weights. These are CEILINGS at 100% of peak; real kernels get less -> lesson 02 measures how much.")
