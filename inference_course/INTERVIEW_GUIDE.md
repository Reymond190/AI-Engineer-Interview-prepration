# LLM inference optimization: interview guide

Short answers to know from memory. **Measured** numbers come from this lab:
RTX 5080 16 GB, vLLM, `qwen3` profile (Qwen3-8B-AWQ) unless noted. The lesson
number is in parentheses; rerun the lesson to reproduce the number.

---

## 1. The roofline (00)

**What limits an inference kernel?**
Either memory bandwidth or compute. Attainable FLOP/s = min(peak FLOP/s,
bandwidth × arithmetic intensity), where AI = FLOPs / bytes moved. The ridge
point (peak / bandwidth) separates the two regimes. **Measured:** 889 GB/s read
(93% of the 960 GB/s spec; copy reaches 747 GB/s), 112 TFLOPS BF16 → ridge
≈ 126 FLOP/byte.

**AI of a matmul Y = X[M,K] · W[K,N]?**
2MNK / (bytes of X + W + Y). In 16-bit, with M ≪ N, K, it is ≈ M. Decode has
M = batch, so batch-1 decode runs at AI ≈ 1, two orders of magnitude below the ridge.

**Why is batching nearly free in decode?**
The weight read is shared by every row of X. **Measured** (8192², BF16): M=1
takes 0.18 ms, and M=64 still takes 0.20 ms. Time starts growing between
M=128 and M=256, near the predicted ridge (~126). Per token that is 184 µs at
M=1 vs 1.4 µs at M=256.

**Why do kernel fusion and FlashAttention matter?**
Elementwise ops have AI < 1, so each unfused op is a full read and write of
the tensor. **Measured:** relu(x·w+b) on 512 MiB runs in 5.4 ms eager
(3 kernels) and 2.7 ms with one fused torch.compile/Triton kernel, which is
88% of ideal bandwidth.

**FP8 vs BF16 on this GPU?**
**Measured:** 384 vs 112 TFLOPS (3.4×) for 8192³ matmuls. FP8 helps
compute-bound prefill. For memory-bound decode, what counts is how many bytes
the weights take.

**How do you time GPU code?**
CUDA is asynchronous, so time with CUDA events (or synchronize first), warm
up (autotuning, compilation, clocks) and take the median. Use buffers larger
than L2 (64 MiB here) when you want to measure DRAM.

## 2. Decode and KV-cache math (01)

**Upper bound on batch-1 decode speed?**
Bandwidth / bytes read per step. The bytes per step are all layer weights
plus the lm_head; the embedding table is excluded, because a lookup reads one
row. **Measured:** Qwen3-8B-AWQ reads 4.85 GB per step → 178–182 tok/s
ceiling. Qwen3.5-9B w4a16 reads 8.0 GB → 110 tok/s.

**Why is "params × bits / 8" wrong for quantized models?**
Embeddings and lm_head often stay BF16. In Qwen3.5-9B w4a16 the BF16 lm_head
(248k vocab × 4096) is 2.03 GB, **25%** of the bytes read per token. At 4-bit
it would raise the ceiling from 110 to 136 tok/s (+23%), quality cost not yet
measured.

**KV-cache size?**
2 × layers × kv_heads × head_dim × bytes per token. Qwen3-8B: 2·36·8·128·2 =
**144 KiB/token**, 4.5 GiB for a 32k sequence. GQA (32 query heads, 8 KV heads)
is 4× smaller than MHA. The hybrid Qwen3.5 keeps KV in only 8 of 32 layers:
**32 KiB/token**, plus a fixed ~24 MiB Gated-DeltaNet state per sequence.

**How many sequences fit?**
KV budget = VRAM × gpu_memory_utilization − weights − activations/CUDA graphs.
**Measured:** vLLM reported 7.45 GiB → 54,272 tokens → 1.66 sequences at 32k
for Qwen3-8B. My estimate (46k tokens) was 15% low because I assumed 1.5 GiB of
overhead and the real figure was about 0.4 GiB.

**When does batching stop paying in decode?**
When the KV reads outgrow the weight reads. At 32k context and batch 4,
Qwen3-8B reads 4× the model size in KV per step, so the roofline gain is only
1.6× (vs 3.7× at 1k context). Attention is memory-bound too. The fixes are
GQA/MLA, hybrid layers, KV quantization and sliding windows.

**Prefill vs decode?**
Prefill has AI ≈ prompt length, so it is compute-bound. Decode has AI ≈ batch,
so it is memory-bound. **Measured (02):** prefill costs 190 µs per prompt token
(75% MFU at 4k tokens) vs 8.4 ms per decode token, 44× more. That asymmetry
motivates chunked prefill (mixing the two in one step) and prefill/decode
disaggregation (separate GPUs).

## 3. Serving metrics and continuous batching (02)

**What metrics do you report?**
- TTFT (queueing + prefill), the latency a user feels first.
- TPOT/ITL, the streaming pace.
- e2e = TTFT + (n − 1) · TPOT.
- Throughput (output tok/s across all requests), which drives cost.
- Goodput (requests/s that meet the SLO), which is what actually matters.

Report p50/p95/p99, not means.

**How do you benchmark honestly?**
Stream and timestamp on the client, take token counts from the server's usage
block, fix the output length with `ignore_eos`, randomize the prompt prefix
(otherwise prefix caching fakes a fast TTFT; **measured** 803 → 22 ms, 37×,
on a repeated 4k prompt), and warm up first.

**Batch-1 serving vs the roofline?**
**Measured:** 120 tok/s (TPOT 8.4 ms) against a 182 tok/s ceiling, so **66%**
efficiency. The gap comes from GEMV kernels reaching ~80% of bandwidth, plus
per-step scheduler/CPU overhead, sampling, and the small ops (norm, rotary, KV
writes).

**What does continuous batching buy, and where does it break?**
New requests join the running batch at every step. **Measured**
(max-num-seqs 4, 128 output tokens): 105 → 419 tok/s from 1 to 4 streams,
with TPOT unchanged (9.7 → 8.8 ms). At 8 streams, throughput stays flat
(415 tok/s), TTFT p50 jumps from 214 ms to 2.6 s because requests queue for a
slot, and goodput (TTFT < 1 s, TPOT < 50 ms) **collapses from 1.64 to 0.20
req/s**. Raising throughput isn't the goal once requests start missing the SLO.
