"""
00_gpu_roofline.py
------------------
Measure the two numbers that bound every LLM inference workload on this GPU:
memory bandwidth (bytes/s) and matmul throughput (FLOP/s). Then watch a
decode-shaped matmul move from memory-bound to compute-bound as the batch grows.

Run: WSL, conda env `ft`, vLLM STOPPED (it holds the VRAM).
    python inference_course/00_gpu_roofline.py [--quick]

Writes outputs/gpu.json; lessons 01 and 02 build their predictions on it.

Interview topics
- The roofline model: attainable FLOP/s = min(peak FLOP/s, bandwidth x arithmetic intensity).
- Arithmetic intensity (AI) = FLOPs / bytes moved. Ridge point = peak FLOP/s / bandwidth:
  below it you are memory-bound, above it compute-bound.
- Y[M,N] = X[M,K] @ W[K,N] in 16-bit: AI = 2MNK / 2(MK + KN + MN) ~ M when M << N, K.
  Decode has M = batch size, so single-stream decode runs at AI ~ 1: pure bandwidth.
- Why batching is "free" in decode: until M reaches the ridge, the weights are
  read once regardless of M, so time per step barely moves while tokens/step grow.
- Elementwise ops have AI < 1, so they are always memory-bound. That's why kernel fusion matters:
  each unfused op costs a full read + write of the tensor.
- Spec-sheet vs achievable: measured copy bandwidth and matmul TFLOPs sit below the
  datasheet numbers; plan with measured numbers.
- Timing CUDA correctly: kernels are async; use CUDA events (or synchronize), warm up, take a median.
"""

from __future__ import annotations

import json

import torch

from common import GB, GPU_JSON, OUT, cuda_ms, free_gpu, kv, parse_args, section, table

args = parse_args(__doc__)
assert torch.cuda.is_available(), "needs a CUDA GPU (WSL env ft)"
dev = "cuda"
ITERS = 5 if args.quick else 20
free_gb = torch.cuda.mem_get_info()[0] / GB
if free_gb < 6:
    raise SystemExit(f"only {free_gb:.1f} GB VRAM free - stop vLLM first (it reserves ~85% of the GPU)")

props = torch.cuda.get_device_properties(0)
print(f"GPU: {props.name}  sm_{props.major}{props.minor}  {props.total_memory / GB:.1f} GB  "
      f"{props.multi_processor_count} SMs  L2 {getattr(props, 'L2_cache_size', 0) / 2**20:.0f} MiB")

# ---------------------------------------------------------------------------
section("1. Memory bandwidth: the decode speed limit")
# ---------------------------------------------------------------------------
# 1 GiB buffers: far bigger than L2, so we measure DRAM, not cache.
n = 2**29  # bf16 elements = 1 GiB
x = torch.randn(n, device=dev, dtype=torch.bfloat16)
y = torch.empty_like(x)
nbytes = x.numel() * x.element_size()

# Lambdas bind tensors as default args: they are deleted later to free VRAM between sections.
t_copy = cuda_ms(lambda x=x, y=y: y.copy_(x), iters=ITERS)
bw_copy = 2 * nbytes / (t_copy / 1e3) / GB                     # read + write
t_read = cuda_ms(lambda x=x: x.sum(dtype=torch.float32), iters=ITERS)
bw_read = nbytes / (t_read / 1e3) / GB                          # read only (reduction)
bw = max(bw_copy, bw_read)
print(f"copy 1 GiB: {t_copy:.2f} ms -> {bw_copy:.0f} GB/s (read+write)")
print(f"sum  1 GiB: {t_read:.2f} ms -> {bw_read:.0f} GB/s (read only)")
print("RTX 5080 spec: 960 GB/s (GDDR7, 256-bit @ 30 Gbps). Decode streams weights = reads, so we use the higher one.")
kv("bandwidth", copy_gbps=bw_copy, read_gbps=bw_read, used_gbps=bw, spec_gbps=960.0, frac_of_spec=bw / 960)
del x, y
free_gpu()

# ---------------------------------------------------------------------------
section("2. Matmul throughput per dtype: the prefill speed limit")
# ---------------------------------------------------------------------------
S = 8192
flops = 2 * S**3
res = {}
for name, dt in [("bf16", torch.bfloat16), ("fp16", torch.float16), ("fp32_tf32", torch.float32), ("fp32", torch.float32)]:
    torch.backends.cuda.matmul.allow_tf32 = name == "fp32_tf32"
    a = torch.randn(S, S, device=dev, dtype=dt)
    b = torch.randn(S, S, device=dev, dtype=dt)
    t = cuda_ms(lambda a=a, b=b: a @ b, iters=max(3, ITERS // 2))
    res[name] = flops / (t / 1e3) / 1e12
    del a, b
torch.backends.cuda.matmul.allow_tf32 = False

# FP8 (e4m3) tensor cores: Blackwell/Ada have them. torch._scaled_mm is the low-level entry point
# (B must be column-major, hence b.t() of a row-major [N, K]). Phase 2 uses this for quantization.
try:
    a8 = torch.randn(S, S, device=dev).to(torch.float8_e4m3fn)
    b8 = torch.randn(S, S, device=dev).to(torch.float8_e4m3fn)
    one = torch.tensor(1.0, device=dev)
    t = cuda_ms(lambda a8=a8, b8=b8: torch._scaled_mm(a8, b8.t(), scale_a=one, scale_b=one, out_dtype=torch.bfloat16),
                iters=max(3, ITERS // 2))
    res["fp8"] = flops / (t / 1e3) / 1e12
    del a8, b8
except Exception as e:  # noqa: BLE001 - FP8 support depends on GPU/torch build; report and continue
    print(f"fp8 _scaled_mm unavailable: {type(e).__name__}: {str(e)[:120]}")
free_gpu()
table([{"dtype": k, "TFLOPS": v} for k, v in res.items()])
ridge = res["bf16"] * 1e12 / (bw * GB)
print(f"\nridge point (bf16) = {res['bf16']:.0f} TFLOPS / {bw:.0f} GB/s = {ridge:.0f} FLOP/byte")
print(f"A kernel needs > ~{ridge:.0f} FLOPs per byte it moves before the tensor cores, not DRAM, are the limit.")
kv("matmul", **{f"tflops_{k}": v for k, v in res.items()}, ridge_bf16=ridge)

# ---------------------------------------------------------------------------
section("3. Decode-shaped matmul: Y[M,N] = X[M,K] @ W[K,N], M = batch")
# ---------------------------------------------------------------------------
# One 8192x8192 bf16 weight (128 MB, > L2). M is how many sequences decode together.
K = N = 8192
W = torch.randn(K, N, device=dev, dtype=torch.bfloat16)
rows = []
for M in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]:
    X = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    t = cuda_ms(lambda X=X, W=W: X @ W, iters=ITERS)
    moved = 2 * (M * K + K * N + M * N)
    ai = 2 * M * N * K / moved
    rows.append({"M": M, "ms": t, "AI": ai, "GB/s": moved / (t / 1e3) / GB, "TFLOPS": 2 * M * N * K / (t / 1e3) / 1e12,
                 "us/token": t * 1e3 / M})
    del X
table(rows)
t1 = rows[0]["ms"]
knee = next((r["M"] for r in rows if r["ms"] > 1.5 * t1), rows[-1]["M"])
by_m = {r["M"]: r for r in rows}
print(f"\nTime stays ~flat until M ~ {knee} (first M with > 1.5x the M=1 time); roofline predicts the ridge at AI ~ M ~ {ridge:.0f}.")
print("Below the knee extra sequences ride along on the same weight read: that is why continuous batching works.")
kv("batch-sweep", ms_m1=t1, ms_m64=by_m[64]["ms"], ms_m4096=by_m[4096]["ms"], knee_m=knee, ridge=ridge,
   gbps_m1=by_m[1]["GB/s"], tflops_m4096=by_m[4096]["TFLOPS"], us_per_tok_m1=by_m[1]["us/token"],
   us_per_tok_m256=by_m[256]["us/token"])
del W
free_gpu()

# ---------------------------------------------------------------------------
section("4. Elementwise ops are memory-bound -> fusion")
# ---------------------------------------------------------------------------
# relu(x * w + b): 3 eager kernels, each reads and writes the whole tensor.
# torch.compile (Inductor) generates ONE Triton kernel: one read of x, one write.
x = torch.randn(2**28, device=dev, dtype=torch.bfloat16)  # 512 MiB
w = torch.randn_like(x)
b = torch.randn_like(x)


def eager(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.relu(x * w + b)


# ms-scale kernels: more iterations are cheap, and one noisy median here once halved the speedup.
t_eager = cuda_ms(lambda x=x, w=w, b=b: eager(x, w, b), warmup=5, iters=max(ITERS, 15))
fused = torch.compile(eager)
t_fused = cuda_ms(lambda x=x, w=w, b=b: fused(x, w, b), warmup=5, iters=max(ITERS, 15))
ideal = 4 * x.numel() * 2 / (bw * GB) * 1e3  # read x, w, b + write out, once
print(f"eager  : {t_eager:.2f} ms  (x*w -> tmp, tmp+b -> tmp2, relu -> out: 3 kernels, 7 tensor passes)")
print(f"fused  : {t_fused:.2f} ms  (1 kernel, 4 tensor passes)")
print(f"ideal at {bw:.0f} GB/s: {ideal:.2f} ms. FLOPs are irrelevant here: ~1 FLOP per 2 bytes.")
kv("fusion", eager_ms=t_eager, fused_ms=t_fused, speedup=t_eager / t_fused, fused_vs_ideal=ideal / t_fused)
del x, w, b
free_gpu()

OUT.mkdir(parents=True, exist_ok=True)
GPU_JSON.write_text(json.dumps({"name": props.name, "bw_gbps": bw, "bw_copy_gbps": bw_copy, "bw_read_gbps": bw_read,
                                **{f"tflops_{k}": v for k, v in res.items()}, "ridge_bf16": ridge,
                                "decode_knee_m": knee}, indent=2))
print(f"\nsaved {GPU_JSON}")
