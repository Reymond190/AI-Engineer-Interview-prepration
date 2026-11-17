"""
02_serving_benchmark.py
-----------------------
A small load generator for the live vLLM server: TTFT, TPOT, throughput and
goodput under rising concurrency, compared with the roofline predictions of
lesson 01. This harness is reused by every serving lesson that follows
(scheduler knobs, quantization, speculative decoding).

Run: vLLM UP (`bash server/serve.sh`), any env (stdlib HTTP only). For the
roofline comparison run it in WSL, where the HF cache and outputs/gpu.json are.
    python inference_course/02_serving_benchmark.py [--quick]
If no server answers, it prints [skip] and run_all.py marks its checks skipped.

Interview topics
- The serving metrics: TTFT (queue + prefill), TPOT/ITL (decode pace), e2e latency =
  TTFT + (n-1) x TPOT, throughput (output tok/s across all requests), goodput
  (requests/s that meet an SLO, e.g. TTFT < 1 s and TPOT < 50 ms).
- Measuring honestly: stream and timestamp tokens client-side, take n from the
  server's usage block, use ignore_eos to fix the output length, randomize the
  prompt prefix so prefix caching doesn't fake a fast prefill, warm up first.
- Continuous batching: new requests join the running batch every step, so decode
  throughput scales ~linearly with concurrency while TPOT barely moves (lesson 00:
  the weight read is shared). Past --max-num-seqs, requests queue and TTFT explodes.
- TTFT grows with prompt length (prefill is compute-bound); TPOT doesn't.
- Roofline as a sanity check: measured batch-1 tok/s must sit below bandwidth /
  bytes-per-step; the ratio is the serving stack's efficiency.
- Percentiles over means: p95/p99 TTFT is what users feel and what SLOs are written on.
"""

from __future__ import annotations

import statistics
import time
from concurrent.futures import ThreadPoolExecutor

from common import (
    SERVED,
    kv,
    load_gpu,
    load_model_info,
    parse_args,
    pct,
    prefill_ms,
    roofline_step,
    section,
    served_model,
    stream_completion,
    table,
    unique_prompt,
)

args = parse_args(__doc__)
model = served_model()
if model is None:
    print("no vLLM server reachable at AGENT_BASE_URL; start `bash server/serve.sh` and rerun")
    kv("skip", reason="no_server")
    raise SystemExit(0)
print(f"server model: {model}")
OUT_TOK = 128 if args.quick else 256
REPS = 2 if args.quick else 4

# warm up: CUDA graphs, sampler, first-request allocations
for _ in range(2):
    stream_completion(model, unique_prompt(32), 16)

# ---------------------------------------------------------------------------
section("1. Single stream: TPOT vs the roofline ceiling")
# ---------------------------------------------------------------------------
runs = [stream_completion(model, unique_prompt(64), OUT_TOK) for _ in range(REPS + 1)]
tpot = statistics.median(r["tpot"] for r in runs)
ttft = statistics.median(r["ttft"] for r in runs)
tok_s = 1 / tpot
print(f"prompt ~{runs[0]['prompt_tokens']} tok, {OUT_TOK} out: TTFT {ttft * 1e3:.0f} ms, TPOT {tpot * 1e3:.2f} ms "
      f"-> {tok_s:.1f} tok/s")
profile = SERVED.get(model)
info = load_model_info(profile) if profile else None
if info:
    gpu = load_gpu()
    ceil = roofline_step(info, gpu, 1, runs[0]["prompt_tokens"] + OUT_TOK // 2)
    eff = tok_s / ceil["tok_s"]
    print(f"roofline ceiling ({info['decode_bytes'] / 1e9:.2f} GB/step at {gpu['bw_gbps']:.0f} GB/s): "
          f"{ceil['tok_s']:.1f} tok/s -> serving efficiency {eff:.0%}")
    print("The gap: GEMV kernels reach ~80% of copy bandwidth (lesson 00, M=1), plus per-step CPU/scheduler")
    print("overhead, sampling, and small ops (norms, rotary, KV/state writes) that each touch memory.")
    kv("single", tok_s=tok_s, tpot_ms=tpot * 1e3, ttft_ms=ttft * 1e3, ceiling_tok_s=ceil["tok_s"], efficiency=eff)
else:
    print("(no roofline: model not in common.MODELS or HF cache not visible from this env)")
    kv("single", tok_s=tok_s, tpot_ms=tpot * 1e3, ttft_ms=ttft * 1e3)

# ---------------------------------------------------------------------------
section("2. TTFT vs prompt length: prefill is compute-bound")
# ---------------------------------------------------------------------------
rows = []
res = {}
for words in (128, 1024, 4096):
    rs = [stream_completion(model, unique_prompt(words), 8) for _ in range(REPS)]
    t = statistics.median(r["ttft"] for r in rs) * 1e3
    p = rs[0]["prompt_tokens"]
    row = {"prompt_tok": p, "ttft_ms": t, "us_per_prompt_tok": t * 1e3 / p}
    if info:
        row["roofline_ms"] = prefill_ms(info, gpu, p)
        row["mfu"] = row["roofline_ms"] / t
    rows.append(row)
    res[words] = row
table(rows)
print(f"\nPer-token prefill cost is ~constant ({res[4096]['us_per_prompt_tok']:.0f} us vs {tpot * 1e6:.0f} us per decode "
      f"token, {tpot * 1e6 / res[4096]['us_per_prompt_tok']:.0f}x): TTFT ~ linear in prompt.")
print("Short prompts are dominated by fixed overhead (HTTP, scheduling, one decode step), so MFU is low there.")
kv("prefill", ttft_128=res[128]["ttft_ms"], ttft_1k=res[1024]["ttft_ms"], ttft_4k=res[4096]["ttft_ms"],
   mfu_4k=res[4096].get("mfu", 0.0))

# ---------------------------------------------------------------------------
section("3. Prefix caching teaser: the same long prompt twice")
# ---------------------------------------------------------------------------
p = unique_prompt(4096)
cold = stream_completion(model, p, 8)["ttft"] * 1e3
warm = statistics.median(stream_completion(model, p, 8)["ttft"] for _ in range(REPS)) * 1e3
print(f"cold {cold:.0f} ms -> repeated {warm:.0f} ms ({cold / warm:.1f}x). Lesson 05 covers when this works "
      "(block hashing, hybrid-model caveats) and when it doesn't.")
kv("prefix-cache", cold_ms=cold, warm_ms=warm, speedup=cold / warm)

# ---------------------------------------------------------------------------
section("4. Concurrency sweep: continuous batching and queueing")
# ---------------------------------------------------------------------------
SLO_TTFT, SLO_TPOT = 1.0, 0.050


def wave(conc: int) -> dict:
    n_req = conc * REPS
    t0 = time.perf_counter()
    with ThreadPoolExecutor(conc) as ex:
        rs = list(ex.map(lambda _: stream_completion(model, unique_prompt(256), OUT_TOK), range(n_req)))
    wall = time.perf_counter() - t0
    ok = [r for r in rs if r["ttft"] < SLO_TTFT and r["tpot"] < SLO_TPOT]
    return {"conc": conc, "out_tok_s": sum(r["out_tokens"] for r in rs) / wall,
            "tpot_ms_p50": pct([r["tpot"] for r in rs], 0.5) * 1e3,
            "ttft_ms_p50": pct([r["ttft"] for r in rs], 0.5) * 1e3, "ttft_ms_p95": pct([r["ttft"] for r in rs], 0.95) * 1e3,
            "goodput_rps": len(ok) / wall, "slo_ok": f"{len(ok)}/{n_req}"}


rows = [wave(c) for c in (1, 2, 4, 8)]
table(rows)
by = {r["conc"]: r for r in rows}
print(f"\nSLO: TTFT < {SLO_TTFT * 1e3:.0f} ms and TPOT < {SLO_TPOT * 1e3:.0f} ms. Up to --max-num-seqs (4 in serve.sh)")
print("throughput grows ~linearly with almost the same TPOT. At 8, half the requests wait in the queue for a slot:")
print("TTFT jumps by a whole request's decode time while throughput stays flat.")
kv("concurrency", tok_s_c1=by[1]["out_tok_s"], tok_s_c4=by[4]["out_tok_s"], tok_s_c8=by[8]["out_tok_s"],
   tpot_c1=by[1]["tpot_ms_p50"], tpot_c4=by[4]["tpot_ms_p50"], ttft_p50_c4=by[4]["ttft_ms_p50"],
   ttft_p50_c8=by[8]["ttft_ms_p50"], ttft_p95_c8=by[8]["ttft_ms_p95"],
   goodput_c4=by[4]["goodput_rps"], goodput_c8=by[8]["goodput_rps"])
