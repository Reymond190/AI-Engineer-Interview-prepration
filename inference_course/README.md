# LLM inference optimization course (runnable)

The specialization track: make LLMs faster and cheaper to serve, from vLLM
scheduler knobs down to GPU kernels. Every lesson **measures** on this machine
(RTX 5080 16 GB, vLLM in WSL) and prints `[tag] key=value` lines that
`run_all.py` checks for the effect being taught. Same style as `ml_course/`:
each script opens with an **"Interview topics"** docstring, and
`INTERVIEW_GUIDE.md` holds the short answers with the numbers measured here.

## Setup

WSL, conda env `ft` (torch + Triton). Lessons need different machine states:

| Lessons | State | Why |
|---|---|---|
| GPU (00, later 12-17) | **vLLM stopped** | vLLM reserves ~85% of VRAM; a full GPU spills to RAM on WSL and timings become wrong |
| Math (01) | anything | reads safetensors headers from `~/.cache/huggingface` + `outputs/gpu.json` |
| Serving (02, later 03-11) | **vLLM up** (`bash server/serve.sh [profile]`) | stdlib HTTP client; prints `[skip]` if no server |

```bash
# WSL, project root, conda activate ft
python inference_course/00_gpu_roofline.py          # first: writes outputs/gpu.json
python inference_course/01_decode_math.py
bash server/serve.sh qwen3 &                         # then, with the server up:
python inference_course/02_serving_benchmark.py
python inference_course/run_all.py 00 01             # outcome checks (vLLM stopped)
python inference_course/run_all.py 02                # (vLLM up)
```

**WSL RAM note:** WSL currently gets 7.2 GiB of RAM plus 2 GiB of swap, and vLLM
can't mmap the 11 GB `qwen35` checkpoint ("Cannot allocate memory"), so the
serving numbers so far use the `qwen3` profile. To use `qwen35`, raise
`memory=` (or `swap=`) in `%UserProfile%\.wslconfig`.

## Roadmap

| Phase | # | Script | Core idea | Status |
|---|---|---|---|---|
| 0 Fundamentals | 00 | `00_gpu_roofline.py` | Measured bandwidth + TFLOPs per dtype, ridge point, decode-shaped matmul from memory- to compute-bound, fusion | done |
| | 01 | `01_decode_math.py` | Exact bytes per decode step from safetensors headers, decode ceiling, KV per token (GQA, hybrid), batch vs context, KV budget, prefill vs decode | done |
| | 02 | `02_serving_benchmark.py` | TTFT/TPOT/throughput/goodput harness, roofline efficiency, prefill scaling, prefix-cache teaser, continuous batching + queueing | done |
| 1 Serving | 03-06 | | Continuous batching internals, chunked prefill, prefix caching (and hybrid caveats), CUDA graphs, `max-num-seqs` / memory trade-offs, vLLM vs SGLang | next |
| 2 Quantization | 07-09 | | w4a16 vs FP8 vs NVFP4 vs FP8 KV cache; speed vs quality on the lab's evals; lm_head quantization | |
| 3 Speculative decoding | 10-11 | | n-gram, draft model, EAGLE-3, Qwen3.5 MTP heads; acceptance rate vs speedup | |
| 4 Kernels | 12-15 | | Triton: fused RMSNorm/softmax, tiled matmul, FlashAttention forward; profiling; torch.compile | |
| 5 Capstone | 16-17 | | Paged attention + continuous-batching mini-engine; disaggregated prefill/decode, tensor parallelism | |

Logs: `inference_course/outputs/logs/` (gitignored).
