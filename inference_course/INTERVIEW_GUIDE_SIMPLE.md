# LLM inference, explained simply

The plain-language version of [INTERVIEW_GUIDE.md](INTERVIEW_GUIDE.md). Same
ideas, fewer formulas. Read this first, then the full guide for the exact
numbers and terms an interviewer expects.

All numbers come from **your own GPU** (RTX 5080) running Qwen3-8B.

---

## The one picture to remember

A GPU is a **super-fast kitchen with a slow delivery truck**.

- The **kitchen** (the compute cores) can cook unbelievably fast: 112 trillion operations per second.
- The **truck** (memory bandwidth) brings ingredients (the model's weights) from the storeroom (VRAM) at "only" 889 GB per second.

Most of the time, the kitchen sits idle **waiting for the truck**. Almost
everything in inference optimization is a way to either:
1. make the truck carry less (quantization, smaller KV cache), or
2. cook more dishes per truck trip (batching).

---

## 1. Is the GPU waiting on memory or on math? (lesson 00)

**Memory-bound vs compute-bound**
- *Memory-bound:* the chef waits for ingredients. Speed depends on the truck.
- *Compute-bound:* ingredients are piled up and the chef is the bottleneck. Speed depends on the kitchen.

**Arithmetic intensity** = how much cooking you do per kilogram of
ingredients delivered. Low intensity → you're waiting on the truck. On your
GPU the break-even point is about **126 operations per byte**: below that you
wait on memory, above it on compute.

**Why generating text is slow**
To produce **one** word, the model must read **all** of its weights (about
5 GB) from memory. It does very little math with them, so the GPU spends most
of its time waiting on the truck.

**Why batching is almost free**
If 64 users each want their next word, the truck still delivers the weights
only once, and the chef cooks 64 dishes from that one delivery.
**Measured:** 1 user took 0.18 ms, and 64 users took 0.20 ms. That's nearly the same time
for 64× the work.

**Why "fusing" operations helps**
Doing three small steps separately is like making three trips to the
storeroom. A fused kernel does all three in one trip.
**Measured:** 2× faster (5.4 ms → 2.7 ms).

**FP8 (8-bit numbers)**
Smaller numbers mean faster math: **384 vs 112** trillion ops/s on your GPU.
Smaller numbers also mean less for the truck to carry.

**Timing GPU code**
The GPU works in the background, so a normal stopwatch just measures "I asked
the GPU to start." Use GPU timers, run a few warm-up rounds first, and take
the middle result.

---

## 2. Speed and memory math (lesson 01)

**Max speed for one user = memory speed ÷ model size read per word**
Qwen3-8B reads 4.85 GB per word → 889 / 4.85 ≈ **180 words per second
max**. You can't beat this without reading fewer bytes.

**The trap in quantized ("4-bit") models**
People assume "4-bit = everything is small." But parts of the model often
stay big. In Qwen3.5-9B, one leftover full-size part (the `lm_head`, the
final layer that picks the next word) is **25%** of everything read for each
word. Shrinking it too would give about **23% more speed**.

**What is the KV cache?**
The model's **short-term memory** of the conversation. For every word seen so
far it stores a note so it doesn't have to re-read the whole conversation.
- Qwen3-8B: **144 KB per word** → a 32,000-word conversation needs **4.5 GB**.
- Tricks that shrink it:
  - **GQA**: attention heads share notes, 4× smaller here.
  - **Hybrid models** like Qwen3.5: most layers keep a fixed-size summary instead of per-word notes. 32 KB per word, 4.5× smaller.

**How many users fit at once?**
GPU memory = model weights + short-term memory for every user + some
workspace. **Measured:** after loading Qwen3-8B, vLLM had room for **54,272
words** of memory, which is fewer than two users with full 32k conversations.

**When batching stops helping**
With long conversations, each user's short-term memory gets bigger than the
model itself. Then the truck mostly carries memories, not weights, so adding
users barely helps. At 32k words with 4 users, the gain drops from ~4× to ~1.6×.

**Reading the prompt vs writing the answer**
- **Reading your prompt (prefill)** happens all at once and in parallel, so it's math-heavy and fast per word: **0.19 ms per word**.
- **Writing the answer (decode)** goes one word at a time, waits on memory, and is slow per word: **8.4 ms per word**, **44× slower**.

This is why systems split them up ("chunked prefill",
"prefill/decode disaggregation").

---

## 3. Measuring a real server (lesson 02)

**The numbers that matter**

| Name | Plain meaning |
|---|---|
| **TTFT** (time to first token) | How long until the first word appears |
| **TPOT** (time per output token) | How fast the words then stream |
| **Throughput** | Total words per second across all users (= cost efficiency) |
| **Goodput** | How many requests per second are *actually fast enough* for users |

Look at the slow cases (95th percentile), not the average. Users remember the
slow ones.

**Benchmarking without fooling yourself**
- Make every prompt slightly different. The server remembers prompts it has
  already seen (prefix caching), and a repeated prompt looks **37× faster**
  than it really is (803 ms → 22 ms).
- Force a fixed answer length, and warm up first.

**How close to the theoretical maximum do we get?**
**Measured:** 120 words/s out of a possible 180 → **66%**. The rest is lost to
imperfect memory reads, bookkeeping between steps, and many small extra
operations.

**Continuous batching, and when it breaks**
The server lets new users jump into the running batch at every step, like a
bus that picks up passengers at every stop.

| Users | Words/sec total | Speed per user | Wait for first word |
|---|---|---|---|
| 1 | 105 | normal | 74 ms |
| 4 | **419** (4×!) | still normal | 214 ms |
| 8 | 415 (no gain) | still normal | **2.6 s**, half are waiting in line |

The server only has 4 seats (`max-num-seqs 4`). With 8 users, half wait for a
seat. Total output stays the same, but the number of users who get a fast
experience **falls from 1.64 to 0.20 per second**. More throughput doesn't
help if users are left waiting.

---

## Cheat sheet: one line each

- Generating text is **memory-bound**: the GPU waits for weights.
- **Batching** is nearly free because one weight read serves many users.
- **Quantization** speeds things up because there are fewer bytes to read.
- **KV cache** is the model's short-term memory, and it limits how many users fit.
- **Prefill** (reading the prompt) is fast and math-bound. **Decode** (writing) is slow and memory-bound.
- Measure **TTFT, TPOT, throughput, goodput**, and use percentiles.
- Past the server's capacity, users **queue**: throughput stays flat while latency explodes.
