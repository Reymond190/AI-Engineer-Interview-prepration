# AI/ML Engineer Interview Cheat Sheet

*Joseph Reymond B — tailored to LLM / RAG / Agentic AI / Post-Training / ML Platform roles*

---

## 0. Your 60-Second Pitch

> "I'm a Technical Lead at Tata Communications with 7+ years in Python backend and applied AI. I built an enterprise RAG assistant used by 20,000+ employees, a real-time voice agent designed to cut support workload by ~80%, and a VLM-based invoice pipeline with human-in-the-loop approvals. I also own LLM serving on Triton with quantization, and run a hands-on local lab benchmarking LoRA/QLoRA/DPO/GRPO fine-tuning and multi-agent LangGraph architectures. My strength is taking LLM ideas to reliable, scalable production systems."

**Keep ready:** 1 win story, 1 failure story, 1 conflict story, 1 "hardest technical problem" story (see §10).

---

## 1. ML Fundamentals (Must-Know)

| Concept | One-liner |
|---|---|
| Bias vs Variance | High bias = underfit (too simple); high variance = overfit (too sensitive to training data) |
| Overfitting fixes | More data, regularization (L1/L2), dropout, early stopping, simpler model, augmentation |
| L1 vs L2 | L1 → sparse weights (feature selection); L2 → small, spread-out weights |
| Train/Val/Test | Train fits, val tunes hyperparams, test = final unbiased estimate (touch once) |
| Cross-validation | k-fold: rotate validation fold; use stratified for imbalanced classes |
| Gradient descent | Batch (all data), SGD (1 sample), Mini-batch (standard) |
| Optimizers | SGD+momentum, Adam (adaptive LR + momentum), AdamW (decoupled weight decay — default for transformers) |
| Learning rate | Too high → diverge; too low → slow. Use warmup + cosine decay for transformers |
| Vanishing gradients | Fix with ReLU/GELU/SwiGLU, residual connections, normalization, good init |
| BatchNorm vs LayerNorm | BN normalizes across batch (CNNs); LN/RMSNorm across features per token (Transformers) |
| Data leakage | Info from test/future leaking into training → inflated metrics. Split before preprocessing |

### Metrics

| Task | Metrics | When |
|---|---|---|
| Classification | Precision = TP/(TP+FP), Recall = TP/(TP+FN), F1 = harmonic mean | Imbalanced → F1, PR-AUC over accuracy |
| Ranking/Retrieval | Recall@k, MRR, nDCG, Hit Rate | RAG retrieval evaluation |
| Regression | MAE (robust), RMSE (penalizes large errors), R² | |
| Generation | BLEU/ROUGE (weak), BERTScore, LLM-as-judge, human eval | |
| LLM | Perplexity (lower = better), faithfulness, answer relevancy, pass@k | |

**Precision vs Recall trade-off:** Fraud/medical → favor recall. Spam filter → favor precision.

### Classic Algorithms (quick recall)

- **Linear/Logistic Regression** — baseline, interpretable; logistic = sigmoid + cross-entropy
- **Decision Tree** — splits on Gini/entropy; overfits alone
- **Random Forest** — bagging + feature randomness → lowers variance
- **Gradient Boosting (XGBoost/LightGBM)** — sequential trees fix residuals → lowers bias; king of tabular data
- **SVM** — max-margin; kernel trick for non-linear
- **k-Means** — choose k via elbow/silhouette; sensitive to init and scale
- **PCA** — project onto directions of max variance; for dimensionality reduction

---

## 2. Deep Learning & Transformers

### Transformer Core

- **Self-attention:** `Attention(Q,K,V) = softmax(QKᵀ / √d_k) · V` (`√d_k` keeps softmax from saturating → healthy gradients)
- **Multi-head:** several attention heads learn different relationships, then concat + project
- **Positional encoding:** sinusoidal, **RoPE** (rotary relative phase; extended to long context via **NTK-aware scaling / YaRN**), ALiBi
- **Modern block:** Pre-LN with **RMSNorm** (drops mean-centering, faster) + Attention + **SwiGLU** FFN + residual stream
- **Complexity:** \(O(n^2)\) compute in sequence length → why long prefill is expensive; effective retrieval degrades well before advertised context limit ("lost in the middle" / Needle-in-a-Haystack)

### Architecture Families

| Type | Examples | Use |
|---|---|---|
| Encoder-only | BERT, RoBERTa, BGE, DeBERTa | Classification, embeddings, cross-encoder reranking, NER |
| Decoder-only | GPT, Llama, Qwen, Mistral, DeepSeek | Text generation, chat, reasoning, agents |
| Encoder-decoder | T5, BART, Whisper | Translation, summarization, ASR |

### Modern Architecture & Reasoning Terms

- **Tokenization:** BPE / SentencePiece; ~0.75 words/token in English; larger vocabs (128K–152K) compress text better but inflate the output logits tensor
- **KV Cache & Attention variants:**
  - **MQA / GQA:** group query heads to share K/V heads → shrinks KV cache 4–8×
  - **MLA (Multi-head Latent Attention):** compresses KV into a low-rank latent vector (DeepSeek-V2/V3) → huge KV savings
  - **Hybrid / Linear attention:** interleaves linear/state-space layers (e.g., Gated DeltaNet / Mamba) with full attention layers (Qwen3.5, Jamba)
  - **FlashAttention-2/3:** IO-aware exact tiling in SRAM; avoids writing \(n \times n\) attention matrix to HBM → \(O(n)\) memory, 2–4× faster
- **MoE (Mixture of Experts):** router activates top-\(k\) FFNs per token → large capacity at small active FLOPs (watch load-balancing loss & VRAM capacity)
- **Reasoning Models & Test-Time Compute:** emit `<think>` chain-of-thought tokens trained via RLVR; avoid `temperature=0` on thinking models (causes repetition loops); enforce thinking token budgets and reserve thinking mode for planners/critics/judges, not simple routers
- **Decoding:** greedy, beam search, temperature, top-k, top-p (nucleus), min-p

---

## 3. LLM Post-Training & Fine-Tuning

### Post-Training Stages

| Stage | Objective / Mechanism | What It Teaches & Caveats |
|---|---|---|
| **Pre-training** | Next-token cross-entropy on trillions of tokens | World knowledge & raw capabilities (expensive) |
| **SFT** | Completion-only CE on curated `(prompt, response)` pairs | Strict output formats, tone, house rules, tool-calling syntax |
| **RLHF (PPO)** | Reward model (Bradley-Terry) + PPO with \(\beta \cdot \text{KL}(\pi \Vert \pi_{\text{ref}})\) | Aligns preferences; needs **4 models in VRAM** (policy, ref, reward, critic) — heavy & unstable |
| **DPO** | Closed-form preference loss on `(prompt, chosen, rejected)` | Drops reward & critic models (only policy + frozen ref); do SFT first; watch chosen log-prob drop |
| **RLVR / GRPO** | Sample \(G\) rollouts/prompt, score via verifiable code/math/schema graders, advantage = \((r - \mu_G)/\sigma_G\) | Drops the critic network; turns `pass@k` into `pass@1` (recipe behind DeepSeek-R1 / reasoning models) |

### Alignment Variants & Pitfalls (2025–2026 Staples)

- **DPO variants:** **ORPO** (SFT + odds-ratio penalty in 1 stage, no ref model), **SimPO** (length-normalized, reference-free), **IPO** (regularized against deterministic preference overfitting), **KTO** (unpaired thumbs-up/down data). With PEFT, get \(\pi_{\text{ref}}\) for free by disabling the LoRA adapter (merge SFT adapter first!). Mine **on-policy negatives** on held-out phrasings.
- **GRPO pitfalls & fixes:**
  - **Reward hacking:** model games naive rewards (e.g., printing numbers 0–199 to hit "answer appears in text") → make graders strict, penalize length/gibberish.
  - **Zero-variance groups:** if all \(G\) samples score 0 or all score 1, advantage = 0 (no gradient) → tasks must be "hard but solvable" (**DAPO** dynamically filters zero-variance prompts).
  - **Length & difficulty bias:** standard GRPO divides by sequence length (penalizes long wrong answers less per token) and group std → **Dr. GRPO** removes both; **DAPO** uses token-level loss + `clip-higher` to prevent entropy collapse; **GSPO** uses sequence-level importance ratios (better for MoE).

### When to Use What

| Need | Choose |
|---|---|
| Quick steering / prototyping | **Prompt engineering + few-shot + schema decoding** |
| Fresh / private / changing facts + citations | **RAG** (FT is a poor way to inject facts — goes stale & hallucinates) |
| Strict format, tacit house rules, tool schemas, tone | **Fine-tuning (LoRA / QLoRA + DPO/GRPO)** |
| Serve 9B/70B quality at 0.6B–4B latency & cost | **Distillation** + task-specific LoRA |

> Rule of thumb: **Prompt → RAG → Fine-tune / Distill** (combine RAG + FT when the model needs better context-use behavior).

### Training Memory Math & Optimization Levers

- **Full FT memory (~16 B/param in mixed-precision AdamW):** `fp32` master weights (4B) + grads (4B) + Adam \(m, v\) (4B + 4B) + activations + **logits tensor** (\(\text{batch} \times \text{seq} \times \text{vocab} \times 4\text{B}\) — e.g., 152K vocab = multi-GB spike!).
  - *Rough 9B VRAM:* Full FT ≈ 144 GB (before activations) | bf16 LoRA ≈ 18 GB | 4-bit QLoRA ≈ 5–6 GB.
  - *Gotcha:* Pure `bf16` weights without `fp32` master weights ruin full FT at `lr=1e-5` because tiny updates round to zero (8 mantissa bits).
- **VRAM optimization order on constrained GPUs:**
  1. **LoRA / QLoRA** (eliminates optimizer states & gradients on frozen base weights; activations still remain).
  2. **Fused Linear Cross-Entropy** (**Liger Kernels** / Cut-CE / TRL fused head) — computes CE in chunks without materializing full `[B, S, V]` logits.
  3. **Padding-free packing (BFD)** — concatenates short examples, resets `position_ids`, and uses FlashAttention varlen / block-diagonal masks so examples never cross-attend.
  4. **Gradient checkpointing** (`use_cache=False`) — recomputes activations in backward pass (~20–35% slower, big long-context VRAM win).
  5. **8-bit / Paged AdamW** (bitsandbytes) or low-rank full-param optimizers (**GaLore**, **APOLLO**, **Spectrum**).
- **Gradient accumulation:** effective batch = `micro_batch × grad_accum × num_gpus`; normalize loss by *total trained tokens* across the window, not per micro-batch.

### LoRA & QLoRA Internals

- **Math & Init:** \(W' = W + \frac{\alpha}{r}BA\), where \(A \in \mathbb{R}^{r \times d_{\text{in}}} \sim \mathcal{N}(0, \sigma^2)\) and \(B \in \mathbb{R}^{d_{\text{out}} \times r} = 0\), so \(\Delta W = 0\) at step 0 (starts identical to base model).
- **Target modules:** Target **all linear layers** (`q, k, v, o, gate, up, down`), not just `q_proj, v_proj` — matters far more than raising rank \(r\) (typical \(r = 8\text{–}32\), \(\alpha = r \text{ or } 2r\), LoRA LR \(\approx 1\text{e-}4 \text{ to } 3\text{e-}4\) vs Full FT \(1\text{e-}5\)).
- **QLoRA:** Freezes base in **4-bit NF4** (quantiles of \(\mathcal{N}(0,1)\), lower error than uniform INT4/FP4 on bell-curved weights) + **double quantization** (quantizes quantization scales) + **paged optimizers**; trains bf16 LoRA on top (~20–40% slower due to on-the-fly dequantization).
- **Merging rule:** Merge \(W \leftarrow W + \frac{\alpha}{r}BA\) into **bf16/fp32 base weights** for zero inference latency — *never* merge directly into 4-bit weights (re-quantize with AWQ/GPTQ after bf16 merge).
- **Key PEFT variants:** **DoRA** (decomposes weight into magnitude + LoRA direction, +1–2 pts, slower train), **rsLoRA** (\(\alpha/\sqrt{r}\) scaling keeps large ranks stable), **LoRA+** (higher LR on \(B\) than \(A\)), **PiSSA / OLoRA** (SVD/QR init).

### Data Hygiene, Forgetting, Distillation & Multi-GPU

- **Silent data bugs:**
  - **Chat-template parity:** Train text must match serve text byte-for-byte (including special tokens & Qwen3's empty `<think>\n\n</think>` block).
  - **Completion-only masking (**`labels = -100` **on prompt):** Loss only on assistant reply, **including** `<eos>` **/** `<|im_end|>` (or the model never stops); tokenize prompt & completion carefully so BPE doesn't merge across the boundary.
  - **Padding side:** **Right-pad** for training; **left-pad** for batched generation.
  - **Splits & decontamination:** Hold out *phrasings/templates* (not just random rows), include **negative examples** (e.g., 25% "no tool needed" so tool-tuned models don't always call tools), and decontaminate against eval sets via n-gram / MinHash overlap.
- **Catastrophic forgetting:** Narrow format SFT can collapse general reasoning even when generic perplexity barely moves! Mitigate via **behavioral regression suites**, 5–20% general **replay data**, KL penalty, adapter merging (**TIES / DARE**), or **multi-LoRA serving** (route only target traffic to the adapter).
- **Knowledge Distillation (KD):**
  - **Sequence-level KD:** SFT student on teacher completions (works across tokenizers & black-box APIs; watch teacher error propagation).
  - **Logit KD:** \(\alpha T^2 \text{KL}(p_t^T \Vert p_s^T) + (1-\alpha)\text{CE}\) — transfers soft "dark knowledge" (requires shared vocabulary).
  - **On-policy / GKD:** Student generates rollouts, teacher scores student's own prefixes (fixes exposure bias).
- **Multi-GPU Scaling:** **DDP** (full model copy per GPU, all-reduce grads) → **DeepSpeed ZeRO-1/2/3 & PyTorch FSDP** (shard optimizer states [1] → +grads [2] → +params [3], cutting VRAM by ~\(1/N\)) → **Tensor Parallelism** (intra-node NVLink) → **Pipeline / Context Parallelism** (inter-node / ring attention for 100K+ seqs).
- **Lab proof points to quote (Qwen3-0.6B/4B on 16 GB RTX 5080):** Ticket JSON LoRA `0.15 → 0.68` (+DPO `→ 0.78`) on unseen phrasings; tool-call LoRA `0.28 → 1.00` in 40 steps; Liger fused CE cut peak VRAM `18.2 → 8.6 GB` (`3.8×` tok/s); narrow SFT dropped math `0.68 → 0.00` while perplexity stayed flat (`9.2 → 9.0`), proving why behavioral regression tests are mandatory.

---

## 4. RAG — Your Strongest Area

### Pipeline

```
Ingest → Parse → Chunk → Embed → Index (Vector DB + BM25)
Query → (Route / Rewrite) → Hybrid Retrieve (ACL-filtered) → Cross-Encoder Rerank → Prompt → LLM → Cite & Verify
```

### Design Decisions & Talking Points

| Component | Options / Trade-offs |
|---|---|
| **Chunking** | Fixed-size + overlap (simple); recursive/markdown-structure; semantic; **parent-child** (retrieve small leaf, feed larger parent); **contextual retrieval** (prepend doc summary to chunk before embedding). Typical: 300–800 tokens, 10–20% overlap |
| **Embeddings** | BGE-M3, Qwen3-Embedding, E5, OpenAI `text-embedding-3`; **Matryoshka** allows dimension truncation; normalize for cosine |
| **Vector DB & Index** | pgvector, Qdrant, Milvus, Weaviate, Elasticsearch. **HNSW** (fast graph, RAM-heavy), **IVF-PQ** (clustered + compressed, slight recall drop) |
| **Retrieval** | Dense + **BM25 / SPLADE lexical hybrid** (catches exact SKUs, error codes, names) fused via **Reciprocal Rank Fusion (RRF)** |
| **Reranking** | Cross-encoder (`bge-reranker`, Cohere Rerank) or late-interaction (**ColBERT**) on top-50 → top-5 → biggest precision jump |
| **Query side** | Query decomposition, HyDE, metadata extraction, **Agentic / Corrective RAG** (grade chunks → rewrite or fall back to web/abstain) |

### Common RAG Failures → Fixes

| Problem | Fix |
|---|---|
| Right doc not retrieved | Hybrid search (BM25 + dense), contextual chunk headers, query rewriting, metadata filters |
| Retrieved but wrong answer | Cross-encoder reranking, fewer/cleaner chunks, put best chunks first/last (fix "lost in the middle") |
| Hallucination | Enforce inline citations, groundedness verifier, explicit "abstain if not in context" path |
| Stale data / Access control | Content-hash incremental indexing; **pre-filter by user ACLs/RBAC inside Vector DB query** (critical at 20K users!) |

### RAG & LLM-as-Judge Evaluation (DeepEval / RAGAS)

- **Retrieval:** Context Recall, Context Precision, MRR, nDCG@k | **Generation:** Faithfulness, Answer Relevancy
- **LLM-as-Judge biases & fixes:** Judges suffer from **position bias** (swap A/B order), **verbosity bias** (prefer longer answers), and **self-preference**. Use narrow integer rubrics (1–5), require `reasoning` **field before** `score`, and calibrate the judge against 100–200 human-labeled examples.

---

## 5. AI Agents & Multi-Agent Systems

### Workflows vs Agents (Anthropic Principle)

- **Workflow:** Control flow is hardcoded in Python (chaining, routing, parallel map-reduce, evaluator-optimizer); LLMs just execute nodes.
- **Agent:** LLM dynamically chooses tools, observations, and loop termination (**ReAct**).
- **Rule:** Start with the simplest deterministic workflow; add open-ended agency only when steps can't be predicted upfront.

### Multi-Agent Topologies

| Pattern                               | Shape                          | Best For                                      | Key Failure Mode & Mitigation                                                                             |
| ------------------------------------- | ------------------------------ | --------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| **Supervisor / Agents-as-Tools**      | Star                           | Specialist domains, central governance        | Extra LLM hop per step; supervisor skips workers → enforce preconditions in code                          |
| **Handoffs / Swarm**                  | Peer-to-peer                   | Conversational triage (support → billing)     | Ping-pong loops & **handoff amnesia** → pass open task explicitly, cap handoffs                           |
| **Orchestrator-Workers (**`Send`**)** | Dynamic fan-out                | Deep research, parallel solvers (`agent_lab`) | Correlated failures if prompts match → assign **diverse strategies** + independent verifier               |
| **Plan-and-Execute**                  | Planner → Executor + Replanner | Long-horizon multi-step tasks                 | Brittle initial plans → replan on unexpected tool observations                                            |
| **Evaluator-Optimizer**               | Generator ↔ Critic             | Code gen, extraction, SQL                     | Un-grounded self-critique breaks working code → ground critic in unit tests/validators; keep best attempt |

### LangGraph Internals (Production Must-Knows)

- **Pregel execution:** Runs in discrete **super-steps** (all active nodes run in parallel → state updates merge atomically via reducers → checkpoint saved).
- **State & Reducers:** Nodes return *partial* state dicts. Default overwrites; `operator.add` appends lists; `add_messages` upserts by message ID & handles `RemoveMessage`. *Gotcha:* Parallel branches updating the same key without a reducer raise `InvalidUpdateError`.
- **Routing primitives:** Pure `add_conditional_edges`, `Command(update=..., goto=...)` (atomic state update + route in one return, including `Command.PARENT` from subgraphs), and `Send(node, payload)` for dynamic map-reduce fan-out.
- **Persistence & HITL (**`interrupt`**):** Postgres checkpointer gives multi-turn `thread_id` state, crash recovery, and time-travel (`get_state_history`).
  - **Critical** `interrupt()` **gotcha:** Calling `Command(resume=val)` **re-runs the paused node from line 1**! Any DB write or external API call *before* `interrupt()` must be idempotent or placed in an earlier node.
- **Streaming modes:** `messages` (token-by-token UI), `updates` (node deltas), `values` (full state), `custom` (`get_stream_writer()` progress events), `subgraphs=True`.

### Context Engineering & Memory Taxonomy

- **Four Context Moves:** **Write** (scratchpads, todo lists, files) · **Select** (retrieve relevant memories/tools) · **Compress** (rolling summary, compact old tool outputs to fit KV budget) · **Isolate** (sub-agents get fresh contexts and return compact summaries).
- **Memory Layers:**
  - *Short-term:* Thread messages in Checkpointer (trimmed/summarized).
  - *Long-term (*`Store`*, cross-thread, user-namespaced):* **Semantic** (user facts/preferences), **Episodic** (past successful trajectories / dynamic few-shots), **Procedural** (evolving system instructions). Written on the hot path or async background workers.

### Tooling, Protocols (MCP/A2A), Sandboxing & Agent Evals

- **Tool design:** Keep `<15–20` tools per agent (accuracy drops past that → split agents or use dynamic tool retrieval); narrow typed schemas; **idempotent writes**; return actionable error strings to the LLM (`"Unknown SKU; valid format: XX-000"`) instead of raising unhandled exceptions.
- **Constrained decoding gotcha:** Grammar/JSON-schema enforcement (`xgrammar`) guarantees valid JSON *shape*, **not** semantics — the model never sees Pydantic descriptions unless you render them in the prompt, and `reasoning` fields must precede `score`/`route` fields.
- **Protocols:** **MCP (Model Context Protocol)** standardizes **Tools** (model-called), **Resources** (app-attached data), and **Prompts** over JSON-RPC (`stdio` local or **Streamable HTTP** + OAuth remote). Watch out for **tool poisoning / rug-pulls** and **indirect prompt injection** via untrusted tool results. **A2A** handles cross-vendor agent-to-agent delegation.
- **Code & Agent Sandboxing:** Isolated containers per agent (Docker / Firecracker / E2B), non-root, `--cap-drop ALL`, CPU/RAM/PID caps, wall-clock + per-command timeouts, read-only candidate mounts, network egress controls, **zero secrets in env/mounts**.
- **3-Layer Agent Evaluation:**
  1. *Deterministic unit tests:* Scripted fake LLM `AIMessage(tool_calls=...)` testing graph edges, retries, and HITL interrupts.
  2. *Trajectory evals:* Compare tool-call sequences against reference (`exact` / `unordered` / `superset`) + step/token efficiency.
  3. *Outcome evals:* Deterministic code/DB state assertions first (verification > majority voting!), rubric LLM-as-judge second.

---

## 6. LLM Inference & Optimization (Triton / vLLM / Quantization)

### Prefill vs Decode & KV Cache Math

- **Two phases:** **Prefill** processes the prompt in parallel → **compute-bound** (drives **TTFT**). **Decode** generates 1 token/step loading all weights → **memory-bandwidth-bound** (drives **TPOT / ITL**).
- **Weight memory:** \(\text{Params} \times \text{bytes/param}\) (7B @ FP16 ≈ 14 GB; @ INT4 ≈ 3.5–4 GB).
- **KV Cache memory formula:**
  \[
  \text{KV Bytes} = 2 \times n_{\text{layers}} \times n_{\text{kv\_heads}} \times d_{\text{head}} \times \text{seq\_len} \times \text{batch} \times \text{bytes}
  \]

  *(Reduced by GQA/MLA, FP8 KV cache quantization, PagedAttention, prefix caching, and context compaction).*

### Quantization

| Format | Notes |
|---|---|
| FP16 / BF16 | 2 B/param; BF16 matches FP32 exponent range (no overflow) |
| FP8 (W8A8) | Native Tensor Core matmuls on Hopper/Ada/Blackwell — sweet spot for high-QPS serving |
| INT8 (`SmoothQuant`) | W8A8; migrates activation outliers into weights so both quantize cleanly |
| INT4 (`AWQ`, `GPTQ`) | Weight-only (W4A16); **AWQ** scales salient channels identified by activation magnitude; huge bandwidth speedup at low-to-mid batch sizes |
| GGUF / NVFP4 | `llama.cpp` CPU/edge k-quants; hardware 4-bit float on Blackwell |

- **PTQ** = fast calibration on ~128–512 domain samples; **QAT** = simulate quantization in training when INT4/FP8 PTQ drops task accuracy.

### Serving Stack & High-Throughput Techniques

| Tool / Technique | Why It Matters |
|---|---|
| **vLLM / SGLang** | **PagedAttention** (<4% KV waste), **continuous batching**, **Radix/prefix caching**, **Multi-LoRA** (`--enable-lora`) |
| **TensorRT-LLM + Triton** | Compiled kernel fusion + Triton multi-model ensembles (e.g., VAD → ASR → LLM), dynamic batching, gRPC/C++ backend |
| **Chunked Prefill** | Splits long prefills into fixed token chunks batched alongside decodes → prevents long prompts from stalling TPOT |
| **Disaggregated Serving** | Separate GPU pools for compute-heavy Prefill vs bandwidth-heavy Decode, streaming KV blocks over RDMA/NVLink |
| **Speculative Decoding** | Small draft model / Medusa / EAGLE heads propose \(k\) tokens; target model verifies in **1 forward pass** (lossless, 1.5–2.5× lower TPOT) |
| **Parallelism** | **Tensor Parallel** (split layer heads/FFN within node over NVLink) vs **Pipeline Parallel** (split layers across nodes) |

---

## 7. Multimodal / VLM & Voice AI (Your Differentiators)

### VLM & Document / Invoice Pipelines (Matches Your Pitch!)

```
PDF/Image → Dynamic High-Res Tiling → Vision Encoder (ViT/SigLIP) → Projector/Merger → LLM → JSON + BBoxes → Invariant Checks → Auto-Approve or L1/L2 HITL
```

- **How modern VLMs work:** Vision backbone (**SigLIP / ViT**) encodes patches → **MLP / Pixel-Shuffle projector** compresses visual tokens into LLM embedding space → concatenated with text tokens.
- **Invoice resolution gotcha:** Fixed \(336\times 336\) resize blurs tiny line-item digits! Modern VLMs (Qwen2.5-VL, InternVL, Pixtral) use **native dynamic resolution / tiling**, which preserves text legibility but emits **1K–4K+ visual tokens/page** (watch TTFT & KV cache!).
- **VLM vs OCR + Text LLM:** OCR (`PaddleOCR`/Textract) + text LLM is cheaper and faster on clean templates; **end-to-end VLMs** win on messy multi-column tables, stamps, skew, handwriting, and direct visual grounding.
- **Production HITL routing (never trust verbalized LLM confidence!):** Gate auto-approval on **deterministic accounting invariants** (`sum(line_items) + tax - discount == total`, valid PO/vendor match in ERP, currency/date sanity, duplicate-invoice check) + field token logprobs / multi-pass agreement; route anomalies to L1/L2 human queues with bounding-box highlights and immutable audit logs.

### Real-Time Voice AI (< 800 ms Voice-to-Voice)

```
Mic → AEC + VAD + Semantic Turn Detector → Streaming ASR (Faster Whisper) → LLM (SSE) → Clause-Chunked TTS → Speaker
                     ↑_________________________ WebRTC / Twilio Media Stream _________________________↓
```

- **Turn detection / Endpointing:** Raw silence VAD either cuts users off mid-pause ("let me check...") or adds 1s lag → combine VAD (`Silero`) with a lightweight **semantic end-of-turn (EOU) model** that inspects partial transcript completeness.
- **Full-duplex Barge-in:** Require **AEC (Acoustic Echo Cancellation)** so the bot doesn't hear itself; on user speech start, immediately cancel the in-flight LLM stream, flush the TTS/WebRTC outbound audio buffer, and **truncate assistant chat history to only the words actually spoken** before interruption.
- **Latency tricks:** Prefer **WebRTC** (UDP, jitter handling) for client audio; stream LLM tokens into TTS at the **first clause/punctuation boundary**; play natural conversational acks/fillers when awaiting slow backend tool calls; evaluate **speech-to-speech** models (Nova Sonic, GPT Realtime) vs cascaded ASR→LLM→TTS (cascaded gives easier PII redaction, deterministic guardrails, and auditability).

---

## 8. ML System Design & Security Framework

**45-Minute Blueprint:**

1. **Clarify (5m):** Users, QPS/concurrency, latency SLA (TTFT p95), accuracy bar, data residency/PII, budget.
2. **Metrics (3m):** *Offline* (Recall@k, exact-match, trajectory pass rate) + *Online* (task completion, deflection rate, HITL escalation %, cost/successful task, p95 latency).
3. **Data & Ingestion (5m):** Sources, parsing, ACL tags, synthetic/human golden eval set.
4. **Architecture (12m):** Baseline first (Prompt + Hybrid RAG), then justify upgrades (Reranker, LoRA/Distillation, LangGraph HITL).
5. **Serving & Cost (10m):** Model routing (0.6B/8B vs 70B), quantization (FP8/AWQ), prefix caching, async queues (Celery/Kafka) for batch docs.
6. **Guardrails, Evals & Monitoring (10m):** Security, CI eval gates, canary rollout, drift & trace alerts.

### Critical Production Gotchas to Mention Proactively

- **Semantic cache trap:** Never semantic-cache queries with entity IDs, user-specific state, or negations (*"Cancel order #123"* vs *"#124"*, *"Is X covered?"* vs *"Is X NOT covered?"* have ~0.98 cosine similarity!). Scope caches strictly or normalize intent + exact parameters first.
- **Direct vs Indirect Prompt Injection:** *Direct* = malicious user prompt; *Indirect* = untrusted instructions hidden inside retrieved docs, invoices, emails, or MCP tool outputs. **Defense-in-depth:** spotlight/delimit untrusted data, input/output guardrail classifiers, least-privilege scoped credentials, outbound domain allowlists, and **deterministic code/HITL policy gates** on state-changing tools.

---

## 9. MLOps & LLMOps

| Area | Tools |
|---|---|
| Experiment & Eval tracking | Weights & Biases, MLflow, LangSmith, **Langfuse** (self-hostable), DeepEval / Promptfoo |
| Data / Model versioning | Hugging Face Hub, DVC, MLflow Model Registry, LakeFS |
| Orchestration | Airflow, Prefect, Dagster, Kubeflow |
| Serving & Gateway | vLLM, Triton, TensorRT-LLM, LiteLLM / Envoy AI Gateway, FastAPI |
| Deployment & Infra | Docker, Kubernetes (KServe / Ray Serve), Helm, GitHub Actions CI |
| Monitoring | Prometheus + Grafana (GPU/KV-cache/TTFT), OpenTelemetry, Evidently (drift) |

- **Data drift** = input query/doc distribution shifts; **Concept drift** = correct answer/policy changes for the same input.
- **Rollout progression:** Offline Golden Eval → **Shadow mode** (mirror live traffic, zero user impact) → **Canary (1–5%)** → **A/B test** with automatic rollback triggers.

---

## 10. Behavioral — STAR Stories From Your Resume

| Question | Story | Key Metric / Proof |
|---|---|---|
| Biggest impact | Enterprise RAG AI Assistant (hybrid search, ACLs, citations) | **20,000+ employees** |
| Hardest technical problem | Full-duplex voice agent (streaming pipeline, barge-in, turn latency) | **~80% support workload target** |
| Reliability / ownership | VLM Invoice HITL pipeline (accounting invariants, retries, L1/L2 queues, audit logs) | **Zero unverified payouts** |
| Performance optimization | Triton + quantization + batching / KV cache tuning | **Lower GPU cost & p95 latency** |
| Scale & systems depth | High-concurrency Twisted async TCP server | **100K+ concurrent connections** |
| Initiative / R&D | Local post-training & sandboxed multi-agent lab (LoRA/DPO/GRPO, LangGraph) | **Measured ablations & eval harnesses** |

**STAR:** Situation (1 line) → Task (1 line) → **Action (70% of time — say "I designed / measured / fixed")** → Result (quantified).

---

## 11. Rapid-Fire Q&A

- **Why does RAG hallucinate?** Bad retrieval, noisy chunks ("lost in the middle"), or weak grounding instructions → inspect retrieval recall first.
- **Why not 1M-token context instead of RAG?** \(O(n^2)\) prefill cost & high TTFT, quality degradation in the middle, no per-chunk RBAC/ACL enforcement, and stale context.
- **Why** `-100` **in SFT labels?** PyTorch `CrossEntropyLoss(ignore_index=-100)` masks prompt tokens so gradients only come from assistant completions (keep `<eos>` unmasked!).
- **Why target all linear layers in LoRA?** QLoRA ablations showed adapting FFN (`gate, up, down`) + all attention projections matches full FT far better than raising rank \(r\) on `q, v` alone.
- **Why did DPO hurt generation fluency?** Over-optimization / missing SFT warmup caused chosen log-probs to drop while widening the margin → lower LR (`5e-6`), increase \(\beta\), or add NLL loss.
- **How does GRPO save memory over PPO?** Replaces PPO's full-size value/critic network with the mean/std reward of \(G\) sampled completions for the same prompt.
- **What happens on LangGraph** `Command(resume=...)`**?** The interrupted node re-runs from the top; anything before `interrupt()` must be side-effect-free/idempotent.
- **Why did constrained JSON output give bad answers?** Grammar enforcement hides schema descriptions from the model and forces premature token commitments if `score`/`label` comes before `reasoning`.
- **Cosine vs dot product?** Identical when vectors are L2-normalized; dot product is faster in SIMD/GPU kernels.
- **How do you handle PII?** Detect/redact pre-LLM (`Presidio` + regex checksums), self-host behind VPC, scrub traces/logs, tokenize reversible entities only when tools need them.

---

## 12. Python & Coding Refreshers

```python
# Async rate-limited LLM fan-out with timeout & error capture
import asyncio

sem = asyncio.Semaphore(10)
async def safe_call(q: str, timeout: float = 15.0):
    async with sem:
        return await asyncio.wait_for(call_llm(q), timeout=timeout)

results = await asyncio.gather(*(safe_call(q) for q in queries), return_exceptions=True)

# Numerically stable Scaled Dot-Product Attention (with causal mask)
import numpy as np
def causal_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    T, d_k = Q.shape
    scores = (Q @ K.T) / np.sqrt(d_k)
    mask = np.triu(np.ones((T, T), dtype=bool), k=1)
    scores = np.where(mask, -np.inf, scores)
    w = np.exp(scores - scores.max(axis=-1, keepdims=True))
    w /= w.sum(axis=-1, keepdims=True)
    return w @ V

# LoRA forward pass: h = Wx + (alpha / r) * B(Ax)
def lora_linear(x: np.ndarray, W: np.ndarray, A: np.ndarray, B: np.ndarray, alpha: float) -> np.ndarray:
    r = A.shape[0]
    return x @ W.T + (alpha / r) * ((x @ A.T) @ B.T)
```

**Also practice:** Softmax + cross-entropy gradient, BPE / chunker with overlap, `OrderedDict` LRU cache + TTL, token-bucket rate limiter, FastAPI `StreamingResponse` (SSE), and core DSA (hash maps, heaps/top-k, sliding window, topological sort / cycle detection for agent graphs).

---

## 13. Questions to Ask the Interviewer

1. How do you gate prompt, retrieval, or model updates today — offline golden sets, trajectory evals, or shadow traffic?
2. Where do you draw the line between API frontier models and self-hosted / fine-tuned open models?
3. What's the biggest production bottleneck right now: retrieval precision, agent reliability, latency, or GPU cost?
4. How are human-in-the-loop reviews and user feedback wired back into evals or post-training data?
5. What would a strong 90-day impact look like in this role?

---

## 14. Final-Week Plan

| Day | Focus |
|---|---|
| 1 | ML fundamentals, metrics & classic algorithms (§1) |
| 2 | Modern Transformers, RoPE/GQA/MLA, attention & LoRA in NumPy (§2, §12) |
| 3 | Post-training deep dive: SFT masking, LoRA/QLoRA math, DPO vs GRPO, forgetting (§3) |
| 4 | RAG architecture, ACLs, hybrid RRF, LLM-as-judge biases (§4) |
| 5 | LangGraph internals, multi-agent failure modes, MCP, sandboxing & agent evals (§5) |
| 6 | Inference math (KV cache, prefill/decode), VLM invoices & Voice barge-in + 2 mock designs (§6, §7, §8) |
| 7 | STAR stories with hard numbers + 60s pitch out loud (§0, §10, §11) |

**Good luck, Joseph! 🚀**
