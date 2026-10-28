# AI/ML Engineer Interview Cheat Sheet

*Joseph Reymond B — tailored to LLM / RAG / Agentic AI / ML Platform roles*

---

## 0. Your 60-Second Pitch

> "I'm a Technical Lead at Tata Communications with 7+ years in Python backend and applied AI. I built an enterprise RAG assistant used by 20,000+ employees, a real-time voice agent designed to cut support workload by ~80%, and a VLM-based invoice pipeline with human-in-the-loop approvals. I also own LLM serving on Triton with quantization. My strength is taking LLM ideas to reliable, scalable production systems."

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
| Vanishing gradients | Fix with ReLU/GELU, residual connections, normalization, good init |
| BatchNorm vs LayerNorm | BN normalizes across batch (CNNs); LN across features per token (Transformers) |
| Data leakage | Info from test/future leaking into training → inflated metrics. Split before preprocessing |

### Metrics

| Task | Metrics | When |
|---|---|---|
| Classification | Precision = TP/(TP+FP), Recall = TP/(TP+FN), F1 = harmonic mean | Imbalanced → F1, PR-AUC over accuracy |
| Ranking/Retrieval | Recall@k, MRR, nDCG, Hit Rate | RAG retrieval evaluation |
| Regression | MAE (robust), RMSE (penalizes large errors), R² | |
| Generation | BLEU/ROUGE (weak), BERTScore, LLM-as-judge, human eval | |
| LLM | Perplexity (lower = better), faithfulness, answer relevancy | |

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

- **Self-attention:** `Attention(Q,K,V) = softmax(QKᵀ / √d_k) · V`
  - √d_k scaling keeps softmax from saturating
- **Multi-head:** several attention heads learn different relationships, then concat + project
- **Positional encoding:** sinusoidal (original), learned, **RoPE** (rotary, used in Llama), ALiBi
- **Block:** Attention → Add & Norm → FFN → Add & Norm (modern: Pre-LN, RMSNorm, SwiGLU)
- **Complexity:** O(n²) in sequence length → why long context is expensive

### Architecture Families

| Type | Examples | Use |
|---|---|---|
| Encoder-only | BERT, RoBERTa | Classification, embeddings, NER |
| Decoder-only | GPT, Llama, Mistral, Qwen | Text generation, chat |
| Encoder-decoder | T5, BART, Whisper | Translation, summarization, ASR |

### Key Terms

- **Tokenization:** BPE / SentencePiece; ~0.75 words per token in English
- **KV Cache:** stores past keys/values so each new token doesn't recompute attention → faster decoding, but uses lots of memory
- **Attention variants:** MQA / **GQA** (share K/V heads → smaller KV cache), **FlashAttention** (IO-aware, exact, faster)
- **MoE (Mixture of Experts):** router activates only a few expert FFNs per token → more parameters at similar compute (Mixtral, DeepSeek)
- **Decoding:** greedy, beam search, temperature, top-k, top-p (nucleus)
  - Temperature ↑ → more random; 0 → deterministic

---

## 3. LLM Training & Adaptation

| Stage | What |
|---|---|
| Pre-training | Next-token prediction on huge corpora |
| SFT | Supervised fine-tuning on instruction/response pairs |
| RLHF | Reward model from human preferences + PPO |
| DPO | Directly optimize on preference pairs — no reward model, simpler and stable |

### When to Use What (classic interview question)

| Need | Choose |
|---|---|
| Fresh / private / changing knowledge | **RAG** |
| Change style, format, tone, or domain behavior | **Fine-tuning** |
| Quick task steering | **Prompt engineering / few-shot** |
| Cheap, fast, narrow task | **Distill** into a small model |

> Rule of thumb: Prompt → RAG → Fine-tune, in order of cost.

### PEFT (Parameter-Efficient Fine-Tuning)

- **LoRA:** freeze weights, learn low-rank matrices `W + BA` (rank r ≈ 8–64); trains <1% of params
- **QLoRA:** LoRA on a 4-bit quantized base model → fine-tune 65B on one GPU
- Key hyperparams: rank `r`, `alpha`, target modules (q_proj, v_proj, …), learning rate

---

## 4. RAG — Your Strongest Area

### Pipeline

```
Ingest → Parse → Chunk → Embed → Index (Vector DB)
Query → (Rewrite) → Retrieve → Rerank → Build Prompt → LLM → Guardrails → Answer + Citations
```

### Design Decisions & Talking Points

| Component | Options / Trade-offs |
|---|---|
| **Chunking** | Fixed-size + overlap (simple); recursive/semantic (better boundaries); parent-child (retrieve small, feed large). Typical: 300–800 tokens, 10–20% overlap |
| **Embeddings** | OpenAI text-embedding-3, BGE, E5, GTE; check MTEB leaderboard; normalize for cosine |
| **Vector DB** | FAISS (library), pgvector (Postgres), Milvus, Qdrant, Weaviate, Pinecone, Elasticsearch |
| **Index types** | HNSW (fast, memory-heavy), IVF (clustered), PQ (compressed) |
| **Retrieval** | Dense + **BM25 hybrid** (fixes exact-match: IDs, codes, names) fused with RRF |
| **Reranking** | Cross-encoder (bge-reranker, Cohere Rerank) on top-k candidates → big precision gain |
| **Query side** | Query rewriting, HyDE, multi-query, metadata filters |
| **Advanced** | Agentic RAG (LangGraph), GraphRAG, self-RAG, corrective RAG |

### Common RAG Failures → Fixes

| Problem | Fix |
|---|---|
| Right doc not retrieved | Hybrid search, better chunking, query rewriting, metadata filters |
| Retrieved but wrong answer | Reranking, fewer but better chunks, clearer prompt |
| Hallucination | Force citations, "answer only from context", faithfulness checks |
| Lost in the middle | Put most relevant chunks first/last; fewer chunks |
| Stale data | Incremental re-indexing pipeline |
| Access control | Filter by user permissions at retrieval time (critical at 20K users!) |

### RAG Evaluation (DeepEval / RAGAS)

- **Retrieval:** Context Precision, Context Recall, Hit Rate, MRR
- **Generation:** Faithfulness (grounded in context?), Answer Relevancy
- Build a **golden dataset** of Q/A pairs; run evals in CI on every change

---

## 5. AI Agents

### Core Concepts

- **Agent = LLM + tools + memory + loop**
- **ReAct:** Thought → Action → Observation → repeat
- **Function / tool calling:** LLM outputs structured JSON → your code executes → result returned
- **Planning patterns:** Plan-and-execute, reflection/self-critique, router, supervisor + workers (multi-agent)
- **Memory:** short-term (conversation window), long-term (vector store / DB), summarization
- **MCP (Model Context Protocol):** standard way to expose tools/data to LLMs

### LangGraph (you've used it)

- State machine / graph: **nodes** (functions), **edges** (conditional routing), shared **state**
- Supports cycles, checkpoints, human-in-the-loop interrupts — better than linear chains for agents

### Production Concerns

- Max-iteration limits, timeouts, cost caps
- Tool input validation; least-privilege permissions
- **Prompt injection** defense: treat retrieved/tool content as data, not instructions
- Human-in-the-loop for irreversible actions (your invoice L1/L2 flow is a great example)
- Tracing: LangSmith, Langfuse, OpenTelemetry

---

## 6. LLM Inference & Optimization (Triton / Quantization)

### Quantization

| Format | Notes |
|---|---|
| FP32 → FP16/BF16 | 2× smaller; BF16 has FP32 range (safer for training) |
| INT8 | ~4× smaller than FP32; small accuracy loss |
| INT4 (GPTQ, AWQ) | Weight-only; huge memory savings; AWQ protects salient weights |
| GGUF | llama.cpp format for CPU / edge |
| FP8 | Native on H100; good speed and quality balance |

- **PTQ** (post-training quantization) = quick; **QAT** (quantization-aware training) = better accuracy, more work
- **Memory estimate:** params × bytes. 7B @ FP16 ≈ 14 GB; @ INT4 ≈ 3.5–4 GB (+ KV cache)

### Serving Stack

| Tool | Strength |
|---|---|
| **vLLM** | PagedAttention, continuous batching — highest throughput open-source |
| **TensorRT-LLM** | NVIDIA-optimized kernels, best raw GPU perf |
| **Triton Inference Server** | Multi-model, multi-framework serving, dynamic batching, ensembles |
| TGI, SGLang, Ollama | Alternatives (Ollama for local) |

### Key Techniques

- **Continuous batching:** add/remove requests per token step → much better GPU utilization than static batching
- **PagedAttention:** KV cache in pages, like OS virtual memory → less fragmentation
- **Speculative decoding:** small draft model proposes tokens, large model verifies in parallel
- **Prefix caching:** reuse KV cache for shared system prompts
- **Tensor / pipeline parallelism:** split model across GPUs

### Metrics to Quote

- **TTFT** (time to first token) — latency users feel
- **TPOT / ITL** (time per output token)
- **Throughput** (tokens/sec, requests/sec)
- GPU utilization, cost per 1K tokens

---

## 7. Voice AI (Your Differentiator) 

```
Mic → VAD → ASR (Faster Whisper) → LLM (streaming) → TTS (ElevenLabs, streaming) → Speaker
              ↑______________ WebSocket / Twilio media stream ______________↓
```

- **Latency budget:** aim for < ~800 ms voice-to-voice. Stream everything; start TTS on the first sentence
- **VAD** (voice activity detection) detects end of speech; **barge-in** = user interrupts the bot (full-duplex)
- **Faster Whisper** = CTranslate2 reimplementation of Whisper → ~4× faster, supports INT8
- Speech-to-speech models (e.g. AWS Nova Sonic, GPT realtime) skip the ASR→LLM→TTS chain

---

## 8. ML System Design Framework

**Use this structure for any design question (45 min):**    

1. **Clarify** (5 min) — users, scale, latency, accuracy, budget, privacy
2. **Define metrics** — offline (recall@k, F1) and online (CSAT, deflection rate, latency)
3. **Data** — sources, labeling, freshness, PII
4. **High-level architecture** — draw boxes
5. **Model choice** — baseline first, then upgrade; API vs self-hosted
6. **Serving** — batching, caching, autoscaling, fallbacks
7. **Evaluation & monitoring** — drift, hallucination rate, feedback loop
8. **Trade-offs & scaling** — cost, latency, failure modes

### Likely Design Prompts

- Design an enterprise knowledge assistant (→ your Project 3)
- Design a customer-support voice bot (→ your Project 4)
- Design document/invoice extraction at scale (→ your Project 1)
- Serve a 70B model to 1,000 concurrent users cost-efficiently
- Build a recommendation system / semantic search

### Production Checklist

- [ ] Semantic cache (Redis) for repeat queries
- [ ] Model routing: small model for easy queries, big for hard
- [ ] Rate limits, retries with backoff, circuit breakers, fallback model
- [ ] Guardrails: PII redaction, toxicity, jailbreak detection
- [ ] Observability: traces, token cost, latency, user feedback (👍/👎)
- [ ] Async queues (Celery/RabbitMQ) for long jobs

---

## 9. MLOps

| Area | Tools |
|---|---|
| Experiment tracking | MLflow, Weights & Biases |
| Data/Model versioning | DVC, MLflow Model Registry, Hugging Face Hub |
| Orchestration | Airflow, Kubeflow, Prefect |
| Serving | Triton, vLLM, KServe, BentoML, FastAPI |
| Deployment | Docker, Kubernetes, Helm, CI/CD (GitHub Actions) |
| Monitoring | Prometheus + Grafana, Evidently (drift), Langfuse (LLM) |

- **Data drift** = input distribution changes; **concept drift** = input→output relationship changes
- Deployment strategies: **shadow**, **canary**, **A/B**, blue-green
- **Feature store** (Feast) = consistent features for training and serving

---

## 10. Behavioral — STAR Stories From Your Resume

| Question | Story | Key Metric |
|---|---|---|
| Biggest impact | Enterprise AI Assistant | 20,000+ employees |
| Hardest technical problem | Real-time full-duplex voice latency | ~80% workload reduction |
| Reliability / ownership | Invoice HITL pipeline: retries, audit logs, L1/L2 approval | Accuracy before payment |
| Performance optimization | Triton + quantization | Lower compute per request |
| Scale | Twisted TCP server | 100K+ concurrent connections |
| Leadership / mentoring | Leading team, mentoring juniors, AI roadmap | Team outcomes |

**STAR:** Situation (1 line) → Task (1 line) → **Action (most time, say "I")** → Result (number).

> Prepare real numbers before interviews: latency before/after, GPU memory saved, accuracy %, cost saved.

---

## 11. Rapid-Fire Q&A

- **Why does RAG hallucinate?** Bad retrieval, too much irrelevant context, or the model ignores context → fix retrieval first.
- **Cosine vs dot product?** Same thing on normalized vectors; cosine ignores magnitude.
- **Why not just use long context instead of RAG?** Cost, latency, lost-in-the-middle, no access control, and data changes.
- **How do you reduce LLM cost?** Caching, smaller or routed models, prompt compression, quantization, batching.
- **How do you evaluate an LLM app with no labels?** LLM-as-judge with rubric, synthetic golden sets, user feedback, spot-check by humans.
- **Temperature for RAG?** Low (0–0.3) for factual answers.
- **Embedding dimension trade-off?** Higher = more expressive but more storage and slower search; Matryoshka embeddings allow truncation.
- **What is hallucination?** Fluent output not grounded in facts or the given context.
- **Context window vs max tokens?** Window = input + output total; max_tokens = output cap.
- **How do you handle PII?** Detect/redact before the LLM call (Presidio), self-host for sensitive data, log carefully.
- **Structured output reliably?** JSON mode / schema-constrained decoding + Pydantic validation + retry.

---

## 12. Python & Coding Refreshers

```python
# Async concurrency — fan out LLM calls
import asyncio
results = await asyncio.gather(*(call_llm(q) for q in queries))

# Rate-limited concurrency
sem = asyncio.Semaphore(10)
async def limited(q):
    async with sem:
        return await call_llm(q)

# Cosine similarity
import numpy as np
def cosine(a, b): return a @ b / (np.linalg.norm(a) * np.linalg.norm(b))

# Top-k from similarity scores
top_k = np.argsort(-scores)[:k]

# Simple chunker with overlap
def chunk(text, size=500, overlap=50):
    return [text[i:i+size] for i in range(0, len(text), size - overlap)]
```

**Also practice:** implement softmax, attention from scratch in NumPy, a basic LRU cache, a FastAPI streaming endpoint (SSE), and common DSA (hash maps, two pointers, sliding window, heaps, BFS/DFS).

```python
# Scaled dot-product attention in NumPy
def attention(Q, K, V):
    scores = Q @ K.T / np.sqrt(K.shape[-1])
    w = np.exp(scores - scores.max(-1, keepdims=True))
    w /= w.sum(-1, keepdims=True)
    return w @ V
```

---

## 13. Questions to Ask the Interviewer

1. How do you evaluate LLM features before and after release?
2. Do you self-host models or use APIs, and how is that decided?
3. What's the biggest bottleneck right now: data, latency, cost, or quality?
4. How do AI engineers and research or product teams collaborate here?
5. What does success look like in the first 90 days?

---

## 14. Final-Week Plan

| Day | Focus |
|---|---|
| 1 | ML fundamentals + metrics (§1) |
| 2 | Transformers + attention from scratch (§2, §12) |
| 3 | Fine-tuning, LoRA, RAG vs FT (§3) |
| 4 | RAG deep dive + evaluation (§4) |
| 5 | Agents + inference optimization (§5, §6) |
| 6 | 2 mock system designs out loud (§8) |
| 7 | STAR stories + pitch rehearsal (§0, §10) |

**Good luck, Joseph! 🚀**
