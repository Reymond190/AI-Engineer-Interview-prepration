"""
15 - Attention and a tiny transformer from scratch
==================================================

Interview topics
----------------
- Scaled dot-product attention: softmax(Q K^T / sqrt(d_k) + mask) V. Each
  query takes a weighted average of the values, weighted by query-key
  similarity. Q = X W_q, K = X W_k, V = X W_v (self-attention: all from the
  same sequence; cross-attention: K, V from another one).
- Why / sqrt(d_k): for random unit-variance q, k the dot product has
  variance d_k. Without scaling, logits grow with d_k, softmax saturates to
  one-hot and its gradient vanishes (section 2).
- Causal mask: position t may only attend to <= t (set future logits to
  -inf before softmax) -> decoder-only LMs train on all positions in
  parallel ("teacher forcing").
- Multi-head: h heads of size d/h in parallel, concatenated, then W_o.
  Same FLOPs as one big head, but each head can attend to different things.
  GQA/MQA share K/V across heads to shrink the KV cache.
- Attention is PERMUTATION EQUIVARIANT: shuffle the tokens and the outputs
  shuffle the same way -> order must be injected: learned/sinusoidal
  absolute positions, relative positions, RoPE (rotate q, k by a
  position-dependent angle; used by Llama/Qwen), ALiBi. (Sections 4, 5.)
- Transformer block (pre-LN): x = x + Attn(LN(x)); x = x + MLP(LN(x)).
  MLP is ~2/3 of the params (4x expansion, or SwiGLU). Residual stream +
  norm = lessons 10-11.
- Cost: O(n^2 d) time and O(n^2) attention memory per layer in sequence
  length n -> FlashAttention (tiling, never materialises the n x n matrix,
  same math), sparse/sliding-window attention, linear attention / SSMs.
- KV cache: when generating, K and V of past tokens don't change, so cache
  them; each new token costs O(n d) instead of recomputing O(n^2 d). Cache
  size = 2 * layers * kv_heads * head_dim * n * bytes per sequence -> the
  serving bottleneck (vLLM's PagedAttention manages it in pages).

Run:  python ml_course/15_attention_transformer.py [--quick]
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import device, kv, parse_args, section, table


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = False, scale: bool = True) -> torch.Tensor:
    d = q.shape[-1]
    logits = q @ k.transpose(-2, -1) / (math.sqrt(d) if scale else 1.0)
    if causal:
        n, m = q.shape[-2], k.shape[-2]
        mask = torch.ones(n, m, dtype=torch.bool, device=q.device).tril(diagonal=m - n)  # also right for a KV cache
        logits = logits.masked_fill(~mask, float("-inf"))
    return logits.softmax(-1) @ v


class MHA(nn.Module):
    def __init__(self, d: int, heads: int) -> None:
        super().__init__()
        self.h, self.dh = heads, d // heads
        self.qkv, self.o = nn.Linear(d, 3 * d), nn.Linear(d, d)

    def forward(self, x: torch.Tensor, causal: bool = False) -> torch.Tensor:
        b, n, d = x.shape
        q, k, v = self.qkv(x).view(b, n, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)   # each (b, h, n, dh)
        out = attention(q, k, v, causal=causal)                                        # (b, h, n, dh)
        return self.o(out.transpose(1, 2).reshape(b, n, d))                            # concat heads, project


class Block(nn.Module):
    def __init__(self, d: int, heads: int) -> None:
        super().__init__()
        self.ln1, self.attn, self.ln2 = nn.LayerNorm(d), MHA(d, heads), nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))          # pre-LN residual blocks
        return x + self.mlp(self.ln2(x))


class TinyTransformer(nn.Module):
    def __init__(self, vocab: int, max_len: int, d: int = 64, heads: int = 4, layers: int = 2, positions: bool = True) -> None:
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_len, d) if positions else None
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))
        self.ln, self.head = nn.LayerNorm(d), nn.Linear(d, vocab)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.tok(x)
        if self.pos is not None:
            h = h + self.pos(torch.arange(x.shape[1], device=x.device))
        for blk in self.blocks:
            h = blk(h)
        return self.head(self.ln(h[:, -1]))     # read out at the LAST position, like the RNNs in lesson 14


def main() -> None:
    args = parse_args(__doc__)
    dev = device()
    torch.manual_seed(args.seed)

    section("1. Scaled dot-product attention from scratch vs torch (with causal mask)")
    q, k, v = (torch.randn(2, 4, 10, 16, device=dev) for _ in range(3))
    ours = attention(q, k, v, causal=True)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    kv("sdpa-parity", max_diff=(ours - ref).abs().max().item())

    section("2. Why divide by sqrt(d_k): softmax saturation")
    rows = []
    for d in [16, 64, 256, 1024]:
        q, k = torch.randn(1000, 1, d, device=dev), torch.randn(1000, 32, d, device=dev)
        row = {"d_k": d}
        for scaled in [False, True]:
            logits = (q @ k.transpose(-2, -1)).squeeze(1) / (math.sqrt(d) if scaled else 1.0)
            p = logits.softmax(-1)
            ent = -(p * p.clamp_min(1e-30).log()).sum(-1).mean().item()
            row["unscaled max p" if not scaled else "scaled max p"] = p.max(-1).values.mean().item()
            row["unscaled entropy" if not scaled else "scaled entropy"] = ent
            if not scaled:  # softmax Jacobian size: how much a small logit change moves the output
                row["unscaled grad"] = (p * (1 - p)).sum(-1).mean().item()
            else:
                row["scaled grad"] = (p * (1 - p)).sum(-1).mean().item()
        rows.append(row)
    table(rows, ["d_k", "unscaled max p", "scaled max p", "unscaled entropy", "scaled entropy", "unscaled grad", "scaled grad"])
    kv("sqrt-d", unscaled_maxp_1024=rows[-1]["unscaled max p"], scaled_maxp_1024=rows[-1]["scaled max p"])
    print(f"  32 keys: uniform attention has entropy ln 32 = {math.log(32):.2f}. Unscaled at d=1024 is almost one-hot,")
    print("  and sum p(1-p) (the softmax gradient scale) collapses. Scaled stays the same for every d_k.")

    section("3. Multi-head attention from scratch vs nn.MultiheadAttention (same weights)")
    d, h = 32, 4
    ours = MHA(d, h).to(dev)
    ref = nn.MultiheadAttention(d, h, batch_first=True).to(dev)
    with torch.no_grad():
        ref.in_proj_weight.copy_(ours.qkv.weight)
        ref.in_proj_bias.copy_(ours.qkv.bias)
        ref.out_proj.weight.copy_(ours.o.weight)
        ref.out_proj.bias.copy_(ours.o.bias)
    x = torch.randn(3, 7, d, device=dev)
    causal_mask = torch.ones(7, 7, dtype=torch.bool, device=dev).triu(1)   # True = NOT allowed (torch convention)
    diff = (ours(x, causal=True) - ref(x, x, x, attn_mask=causal_mask, need_weights=False)[0]).abs().max().item()
    kv("mha-parity", max_diff=diff)

    section("4. Attention without positions is permutation-equivariant")
    blk = Block(32, 4).to(dev).eval()
    x = torch.randn(1, 6, 32, device=dev)
    perm = torch.tensor([3, 0, 5, 1, 4, 2], device=dev)
    with torch.no_grad():
        diff = (blk(x)[:, perm] - blk(x[:, perm])).abs().max().item()
    kv("perm-equivariance", max_diff=diff)
    print("  f(shuffled x) == shuffled f(x): the block has no idea about order unless we add position information.")

    section("5. The lesson-14 memory task: return the FIRST token of a length-50 sequence")
    vocab, length = 8, 50
    steps = 1500 if args.quick else 4000
    rows = []
    for name, pos in [("transformer + learned positions", True), ("transformer, NO positions", False)]:
        torch.manual_seed(args.seed)
        m = TinyTransformer(vocab, length, positions=pos).to(dev)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.01)
        for _ in range(steps):
            x = torch.randint(0, vocab, (128, length), device=dev)
            loss = F.cross_entropy(m(x), x[:, 0])
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            x = torch.randint(0, vocab, (2000, length), device=dev)
            acc = (m(x).argmax(1) == x[:, 0]).float().mean().item()
        rows.append({"model": name, "params": sum(p.numel() for p in m.parameters()), f"acc @ length {length}": acc})
    table(rows)
    kv("first-token", with_pos=rows[0][f"acc @ length {length}"], no_pos=rows[1][f"acc @ length {length}"])
    print("  Lesson 14: RNN/LSTM/GRU are at chance (0.125) here. Attention reaches token 0 in ONE hop, whatever the")
    print("  distance, so length is no obstacle. But without positions it can't tell which token was first.")

    section("6. KV cache: incremental decoding == full recompute, at a fraction of the work")
    d, n = 64, 64
    wq, wk, wv = (torch.randn(d, d, device=dev) / math.sqrt(d) for _ in range(3))
    xs = torch.randn(1, n, d, device=dev)
    full = attention(xs @ wq, xs @ wk, xs @ wv, causal=True)
    k_cache, v_cache, outs = [], [], []
    for t in range(n):                          # generate one position at a time
        xt = xs[:, t:t + 1]
        k_cache.append(xt @ wk)
        v_cache.append(xt @ wv)
        outs.append(attention(xt @ wq, torch.cat(k_cache, 1), torch.cat(v_cache, 1), causal=True))
    inc = torch.cat(outs, 1)
    # Multiply-adds for the K/V projections: no cache recomputes all t of them at step t; the cache does 1.
    no_cache = sum(2 * t * d * d for t in range(1, n + 1))
    with_cache = n * 2 * d * d
    kv("kv-cache", max_diff=(inc - full).abs().max().item(), kv_proj_flops_ratio=no_cache / with_cache)
    layers, kv_heads, head_dim = 32, 8, 128      # a Llama-3-8B-like config, bf16
    per_token = 2 * layers * kv_heads * head_dim * 2
    print(f"  KV cache size for an 8B GQA model: {per_token / 1024:.0f} KiB per token -> "
          f"{per_token * 32768 / 2**30:.1f} GiB for one 32k-token sequence.")


if __name__ == "__main__":
    main()
