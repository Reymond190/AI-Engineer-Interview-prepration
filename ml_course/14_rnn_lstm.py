"""
14 - RNNs, LSTMs, GRUs and long-range memory
============================================

Interview topics
----------------
- Vanilla RNN: h_t = tanh(W_h h_{t-1} + W_x x_t + b). Same weights at every
  step (weight sharing over time). Backprop through time (BPTT) multiplies
  the Jacobian dh_t/dh_{t-1} = diag(tanh') W_h once per step, so gradients to
  early steps vanish (or explode) exponentially in the gap length. In
  practice a vanilla RNN can't learn dependencies longer than ~10-20 steps.
- LSTM: a separate cell state c_t updated ADDITIVELY:
    c_t = f_t * c_{t-1} + i_t * g_t,   h_t = o_t * tanh(c_t)
  forget (f), input (i), output (o) gates are sigmoids. When f ~ 1 the
  gradient flows through c almost unchanged (the "constant error carousel")
  -> the same idea as residual connections, in time. Init the forget-gate
  bias to ~1 so it starts by remembering.
- GRU: merges cell and hidden state, 2 gates (update z, reset r), fewer
  params, usually similar quality.
- Exploding gradients in RNNs -> gradient clipping (lesson 10).
- Truncated BPTT for very long sequences: backprop only k steps back.
- Bidirectional RNNs (offline tasks), seq2seq encoder-decoder; the
  fixed-size hidden state is a bottleneck -> attention (Bahdanau 2014) ->
  transformers (lesson 15), which replace recurrence with direct access to
  every position (O(1) path length) and parallelise over time.
- Recent: state-space models / linear RNNs (S4, Mamba, RWKV, Gated
  DeltaNet as in Qwen3.5's hybrid layers) bring back recurrence with
  parallel training and O(1)-per-token inference memory.

Run:  python ml_course/14_rnn_lstm.py [--quick]
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import device, kv, parse_args, section, table

VOCAB = 8


def batch(n: int, length: int, dev: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Sequence of random tokens; the label is the FIRST token. The model must carry it through `length` steps."""
    x = torch.randint(0, VOCAB, (n, length), device=dev)
    return x, x[:, 0].clone()


class SeqClassifier(nn.Module):
    def __init__(self, kind: str, hidden: int = 64) -> None:
        super().__init__()
        self.emb = nn.Embedding(VOCAB, 32)
        cls = {"RNN (tanh)": nn.RNN, "LSTM": nn.LSTM, "GRU": nn.GRU}[kind]
        self.rnn = cls(32, hidden, batch_first=True)
        self.head = nn.Linear(hidden, VOCAB)
        if kind == "LSTM":  # forget-gate bias = 1: start out remembering (Jozefowicz et al. 2015)
            for name, p in self.rnn.named_parameters():
                if "bias_hh" in name:
                    p.data[hidden:2 * hidden] = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(self.emb(x))
        return self.head(out[:, -1])         # predict from the LAST hidden state only


def grad_to_first_step(model: SeqClassifier, length: int, dev: str) -> float:
    """||dL/d(embedding of token 0)|| relative to ||dL/d(embedding of the last token)||, at init."""
    x, y = batch(256, length, dev)
    e = model.emb(x).detach().requires_grad_(True)
    out, _ = model.rnn(e)
    F.cross_entropy(model.head(out[:, -1]), y).backward()
    g = e.grad.norm(dim=(0, 2))
    return (g[0] / g[-1]).item()


def main() -> None:
    args = parse_args(__doc__)
    dev = device()
    kinds = ["RNN (tanh)", "LSTM", "GRU"]

    section("1. Gradient reaching the first token, relative to the last token (at init)")
    rows = []
    for length in [5, 20, 50, 100]:
        row = {"sequence length": length}
        for kind in kinds:
            torch.manual_seed(args.seed)
            row[kind] = grad_to_first_step(SeqClassifier(kind).to(dev), length, dev)
        rows.append(row)
    table(rows)
    kv("bptt-grad", rnn_len50=rows[2]["RNN (tanh)"], lstm_len50=rows[2]["LSTM"])
    print("  A vanilla RNN's signal from step 0 is ~1e-6 of the last step's at length 20; the LSTM's additive cell")
    print("  path (with forget bias 1) keeps it orders of magnitude larger, though it still decays with length.")

    section("2. Can it remember the first token? (accuracy, chance = 1/8)")
    steps = 1500 if args.quick else 4000
    lengths = [5, 20, 30] if args.quick else [5, 20, 30, 50]
    rows = []
    for length in lengths:
        row = {"sequence length": length}
        for kind in kinds:
            torch.manual_seed(args.seed)
            m = SeqClassifier(kind).to(dev)
            opt = torch.optim.Adam(m.parameters(), lr=3e-3)
            for _ in range(steps):
                x, y = batch(128, length, dev)
                loss = F.cross_entropy(m(x), y)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opt.step()
            with torch.no_grad():
                x, y = batch(2000, length, dev)
                row[kind] = (m(x).argmax(1) == y).float().mean().item()
        rows.append(row)
        print(f"  length {length}: " + "  ".join(f"{k}={row[k]:.3f}" for k in kinds), flush=True)
    table(rows)
    mid = rows[1]
    kv("memory", length=mid["sequence length"], rnn=mid["RNN (tanh)"], lstm=mid["LSTM"], gru=mid["GRU"])
    print(f"  Measured with --quick ({steps} steps): all cells solve length 5; at 20 only the gated cells do; at 30")
    print("  all fail in this budget. Gating extends the reachable range, it doesn't remove the limit: longer gaps")
    print("  need more steps, a curriculum (start short), or attention, which reaches step 0 in one hop.")

    section("3. Parameter counts (input 32, hidden 64)")
    table([{"cell": k, "params": sum(p.numel() for p in SeqClassifier(k).rnn.parameters()),
            "x RNN": sum(p.numel() for p in SeqClassifier(k).rnn.parameters()) / sum(p.numel() for p in SeqClassifier(kinds[0]).rnn.parameters())}
           for k in kinds])
    print("  LSTM = 4 gate blocks (i, f, g, o), GRU = 3 (r, z, n): 4x and 3x a plain RNN cell.")


if __name__ == "__main__":
    main()
