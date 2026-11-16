"""
08 - Backpropagation and autograd from scratch
==============================================

Interview topics
----------------
- A network is a computation graph. Backprop = the chain rule applied in
  reverse topological order: each node gets dL/d(node) ("grad") and passes
  grad * local_derivative to its inputs; grads from several consumers ADD.
  Cost: one backward pass ~ 2x a forward pass, independent of #parameters
  (that's why reverse mode beats finite differences, which need one forward
  per parameter).
- Reverse mode (backprop) is efficient for many inputs -> one scalar loss;
  forward mode for few inputs -> many outputs.
- Memory: the forward pass must KEEP intermediate activations for the
  backward pass. That (not the weights) dominates training memory for big
  batches/sequences -> gradient checkpointing recomputes them instead.
- Matrix layer y = x W + b:  dW = x^T dy,  db = sum(dy),  dx = dy W^T.
  Softmax + cross-entropy combined: d logits = (softmax - onehot) / batch.
- Gradient checking: compare to central differences (f(w+h)-f(w-h))/2h in
  float64; relative error ~1e-7 or lower means correct.
- PyTorch specifics: .grad ACCUMULATES across backward() calls (hence
  optimizer.zero_grad()); torch.no_grad() / inference_mode() skip graph
  building for eval; .detach() cuts the graph (targets, EMA teachers,
  stop-gradient); backward() frees the graph unless retain_graph=True.

Run:  python ml_course/08_autograd_backprop.py [--quick]
"""

from __future__ import annotations

import math
import random

import numpy as np
import torch

from common import kv, parse_args, section, spirals


class Value:
    """A scalar that remembers how it was computed (micrograd-style, after Karpathy)."""

    def __init__(self, data: float, parents: tuple[Value, ...] = (), op: str = "") -> None:
        self.data, self.grad = float(data), 0.0
        self._parents, self._op = parents, op
        self._backward = lambda: None

    @staticmethod
    def wrap(x: Value | float) -> Value:
        return x if isinstance(x, Value) else Value(x)

    def __add__(self, other: Value | float) -> Value:
        other = Value.wrap(other)
        out = Value(self.data + other.data, (self, other), "+")
        def _backward() -> None:
            self.grad += out.grad            # d(a+b)/da = 1
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other: Value | float) -> Value:
        other = Value.wrap(other)
        out = Value(self.data * other.data, (self, other), "*")
        def _backward() -> None:
            self.grad += other.data * out.grad   # d(ab)/da = b
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, k: float) -> Value:
        out = Value(self.data ** k, (self,), f"**{k}")
        def _backward() -> None:
            self.grad += k * self.data ** (k - 1) * out.grad
        out._backward = _backward
        return out

    def tanh(self) -> Value:
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")
        def _backward() -> None:
            self.grad += (1 - t * t) * out.grad
        out._backward = _backward
        return out

    def exp(self) -> Value:
        e = math.exp(self.data)
        out = Value(e, (self,), "exp")
        def _backward() -> None:
            self.grad += e * out.grad
        out._backward = _backward
        return out

    def log(self) -> Value:
        out = Value(math.log(self.data), (self,), "log")
        def _backward() -> None:
            self.grad += out.grad / self.data
        out._backward = _backward
        return out

    __radd__ = __add__
    __rmul__ = __mul__
    def __neg__(self) -> Value: return self * -1
    def __sub__(self, o: Value | float) -> Value: return self + (-Value.wrap(o))
    def __rsub__(self, o: float) -> Value: return Value.wrap(o) + (-self)
    def __truediv__(self, o: Value | float) -> Value: return self * Value.wrap(o) ** -1

    def backward(self) -> None:
        order, seen = [], set()
        def build(v: Value) -> None:           # topological sort: parents before children
            if id(v) not in seen:
                seen.add(id(v))
                for p in v._parents:
                    build(p)
                order.append(v)
        build(self)
        self.grad = 1.0
        for v in reversed(order):
            v._backward()


def numpy_mlp_grads(params: dict[str, np.ndarray], x: np.ndarray, y: np.ndarray) -> tuple[float, dict[str, np.ndarray]]:
    """2-layer MLP, tanh, softmax CE. Forward and hand-written backward."""
    h_pre = x @ params["W1"] + params["b1"]
    h = np.tanh(h_pre)
    logits = h @ params["W2"] + params["b2"]
    z = logits - logits.max(1, keepdims=True)
    p = np.exp(z) / np.exp(z).sum(1, keepdims=True)
    n = len(y)
    loss = float(-np.log(p[np.arange(n), y]).mean())
    d_logits = p.copy()
    d_logits[np.arange(n), y] -= 1
    d_logits /= n                                   # softmax + CE: (p - onehot) / n
    grads = {"W2": h.T @ d_logits, "b2": d_logits.sum(0)}
    d_h = d_logits @ params["W2"].T
    d_hpre = d_h * (1 - h ** 2)                     # tanh' = 1 - tanh^2
    grads["W1"], grads["b1"] = x.T @ d_hpre, d_hpre.sum(0)
    return loss, grads


def main() -> None:
    args = parse_args(__doc__)

    section("1. Scalar autograd vs torch on one expression")
    a, b, c = Value(1.5), Value(-2.0), Value(0.7)
    out = (a * b + c).tanh() * a + (b ** 2) / c - (a * c).exp() * 0.1
    out.backward()
    ta, tb, tc = (torch.tensor(v, dtype=torch.float64, requires_grad=True) for v in (1.5, -2.0, 0.7))
    tout = torch.tanh(ta * tb + tc) * ta + tb ** 2 / tc - torch.exp(ta * tc) * 0.1
    tout.backward()
    diff = max(abs(a.grad - ta.grad.item()), abs(b.grad - tb.grad.item()), abs(c.grad - tc.grad.item()))
    print(f"  ours  grads: a={a.grad:.6f} b={b.grad:.6f} c={c.grad:.6f}")
    print(f"  torch grads: a={ta.grad.item():.6f} b={tb.grad.item():.6f} c={tc.grad.item():.6f}")
    kv("autograd", max_grad_diff=diff)
    print("  'a' is used twice; its grad is the SUM of both paths (the += in every _backward).")

    section("2. Train a tiny MLP with the scalar engine (2 -> 8 -> 1, tanh)")
    rng = random.Random(args.seed)
    x_np, y_np = spirals(n_per_class=40, classes=2, noise=0.1, seed=args.seed)
    data = [([float(v) for v in xi], 1.0 if yi else -1.0) for xi, yi in zip(x_np, y_np)]
    hidden = 8
    w1 = [[Value(rng.gauss(0, 1)) for _ in range(2)] for _ in range(hidden)]
    b1 = [Value(0.0) for _ in range(hidden)]
    w2 = [Value(rng.gauss(0, 1 / math.sqrt(hidden))) for _ in range(hidden)]
    b2 = Value(0.0)
    params = [p for row in w1 for p in row] + b1 + w2 + [b2]
    def forward(xi: list[float]) -> Value:
        h = [(sum((w * xv for w, xv in zip(row, xi)), Value(0.0)) + bb).tanh() for row, bb in zip(w1, b1)]
        return sum((w * hv for w, hv in zip(w2, h)), Value(0.0)) + b2
    steps = 60 if args.quick else 200
    for step in range(steps):
        # Hinge loss on +-1 labels (simple, and every op is one we defined)
        losses = [1 - yi * forward(xi) for xi, yi in data]
        loss = sum((lv if lv.data > 0 else lv * 0.0 for lv in losses), Value(0.0)) * (1 / len(data))
        for p in params:
            p.grad = 0.0                        # zero_grad: grads accumulate otherwise
        loss.backward()
        lr = 0.3 if step < steps // 2 else 0.1
        for p in params:
            p.data -= lr * p.grad
    acc = float(np.mean([(forward(xi).data > 0) == (yi > 0) for xi, yi in data]))
    kv("micrograd-mlp", params=len(params), final_loss=loss.data, train_acc=acc)

    section("3. Vectorised backprop in numpy + gradient check")
    x, y = spirals(n_per_class=50, classes=3, seed=args.seed)
    x = x.astype(np.float64)
    r = np.random.default_rng(args.seed)
    params = {"W1": r.normal(0, 1, (2, 16)), "b1": np.zeros(16), "W2": r.normal(0, 0.25, (16, 3)), "b2": np.zeros(3)}
    _, grads = numpy_mlp_grads(params, x, y)
    worst = 0.0
    h = 1e-6
    for name, w in params.items():
        num = np.zeros_like(w)
        for idx in np.ndindex(w.shape):
            old = w[idx]
            w[idx] = old + h
            lp, _ = numpy_mlp_grads(params, x, y)
            w[idx] = old - h
            lm, _ = numpy_mlp_grads(params, x, y)
            w[idx] = old
            num[idx] = (lp - lm) / (2 * h)
        rel = np.abs(num - grads[name]).max() / max(np.abs(num).max(), np.abs(grads[name]).max(), 1e-12)
        worst = max(worst, float(rel))
        print(f"  {name}: relative error {rel:.2e}")
    kv("gradcheck", worst_rel_err=worst)
    # Compare with torch autograd on the same weights.
    tp = {k: torch.tensor(v, requires_grad=True) for k, v in params.items()}
    tl = torch.nn.functional.cross_entropy(torch.tanh(torch.tensor(x) @ tp["W1"] + tp["b1"]) @ tp["W2"] + tp["b2"], torch.tensor(y))
    tl.backward()
    kv("vs-torch", max_diff=max(float(np.abs(tp[k].grad.numpy() - grads[k]).max()) for k in params))

    section("4. PyTorch gotchas: grads accumulate, no_grad, detach")
    w = torch.tensor([1.0, 2.0], requires_grad=True)
    for _ in range(3):
        (w ** 2).sum().backward()               # no zero_grad in between
    print(f"  after 3 backward() calls without zero_grad: w.grad = {w.grad.tolist()} (true grad is [2, 4])")
    acc_ratio = float(w.grad[0] / 2.0)
    with torch.no_grad():
        y_ng = w * 3
    z = (w * w.detach()).sum()                  # treat the second factor as a constant
    w.grad = None
    z.backward()
    print(f"  under no_grad: requires_grad={y_ng.requires_grad};  d/dw sum(w * w.detach()) = {w.grad.tolist()}"
          " (= w, not 2w)")
    kv("torch-gotchas", accumulated_x=acc_ratio, no_grad_tracks=bool(y_ng.requires_grad), detach_grad0=float(w.grad[0]))

    section("5. Memory: activations, not weights, dominate training at large batch")
    if torch.cuda.is_available():
        dev = "cuda"
        model = torch.nn.Sequential(*[m for _ in range(8) for m in (torch.nn.Linear(1024, 1024), torch.nn.GELU())]).to(dev)
        weights_mb = sum(p.numel() * 4 for p in model.parameters()) / 2**20
        for bs in [64, 4096]:
            xb = torch.randn(bs, 1024, device=dev)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            out = model(xb).sum()
            act_mb = (torch.cuda.memory_allocated() - base) / 2**20
            out.backward()
            print(f"  batch {bs:>5}: weights {weights_mb:.0f} MB, saved activations after forward {act_mb:.0f} MB")
            model.zero_grad(set_to_none=True)
            del out, xb
        kv("activation-memory", weights_mb=weights_mb, activations_bs4096_mb=act_mb)
        del model
        torch.cuda.empty_cache()
    else:
        print("  (no GPU: skipped)")


if __name__ == "__main__":
    main()
