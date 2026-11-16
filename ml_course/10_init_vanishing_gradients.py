"""
10 - Initialisation, vanishing/exploding gradients, residual connections
========================================================================

Interview topics
----------------
- In a deep net every layer multiplies the signal (forward) and the
  gradient (backward) by roughly (weight scale x activation slope). Above
  1 per layer -> explodes; below 1 -> vanishes, exponentially in depth.
- Sigmoid: max slope 0.25 and saturates -> gradients shrink by >= 4x per
  layer; outputs are not zero-centred. tanh is better (slope 1 at 0) but
  saturates too. ReLU: slope 1 for positive inputs (no saturation), but
  "dead ReLUs" if a unit is always negative -> LeakyReLU/GELU/SiLU.
- Init keeps the variance of activations constant across layers:
    Xavier/Glorot: Var(W) = 2/(fan_in + fan_out)  (tanh / linear)
    He/Kaiming:    Var(W) = 2/fan_in              (ReLU: it zeroes half the
                   inputs, so it needs 2x the variance)
  Too small -> activations shrink to 0 layer by layer; too big -> blow up /
  saturate. (Section 1 measures it.)
- Residual connections (ResNet): y = x + F(x). The Jacobian is I + dF/dx,
  so the gradient has an identity path straight to early layers -> very deep
  nets train. Plus normalisation (lesson 11). Transformers = residual stream.
- Exploding gradients (RNNs, high lr, bad init) -> gradient clipping by
  global norm: if ||g|| > c, g <- g * c/||g||. Keeps the direction.
- Symmetry: initialising all weights to the same value means every unit in
  a layer gets the same gradient forever -> random init breaks symmetry.

Run:  python ml_course/10_init_vanishing_gradients.py [--quick] [--depth 100]
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import device, kv, parse_args, section, spirals, table, train_test

ACTS = {"sigmoid": torch.sigmoid, "tanh": torch.tanh, "relu": F.relu}


class Deep(nn.Module):
    def __init__(self, depth: int, width: int, act: str, init: str, residual: bool = False, norm: bool = False) -> None:
        super().__init__()
        self.inp = nn.Linear(2, width)
        self.layers = nn.ModuleList(nn.Linear(width, width) for _ in range(depth))
        self.norms = nn.ModuleList(nn.LayerNorm(width) for _ in range(depth)) if norm else None
        self.out = nn.Linear(width, 3)
        self.act, self.residual = ACTS[act], residual
        for lin in self.layers:
            fan_in = lin.in_features
            std = {"small (0.01)": 0.01, "large (1.0)": 1.0, "xavier": math.sqrt(2 / (2 * fan_in)),
                   "he": math.sqrt(2 / fan_in)}[init]
            nn.init.normal_(lin.weight, 0, std)
            nn.init.zeros_(lin.bias)
            if residual:  # common trick: scale residual branches down so the sum doesn't grow with depth
                lin.weight.data /= math.sqrt(depth)

    def forward(self, x: torch.Tensor, record: list | None = None) -> torch.Tensor:
        h = self.inp(x)
        for i, lin in enumerate(self.layers):
            z = self.norms[i](h) if self.norms is not None else h
            f = self.act(lin(z))
            h = h + f if self.residual else f
            if record is not None:
                h.retain_grad()
                record.append(h)
        return self.out(h)


def probe(model: Deep, x: torch.Tensor, y: torch.Tensor) -> tuple[list[float], list[float]]:
    """Activation std and gradient norm (dL/dh) at every layer, at init."""
    record: list[torch.Tensor] = []
    loss = F.cross_entropy(model(x, record), y)
    loss.backward()
    return [h.detach().std().item() for h in record], [h.grad.norm().item() for h in record]


def main() -> None:
    args = parse_args(__doc__, [("--depth", {"type": int, "default": 20, "help": "hidden layers"})])
    dev = device()
    x, y = spirals(n_per_class=300, classes=3, noise=0.2, seed=args.seed)
    xtr, xte, ytr, yte = train_test(x, y, test=0.3, seed=args.seed)
    xtr, xte = torch.tensor(xtr, device=dev), torch.tensor(xte, device=dev)
    ytr, yte = torch.tensor(ytr, device=dev), torch.tensor(yte, device=dev)
    depth, width = args.depth, 128

    section(f"1. Activation std through {depth} layers, by activation x init")
    rows = []
    for act, init in [("tanh", "small (0.01)"), ("tanh", "large (1.0)"), ("tanh", "xavier"),
                      ("relu", "xavier"), ("relu", "he"), ("sigmoid", "xavier")]:
        torch.manual_seed(args.seed)
        stds, grads = probe(Deep(depth, width, act, init).to(dev), xtr, ytr)
        rows.append({"act": act, "init": init, "std L1": stds[0], "std L5": stds[4], "std L10": stds[9],
                     "std L20": stds[-1], "grad L1 / grad L20": grads[0] / max(grads[-1], 1e-30)})
    table(rows)
    kv("act-std", tanh_small_L20=rows[0]["std L20"], relu_xavier_L20=rows[3]["std L20"], relu_he_L20=rows[4]["std L20"])
    print("  ReLU + Xavier halves the variance each layer (0.5^20 ~ 1e-6 of the variance); He keeps it ~constant.")
    print("  tanh with std 1.0 saturates at +-1 and is in the 'chaotic' regime: the gradient GROWS going backward")
    print("  (last column >> 1). Small init: everything shrinks to 0 (forward and backward).")
    print("  Sigmoid keeps a healthy-looking std (~0.11 around a 0.5 mean) yet its gradient dies: slope <= 0.25.")

    section("2. Gradient size per layer: sigmoid vs relu+He vs residual")
    rows = []
    configs = [("plain sigmoid (xavier)", dict(act="sigmoid", init="xavier")),
               ("plain relu (he)", dict(act="relu", init="he")),
               ("residual + layernorm, relu (he)", dict(act="relu", init="he", residual=True, norm=True))]
    for name, kw in configs:
        torch.manual_seed(args.seed)
        _, grads = probe(Deep(depth, width, **kw).to(dev), xtr, ytr)
        rows.append({"model": name, "||dL/dh|| layer 1": grads[0], "layer 10": grads[9], "layer 20": grads[-1],
                     "first / last": grads[0] / max(grads[-1], 1e-30)})
    table(rows)
    kv("grad-flow", sigmoid_ratio=rows[0]["first / last"], residual_ratio=rows[2]["first / last"])

    section(f"3. Does it train? {depth} hidden layers, same optimizer and steps")
    steps = 600 if args.quick else 2000
    rows = []
    for name, kw in configs + [("plain tanh, small init (0.01)", dict(act="tanh", init="small (0.01)"))]:
        torch.manual_seed(args.seed)
        m = Deep(depth, width, **kw).to(dev)
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        first = None
        for step in range(steps):
            idx = torch.randint(0, len(xtr), (128,), device=dev)
            loss = F.cross_entropy(m(xtr[idx]), ytr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            first = first if first is not None else loss.item()
        with torch.no_grad():
            acc = float((m(xte).argmax(1) == yte).float().mean())
            final = float(F.cross_entropy(m(xtr), ytr))
        rows.append({"model": name, "loss start": first, "train loss end": final, "test acc": acc})
    table(rows)
    kv("deep-train", sigmoid_acc=rows[0]["test acc"], residual_acc=rows[2]["test acc"], small_tanh_acc=rows[3]["test acc"])
    print("  chance = 0.33. Even Adam (which rescales tiny gradients) can't rescue 1e-12 gradients.")
    print("  At 20 layers ReLU + He is enough. With --depth 100 plain ReLU + He also stays at chance, while")
    print("  residual + LayerNorm still reaches ~1.0 (measured): that is the ResNet / transformer recipe.")

    section("4. Exploding gradients and clipping by global norm")
    torch.manual_seed(args.seed)
    m = Deep(depth, width, "relu", "large (1.0)").to(dev)  # far too large for relu: x ~sqrt(128/2)=8 per layer
    loss = F.cross_entropy(m(xtr), ytr)
    loss.backward()
    total = float(torch.nn.utils.clip_grad_norm_(m.parameters(), max_norm=1.0))  # returns the PRE-clip norm
    after = float(torch.sqrt(sum((p.grad ** 2).sum() for p in m.parameters())))
    kv("clip", loss_at_init=float(loss), grad_norm_before=total, grad_norm_after=after)
    print("  Clipping keeps the step bounded so one bad batch can't throw the weights far away. It doesn't fix the")
    print("  init: this net is still useless. Clip as a safety net (LLM training uses max_norm=1.0 by default).")

    section("5. Symmetry: constant init never breaks it")
    m = nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 3)).to(dev)
    for p in m.parameters():
        nn.init.constant_(p, 0.1)
    opt = torch.optim.SGD(m.parameters(), lr=0.5)
    for _ in range(200):
        loss = F.cross_entropy(m(xtr), ytr)
        opt.zero_grad()
        loss.backward()
        opt.step()
    w = m[0].weight.detach()
    distinct = int(torch.unique(w.round(decimals=6), dim=0).shape[0])
    kv("symmetry", distinct_hidden_units=distinct, of_units=16)
    print("  All 16 hidden units received identical gradients every step: the layer is effectively width 1.")


if __name__ == "__main__":
    main()
