"""
11 - BatchNorm vs LayerNorm vs RMSNorm (from scratch, parity with torch)
========================================================================

Interview topics
----------------
- BatchNorm: normalise each FEATURE over the BATCH (and spatial dims for
  conv): x_hat = (x - mean_B) / sqrt(var_B + eps); y = gamma x_hat + beta.
  Training uses batch statistics and updates running averages
  (momentum 0.1); eval uses the running averages. Consequences:
    * a sample's output depends on its batch-mates in train mode;
    * small batches -> noisy statistics -> worse training (batch 2 is bad);
    * train/eval discrepancy; forgetting model.eval() changes predictions;
    * awkward for variable-length sequences and autoregressive decoding.
- LayerNorm: normalise each SAMPLE over its FEATURES. No batch dependence,
  same computation in train and eval, works for any batch size and per
  token -> the transformer default.
- RMSNorm: LayerNorm without mean-centring and without beta:
  y = x / sqrt(mean(x^2) + eps) * gamma. Cheaper, works as well in practice;
  used by Llama, Qwen, Mistral...
- Why normalisation helps: keeps activations in a good range at every
  depth, allows larger learning rates, smoother loss landscape (the original
  "internal covariate shift" explanation is disputed; Santurkar et al. 2018).
- Pre-LN vs post-LN transformers: pre-LN (normalise inside the residual
  branch, before attention/MLP) keeps a clean identity path and trains
  stably without warmup tricks; post-LN (original 2017) needs careful warmup.
- GroupNorm: groups of channels per sample; the vision fix for small batches.

Run:  python ml_course/11_normalization.py [--quick]
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import device, kv, parse_args, section, spirals, table, train_test


class MyBatchNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5, momentum: float = 0.1) -> None:
        super().__init__()
        self.gamma, self.beta = nn.Parameter(torch.ones(d)), nn.Parameter(torch.zeros(d))
        self.register_buffer("running_mean", torch.zeros(d))
        self.register_buffer("running_var", torch.ones(d))
        self.eps, self.momentum = eps, momentum

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            mean, var = x.mean(0), x.var(0, unbiased=False)           # normalise with the BIASED variance
            with torch.no_grad():                                       # ...but track the UNBIASED one (as torch does)
                n = x.shape[0]
                self.running_mean.lerp_(mean, self.momentum)
                self.running_var.lerp_(var * n / (n - 1), self.momentum)
        else:
            mean, var = self.running_mean, self.running_var
        return self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta


class MyLayerNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.gamma, self.beta, self.eps = nn.Parameter(torch.ones(d)), nn.Parameter(torch.zeros(d)), eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean, var = x.mean(-1, keepdim=True), x.var(-1, unbiased=False, keepdim=True)
        return self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta


class MyRMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.gamma, self.eps = nn.Parameter(torch.ones(d)), eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gamma * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


def make_norm(norm: str, d: int) -> nn.Module:
    return {"none": nn.Identity(), "batchnorm": nn.BatchNorm1d(d), "layernorm": nn.LayerNorm(d), "rmsnorm": nn.RMSNorm(d)}[norm]


class PreNormResMLP(nn.Module):
    """h = h + FF(Norm(h)) blocks, final norm, head: the transformer block layout without attention."""

    def __init__(self, norm: str, depth: int = 12, width: int = 128) -> None:
        super().__init__()
        self.embed = nn.Linear(2, width)
        self.norms = nn.ModuleList(make_norm(norm, width) for _ in range(depth))
        self.ffs = nn.ModuleList(nn.Sequential(nn.Linear(width, width), nn.ReLU(), nn.Linear(width, width)) for _ in range(depth))
        self.final, self.head = make_norm(norm, width), nn.Linear(width, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(self.embed(x))
        for norm, ff in zip(self.norms, self.ffs):
            h = h + ff(norm(h))
        return self.head(self.final(h))


def deep_net(norm: str, depth: int = 12, width: int = 128) -> nn.Sequential:
    # tanh after the input layer: without it the first norm sees LN(A x), which ignores the scale of x per
    # sample, so the spiral's radius (its main signal) would be erased. Section 6 shows this.
    layers: list[nn.Module] = [nn.Linear(2, width), nn.Tanh()]
    for _ in range(depth):
        layers.append(nn.Linear(width, width))
        if norm == "batchnorm":
            layers.append(nn.BatchNorm1d(width))
        elif norm == "layernorm":
            layers.append(nn.LayerNorm(width))
        elif norm == "rmsnorm":
            layers.append(nn.RMSNorm(width))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(width, 3))
    net = nn.Sequential(*layers)
    for m in net:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.05)   # deliberately poor init (not He) so normalisation has work to do
    return net


def main() -> None:
    args = parse_args(__doc__)
    dev = device()
    torch.manual_seed(args.seed)

    section("1. Parity with torch: BatchNorm (train + eval), LayerNorm, RMSNorm")
    x = torch.randn(32, 16, device=dev) * 3 + 1
    mine, ref = MyBatchNorm(16).to(dev), nn.BatchNorm1d(16).to(dev)
    for _ in range(5):                         # a few training steps so running stats move
        xb = torch.randn(32, 16, device=dev) * 3 + 1
        d_train = (mine(xb) - ref(xb)).abs().max().item()
    mine.eval(), ref.eval()
    d_eval = (mine(x) - ref(x)).abs().max().item()
    d_rm = (mine.running_var - ref.running_var).abs().max().item()
    ln_d = (MyLayerNorm(16).to(dev)(x) - nn.LayerNorm(16).to(dev)(x)).abs().max().item()
    rms_d = (MyRMSNorm(16).to(dev)(x) - nn.RMSNorm(16, eps=1e-6).to(dev)(x)).abs().max().item()
    kv("parity", bn_train=d_train, bn_eval=d_eval, bn_running_var=d_rm, layernorm=ln_d, rmsnorm=rms_d)

    section("2. BatchNorm output depends on the OTHER samples in the batch")
    bn, ln = nn.BatchNorm1d(16).to(dev).train(), nn.LayerNorm(16).to(dev)
    sample = torch.randn(1, 16, device=dev)
    mates_a, mates_b = torch.randn(7, 16, device=dev), torch.randn(7, 16, device=dev) * 5 + 3
    bn_diff = (bn(torch.cat([sample, mates_a]))[0] - bn(torch.cat([sample, mates_b]))[0]).abs().max().item()
    ln_diff = (ln(torch.cat([sample, mates_a]))[0] - ln(torch.cat([sample, mates_b]))[0]).abs().max().item()
    kv("batch-dependence", batchnorm=bn_diff, layernorm=ln_diff)
    print("  Same input row, different batch-mates: BatchNorm gives a different output; LayerNorm doesn't.")
    bn.eval()
    try:
        bn(sample)
        eval_ok = True
    except ValueError:
        eval_ok = False
    try:
        bn.train()(sample)
        train_ok = True
    except ValueError as e:
        train_ok = False
        print(f"  BatchNorm in train mode with batch size 1: ValueError ('{str(e)[:60]}...')")
    kv("batch1", bn_eval_ok=eval_ok, bn_train_ok=train_ok)

    xs, ys = spirals(n_per_class=300, classes=3, noise=0.2, seed=args.seed)
    xtr, xte, ytr, yte = train_test(xs, ys, test=0.3, seed=args.seed)
    xtr, xte = torch.tensor(xtr, device=dev), torch.tensor(xte, device=dev)
    ytr, yte = torch.tensor(ytr, device=dev), torch.tensor(yte, device=dev)
    scale = 1 if args.quick else 3

    def fit(net: nn.Module, lr: float, bs: int, steps: int, adam: bool = False) -> float | str:
        opt = torch.optim.Adam(net.parameters(), lr=lr) if adam else torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
        net.train()
        for _ in range(steps):
            idx = torch.randint(0, len(xtr), (bs,), device=dev)
            loss = F.cross_entropy(net(xtr[idx]), ytr[idx])
            if not torch.isfinite(loss):
                return "diverged"
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            return (net(xte).argmax(1) == yte).float().mean().item()

    section("3. Plain 12-layer MLP, post-norm (Linear -> Norm -> ReLU), SGD: test acc by learning rate")
    rows = []
    for norm in ["none", "batchnorm", "layernorm", "rmsnorm"]:
        row = {"norm": norm}
        for lr in [1e-3, 1e-2, 1e-1]:
            torch.manual_seed(args.seed)
            row[f"lr {lr:g}"] = fit(deep_net(norm).to(dev), lr, 64, 500 * scale)
        rows.append(row)
    table(rows)
    kv("postnorm", none_best=max(v for k, v in rows[0].items() if k != "norm"), bn_lr01=rows[1]["lr 0.1"],
       ln_lr0001=rows[2]["lr 0.001"])
    print("  std-0.05 init, no residuals: without a norm the signal shrinks layer by layer and nothing learns at")
    print("  any lr. BatchNorm trains across a 100x lr range. LayerNorm/RMSNorm only at low lr in this layout")
    print("  (measured): per-sample norms in a plain post-norm stack are fragile. They shine in the next layout.")

    section("4. Pre-norm residual MLP (h = h + FF(Norm(h)), the transformer layout), SGD by learning rate")
    rows = []
    for norm in ["none", "batchnorm", "layernorm", "rmsnorm"]:
        row = {"norm": norm}
        for lr in [1e-3, 1e-2, 1e-1]:
            torch.manual_seed(args.seed)
            row[f"lr {lr:g}"] = fit(PreNormResMLP(norm).to(dev), lr, 64, 500 * scale)
        rows.append(row)
    table(rows)
    kv("prenorm", none_lr01=str(rows[0]["lr 0.1"]), ln_lr01=rows[2]["lr 0.1"], rms_lr01=rows[3]["lr 0.1"])
    print("  The residual path alone makes depth trainable (lesson 10); the norm is what keeps high lr stable.")

    section("5. Small batches hurt BatchNorm, not LayerNorm/RMSNorm (pre-norm residual, Adam 1e-3)")
    rows = []
    for bs, steps in [(2, 3000), (8, 2000), (64, 500)]:
        row = {"batch size": bs}
        for norm in ["batchnorm", "layernorm", "rmsnorm"]:
            torch.manual_seed(args.seed)
            row[f"{norm} acc"] = fit(PreNormResMLP(norm).to(dev), 1e-3, bs, steps * scale, adam=True)
        rows.append(row)
    table(rows)
    kv("small-batch", bn_bs2=rows[0]["batchnorm acc"], ln_bs2=rows[0]["layernorm acc"], rms_bs2=rows[0]["rmsnorm acc"])
    print("  With 2 samples, BatchNorm's mean/var are noise, and the running stats used at eval don't match")
    print("  what the net saw in training. Per-sample norms don't care about the batch at all.")

    section("6. A LayerNorm pitfall: per-sample normalisation erases input SCALE")
    lin = nn.Linear(2, 64, bias=False).to(dev)
    ln = nn.LayerNorm(64).to(dev)
    pts = torch.tensor([[0.1, 0.2], [0.3, 0.6], [1.0, 2.0]], device=dev)   # same direction, different radius
    out = ln(lin(pts))
    diff = (out[0] - out[2]).abs().max().item()
    kv("ln-scale-invariance", max_diff_r_vs_10r=diff)
    print("  LN(A x) == LN(A * 10x): a point and its 10x-farther twin look identical. If magnitude is the")
    print("  signal (a spiral's radius), put a non-linearity (or bias) before the first norm. deep_net() does.")


if __name__ == "__main__":
    main()
