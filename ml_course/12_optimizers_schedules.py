"""
12 - Optimizers and learning-rate schedules in practice (torch)
===============================================================

Interview topics
----------------
- The learning rate is the most important hyperparameter. Find a range with
  an LR range test (Smith 2015): increase lr exponentially over ~100 steps,
  record the loss, pick ~1/10 of the lr where the loss is lowest / starts
  to blow up (section 1).
- SGD (+ momentum) vs Adam: Adam converges faster with little tuning and
  handles very different gradient scales across parameters (embeddings,
  attention, norms); well-tuned SGD+momentum can generalise as well or
  better on vision CNNs. LLMs/transformers: AdamW almost always.
- Adam + L2 vs AdamW: adding lambda*w to the gradient (L2) gets divided by
  Adam's sqrt(v) like everything else, so parameters with large gradients
  are barely regularised. AdamW (Loshchilov & Hutter) DECOUPLES decay:
  w <- w - lr * (adam_step + lambda * w) -> uniform shrinkage (section 3).
  Usually don't decay biases and norm gains.
- Warmup: Adam's second-moment estimate is noisy in the first steps and
  early gradients are large -> ramp lr up linearly over the first ~1-5% of
  steps. Then decay: cosine to ~10% of max, or linear, or
  warmup-stable-decay (WSD: constant, then a short final decay; lets you
  stop/extend runs). Decaying at the end removes the SGD noise floor (02).
- Batch size: bigger batch = less gradient noise, fewer steps per epoch.
  Linear scaling rule (Goyal et al.): multiply lr by k when batch grows k x
  (SGD; for Adam ~sqrt(k)), with warmup; breaks down past the critical
  batch size.
- Other names worth knowing: Nesterov, AdaGrad (sums squared grads -> lr
  decays to 0; fine for sparse features), RMSProp, LAMB/LARS (layer-wise
  trust ratio for huge batches), Lion (sign of momentum), Muon
  (orthogonalised momentum for matrix params, used in recent LLM training),
  Schedule-Free AdamW, 8-bit Adam (memory).

Run:  python ml_course/12_optimizers_schedules.py [--quick]
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits

from common import device, kv, parse_args, section, table, train_test


def make_net() -> nn.Sequential:
    return nn.Sequential(nn.Linear(64, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 10))


def main() -> None:
    args = parse_args(__doc__)
    dev = device()
    x, y = load_digits(return_X_y=True)
    x = (x / 16.0).astype(np.float32)
    xtr, xte, ytr, yte = train_test(x, y, test=0.3, seed=args.seed)
    xtr, xte = torch.tensor(xtr, device=dev), torch.tensor(xte, device=dev)
    ytr, yte = torch.tensor(ytr, device=dev), torch.tensor(yte, device=dev)
    steps = 600 if args.quick else 2000

    def run(opt_fn, sched_fn=None, bs: int = 64, n_steps: int = steps, seed: int = args.seed) -> dict:
        torch.manual_seed(seed)
        net = make_net().to(dev)
        opt = opt_fn(net.parameters())
        sched = sched_fn(opt) if sched_fn else None
        losses = []
        for _ in range(n_steps):
            idx = torch.randint(0, len(xtr), (bs,), device=dev)
            loss = F.cross_entropy(net(xtr[idx]), ytr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            if sched:
                sched.step()
            losses.append(loss.item())
        with torch.no_grad():
            return {"net": net, "train loss (last 50)": float(np.mean(losses[-50:])),
                    "test loss": F.cross_entropy(net(xte), yte).item(),
                    "test acc": (net(xte).argmax(1) == yte).float().mean().item()}

    section("1. LR range test (Adam): sweep lr exponentially, watch the loss")
    torch.manual_seed(args.seed)
    net = make_net().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-6)
    lrs = np.geomspace(1e-6, 10, 150)
    hist = []
    for lr in lrs:
        for g in opt.param_groups:
            g["lr"] = float(lr)
        idx = torch.randint(0, len(xtr), (64,), device=dev)
        loss = F.cross_entropy(net(xtr[idx]), ytr[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        hist.append(loss.item())
        if not math.isfinite(hist[-1]) or hist[-1] > 4 * hist[0]:
            break
    smooth = np.convolve(hist, np.ones(5) / 5, mode="valid")
    i_min = int(np.argmin(smooth)) + 2
    for lr, loss_at in list(zip(lrs, hist))[::15]:
        print(f"  lr {lr:9.2e}  loss {loss_at:.3f}")
    print(f"  ... stopped at lr {lrs[len(hist) - 1]:.2e} (loss {hist[-1]:.2f})")
    suggested = float(lrs[i_min] / 10)
    kv("lr-range", lr_at_min_loss=float(lrs[i_min]), blowup_lr=float(lrs[len(hist) - 1]), suggested=suggested)

    section("2. Same budget, different optimizers")
    rows = []
    configs = [("SGD lr=0.01", lambda p: torch.optim.SGD(p, lr=0.01)),
               ("SGD+momentum 0.9 lr=0.01", lambda p: torch.optim.SGD(p, lr=0.01, momentum=0.9)),
               ("SGD+Nesterov lr=0.05", lambda p: torch.optim.SGD(p, lr=0.05, momentum=0.9, nesterov=True)),
               ("Adam lr=1e-3", lambda p: torch.optim.Adam(p, lr=1e-3)),
               (f"Adam lr={suggested:.1e} (range test)", lambda p: torch.optim.Adam(p, lr=suggested)),
               ("Adam lr=1e-1 (too high)", lambda p: torch.optim.Adam(p, lr=1e-1))]
    for name, fn in configs:
        r = run(fn)
        rows.append({"optimizer": name, **{k: v for k, v in r.items() if k != "net"}})
    table(rows)
    kv("optimizers", sgd_acc=rows[0]["test acc"], momentum_acc=rows[1]["test acc"], adam_acc=rows[3]["test acc"],
       adam_too_high_acc=rows[5]["test acc"])

    section("3. Adam + L2 vs AdamW (decoupled weight decay)")
    rows = []
    for name, fn in [("Adam, no decay", lambda p: torch.optim.Adam(p, lr=1e-3)),
                     ("Adam + L2 (weight_decay=1e-2)", lambda p: torch.optim.Adam(p, lr=1e-3, weight_decay=1e-2)),
                     ("AdamW (weight_decay=1e-2)", lambda p: torch.optim.AdamW(p, lr=1e-3, weight_decay=1e-2)),
                     ("AdamW (weight_decay=1.0)", lambda p: torch.optim.AdamW(p, lr=1e-3, weight_decay=1.0))]:
        r = run(fn)
        w = r["net"][0].weight.detach()
        # Input pixels that are always 0 at the image border get ZERO gradient: only explicit decay shrinks them.
        dead = (xtr.std(0) == 0)
        rows.append({"optimizer": name, "test acc": r["test acc"], "||W1||": w.norm().item(),
                     "|W1| on always-0 pixels": w[:, dead].abs().mean().item(),
                     "|W1| on used pixels": w[:, ~dead].abs().mean().item()})
    table(rows)
    for r in rows[1:]:
        r_dead = r["|W1| on always-0 pixels"] / rows[0]["|W1| on always-0 pixels"]
        r_used = r["|W1| on used pixels"] / rows[0]["|W1| on used pixels"]
        print(f"  {r['optimizer']:<30} kept {r_dead:9.2e} of the never-used weights, {r_used:.2f} of the used ones")
    kv("adamw", l2_dead_kept=rows[1]["|W1| on always-0 pixels"] / rows[0]["|W1| on always-0 pixels"],
       adamw1_dead_kept=rows[3]["|W1| on always-0 pixels"] / rows[0]["|W1| on always-0 pixels"],
       adamw1_used_kept=rows[3]["|W1| on used pixels"] / rows[0]["|W1| on used pixels"])
    print("  Adam + L2: the decay term goes through Adam's 1/sqrt(v) normalisation. For a weight whose only")
    print("  gradient IS the decay, that makes a full-size step (~lr) toward 0 every time -> wiped out; weights")
    print("  with real gradients are shrunk much less. Decay strength depends on gradient history.")
    print("  AdamW: every weight shrinks by the same factor (1 - lr*wd) per step, independent of its gradients.")
    print("  Note the scale: wd=1e-2 with lr=1e-3 is only ~0.6% total shrinkage over 600 steps (no effect).")

    section("4. Schedules: constant vs warmup + cosine vs step decay")
    rows = []
    warm = max(1, steps // 20)

    def warmup_cosine(opt: torch.optim.Optimizer) -> torch.optim.lr_scheduler.LambdaLR:
        def f(t: int) -> float:
            if t < warm:
                return (t + 1) / warm
            prog = (t - warm) / max(1, steps - warm)
            return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog))   # cosine down to 10% of max
        return torch.optim.lr_scheduler.LambdaLR(opt, f)

    for name, sched in [("constant", None), ("warmup + cosine (to 10%)", warmup_cosine),
                        ("step decay x0.1 at 50%, 75%", lambda o: torch.optim.lr_scheduler.MultiStepLR(o, [steps // 2, 3 * steps // 4], 0.1))]:
        for lr in [3e-3, 3e-2]:
            r = run(lambda p, lr=lr: torch.optim.Adam(p, lr=lr), sched)
            rows.append({"schedule": name, "peak lr": lr, **{k: v for k, v in r.items() if k != "net"}})
    table(rows)
    const = [r for r in rows if r["schedule"] == "constant" and r["peak lr"] == 3e-2][0]
    cos = [r for r in rows if r["schedule"].startswith("warmup") and r["peak lr"] == 3e-2][0]
    kv("schedule", const_train_loss=const["train loss (last 50)"], cosine_train_loss=cos["train loss (last 50)"],
       const_acc=const["test acc"], cosine_acc=cos["test acc"])
    print("  At a low peak lr all three are within noise of each other. At a high peak lr the constant schedule")
    print("  ends bouncing around (noise floor, lesson 02); decaying lets it settle and keeps the fast start.")

    section("5. Batch size and the linear scaling rule (SGD+momentum, fixed number of EPOCHS)")
    rows = []
    epochs_samples = steps * 64
    for bs in [16, 64, 256]:
        for lr_rule in ["fixed lr 0.02", "linear scaling"]:
            lr = 0.02 if lr_rule.startswith("fixed") else 0.02 * bs / 64
            r = run(lambda p, lr=lr: torch.optim.SGD(p, lr=lr, momentum=0.9), bs=bs, n_steps=epochs_samples // bs)
            rows.append({"batch": bs, "lr rule": lr_rule, "lr": lr, "steps": epochs_samples // bs,
                         "test acc": r["test acc"], "test loss": r["test loss"]})
    table(rows)
    kv("batch-scaling", bs256_fixed=rows[4]["test loss"], bs256_scaled=rows[5]["test loss"])
    print("  Same number of samples seen. Big batches take fewer steps, so with a fixed lr they under-train;")
    print("  scaling lr with batch size recovers most of it (until the batch gets too large).")


if __name__ == "__main__":
    main()
