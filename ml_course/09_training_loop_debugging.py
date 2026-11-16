"""
09 - The training loop, debugging habits, and regularising a neural net
=======================================================================

Interview topics
----------------
- The loop: for batch: model.train(); logits = model(x); loss = CE(logits, y);
  opt.zero_grad(); loss.backward(); (clip); opt.step(); (scheduler.step()).
  Evaluate with model.eval() + torch.no_grad().
- Debugging recipe (Karpathy, "A Recipe for Training Neural Networks"):
    1. Check the loss at init: with C balanced classes and a sane init it
       should be ~ln(C). Much higher = overconfident init (scale down the last
       layer). (Section 1.)
    2. Overfit ONE small batch to ~0 loss. If you can't, there is a bug
       (labels misaligned, wrong loss, frozen params, lr far off). (Section 2.)
    3. Then scale up, watch train AND val curves, and add regularisation
       only once you overfit.
- model.train() vs model.eval(): dropout and batch norm behave differently.
  Forgetting eval() at test time = noisy, worse predictions (section 4).
- Regularisers: weight decay (L2), dropout (randomly zero activations with
  prob p, scale the rest by 1/(1-p) = "inverted dropout", so eval needs no
  rescale; an implicit ensemble), data augmentation, early stopping (keep
  the checkpoint with the best val metric), smaller model, label smoothing.
- Early stopping is a regulariser: stop when val loss hasn't improved for
  `patience` epochs and RESTORE the best weights.
- Loss NaN: lr too high, log(0) / division by 0, fp16 overflow, bad data
  (inf/NaN inputs) -> check inputs, lower lr, clip grads, use bf16/fp32.

Run:  python ml_course/09_training_loop_debugging.py [--quick]
"""

from __future__ import annotations

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import device, kv, parse_args, section, spirals, table, train_test


def mlp(width: int = 128, depth: int = 3, dropout: float = 0.0, classes: int = 3) -> nn.Sequential:
    layers: list[nn.Module] = []
    d_in = 2
    for _ in range(depth):
        layers += [nn.Linear(d_in, width), nn.ReLU()]
        if dropout:
            layers.append(nn.Dropout(dropout))
        d_in = width
    layers.append(nn.Linear(d_in, classes))
    return nn.Sequential(*layers)


@torch.no_grad()
def evaluate(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> tuple[float, float]:
    model.eval()
    logits = model(x)
    return float(F.cross_entropy(logits, y)), float((logits.argmax(1) == y).float().mean())


def train(model: nn.Module, xtr, ytr, xva, yva, epochs: int, lr: float = 3e-3, wd: float = 0.0, bs: int = 64,
          patience: int | None = None) -> dict:
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    best = {"val_loss": math.inf, "epoch": -1, "state": None}
    hist = []
    n = len(xtr)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=xtr.device)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            loss = F.cross_entropy(model(xtr[idx]), ytr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        vl, va = evaluate(model, xva, yva)
        tl, ta = evaluate(model, xtr, ytr)
        hist.append((tl, ta, vl, va))
        if vl < best["val_loss"]:
            best = {"val_loss": vl, "epoch": ep, "state": copy.deepcopy(model.state_dict())}
        elif patience is not None and ep - best["epoch"] >= patience:
            break
    return {"hist": hist, "best": best, "stopped": len(hist)}


def main() -> None:
    args = parse_args(__doc__)
    dev = device()
    x, y = spirals(n_per_class=400, classes=3, noise=0.25, seed=args.seed)
    xtr, xte, ytr, yte = train_test(x, y, test=0.5, seed=args.seed)
    xtr, xva, ytr, yva = train_test(xtr, ytr, test=0.3, seed=args.seed)
    t = lambda a, dt=torch.float32: torch.tensor(a, dtype=dt, device=dev)  # noqa: E731
    xtr, xva, xte = t(xtr), t(xva), t(xte)
    ytr, yva, yte = t(ytr, torch.long), t(yva, torch.long), t(yte, torch.long)
    print(f"  device={dev}  train={len(xtr)} val={len(xva)} test={len(xte)}  (3 interleaved spirals)")

    section("1. Loss at init should be ~ln(C)")
    m = mlp().to(dev)
    init_loss, _ = evaluate(m, xtr, ytr)
    bad = mlp().to(dev)
    nn.init.normal_(bad[-1].weight, 0, 10.0)    # overconfident last layer (default std is ~0.05)
    bad_loss, _ = evaluate(bad, xtr, ytr)
    kv("init-loss", expected=math.log(3), default_init=init_loss, last_layer_std10=bad_loss)
    print("  A high starting loss means the first steps are spent un-learning confident nonsense (the 'hockey stick').")

    section("2. Overfit a single batch first")
    m = mlp().to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    xb, yb = xtr[:32], ytr[:32]
    for step in range(300):
        loss = F.cross_entropy(m(xb), yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
    kv("overfit-batch", final_loss=loss.item())
    # The same check catches a classic bug: labels shuffled out of alignment with inputs.
    m = mlp().to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    y_bug = yb[torch.randperm(32, device=dev)]
    for step in range(300):
        loss_bug = F.cross_entropy(m(xb), y_bug)
        opt.zero_grad()
        loss_bug.backward()
        opt.step()
    print(f"  with misaligned labels a big net can STILL memorise 32 points (loss {float(loss_bug):.4f}) -> the test")
    print("  rules out pipeline bugs only when it FAILS; follow with a check that val accuracy beats chance.")

    section("3. Regularisation in the small-data, noisy-label regime (big net, few points)")
    epochs = 400                                # cheap: 150 points = 3 batches per epoch
    small = slice(0, 150)                       # only 150 training points...
    g = torch.Generator(device="cpu").manual_seed(args.seed)
    flip = (torch.rand(150, generator=g) < 0.2).to(dev)                  # ...and 20% of their labels wrong
    y_small = torch.where(flip, (ytr[small] + 1) % 3, ytr[small])
    rows, runs = [], {}
    for name, kw, wd in [("no regularisation", {}, 0.0), ("dropout 0.3", {"dropout": 0.3}, 0.0),
                         ("weight decay 0.1", {}, 0.1), ("early stopping (patience 30)", {}, 0.0)]:
        torch.manual_seed(args.seed)
        model = mlp(width=256, **kw).to(dev)
        r = train(model, xtr[small], y_small, xva, yva, epochs, wd=wd,
                  patience=30 if "early" in name else None)
        final = evaluate(model, xte, yte)
        model.load_state_dict(r["best"]["state"])
        best = evaluate(model, xte, yte)
        tl, ta, vl, va = r["hist"][-1]
        rows.append({"setup": name, "epochs run": r["stopped"], "final train loss": tl, "final val loss": vl,
                     "test acc (final)": final[1], "test acc (best-val ckpt)": best[1]})
        runs[name] = r
    table(rows)
    kv("regularise", none_val_loss=rows[0]["final val loss"], dropout_val_loss=rows[1]["final val loss"],
       wd_val_loss=rows[2]["final val loss"], none_train_loss=rows[0]["final train loss"])
    h = runs["no regularisation"]["hist"]
    print(f"  no-reg val loss: best {min(v[2] for v in h):.3f} at epoch {int(np.argmin([v[2] for v in h]))}, "
          f"final {h[-1][2]:.3f}  <- the overfitting curve: val loss goes back UP while the net memorises")
    print("  the wrong labels. Every remedy helps; restoring the best-val checkpoint helps all of them.")
    print("  (AdamW's decay is lr * wd per step, so wd=0.1 at lr 3e-3 is mild; tune wd jointly with lr.)")

    section("4. Forgetting model.eval() with dropout")
    torch.manual_seed(args.seed)
    model = mlp(width=256, dropout=0.5).to(dev)
    train(model, xtr, ytr, xva, yva, 60 if args.quick else 150)
    _, acc_eval = evaluate(model, xte, yte)
    model.train()                                   # the bug: dropout stays on at test time
    with torch.no_grad():
        accs = [float((model(xte).argmax(1) == yte).float().mean()) for _ in range(5)]
    kv("eval-mode", acc_eval=acc_eval, acc_train_mode_mean=float(np.mean(accs)), acc_train_mode_spread=float(np.ptp(accs)))
    print("  train() mode: a different random subnetwork each call -> lower and non-deterministic accuracy.")

    section("5. Full data, mild regularisation: the end result")
    torch.manual_seed(args.seed)
    model = mlp(width=256, dropout=0.1).to(dev)
    r = train(model, xtr, ytr, xva, yva, 150 if args.quick else 400, wd=0.01, patience=40)
    model.load_state_dict(r["best"]["state"])
    _, acc = evaluate(model, xte, yte)
    kv("final", test_acc=acc, best_epoch=r["best"]["epoch"])


if __name__ == "__main__":
    main()
