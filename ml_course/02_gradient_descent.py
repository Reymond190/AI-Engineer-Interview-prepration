"""
02 - Gradient descent, learning rate, conditioning, optimizers (numpy)
======================================================================

Interview topics
----------------
- GD: w <- w - lr * grad. On a quadratic with curvature (Hessian eigenvalues)
  between mu and L, GD converges iff lr < 2/L; the best fixed lr ~ 2/(L+mu)
  and the number of steps scales with the condition number kappa = L/mu.
  Too-high lr diverges (loss -> inf/nan); too-low crawls.
- Feature scaling matters because unscaled features make kappa huge: the
  loss is a long thin valley and GD zig-zags across it. Standardising
  makes the bowl round (section 2). Trees don't care; GD-trained models,
  kNN, SVM, PCA and regularised models do.
- Momentum: velocity v <- beta v + grad; w <- w - lr v. Averages out the
  zig-zag and accelerates along the valley (~sqrt(kappa) instead of kappa
  steps, heavy-ball / Nesterov).
- RMSProp / Adam: divide each coordinate by a running RMS of its gradient,
  so every parameter gets a similar step size (per-parameter lr). Adam =
  momentum (1st moment) + RMSProp (2nd moment) + bias correction for the
  zero init of both moments. Default betas (0.9, 0.999), eps 1e-8.
- SGD: a minibatch gradient is an unbiased but noisy estimate. With a
  constant lr the loss stalls at a noise floor proportional to lr / batch;
  decaying the lr (or growing the batch) removes the floor (section 4).
  Noise is also a regulariser (tends to find flatter minima).
- Saddle points, not local minima, are the usual problem in high dimensions;
  momentum and noise help escape them.

Run:  python ml_course/02_gradient_descent.py [--quick]
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from common import kv, parse_args, section, table

Grad = Callable[[np.ndarray], np.ndarray]


def gd(grad: Grad, w0: np.ndarray, lr: float, steps: int) -> list[np.ndarray]:
    w, path = w0.copy(), [w0.copy()]
    for _ in range(steps):
        w = w - lr * grad(w)
        path.append(w.copy())
        if not np.isfinite(w).all() or np.abs(w).max() > 1e12:
            break
    return path


def momentum(grad: Grad, w0: np.ndarray, lr: float, steps: int, beta: float = 0.9) -> list[np.ndarray]:
    w, v, path = w0.copy(), np.zeros_like(w0), [w0.copy()]
    for _ in range(steps):
        v = beta * v + grad(w)
        w = w - lr * v
        path.append(w.copy())
    return path


def rmsprop(grad: Grad, w0: np.ndarray, lr: float, steps: int, beta: float = 0.9, eps: float = 1e-8) -> list[np.ndarray]:
    w, s, path = w0.copy(), np.zeros_like(w0), [w0.copy()]
    for _ in range(steps):
        g = grad(w)
        s = beta * s + (1 - beta) * g * g
        w = w - lr * g / (np.sqrt(s) + eps)
        path.append(w.copy())
    return path


def adam(grad: Grad, w0: np.ndarray, lr: float, steps: int, b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8) -> list[np.ndarray]:
    w, m, v, path = w0.copy(), np.zeros_like(w0), np.zeros_like(w0), [w0.copy()]
    for t in range(1, steps + 1):
        g = grad(w)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        mhat, vhat = m / (1 - b1 ** t), v / (1 - b2 ** t)  # bias correction: m, v start at 0
        w = w - lr * mhat / (np.sqrt(vhat) + eps)
        path.append(w.copy())
    return path


def steps_to(path: list[np.ndarray], f: Callable[[np.ndarray], float], tol: float) -> int | str:
    for i, w in enumerate(path):
        if f(w) < tol:
            return i
    return f">{len(path) - 1}"


def main() -> None:
    args = parse_args(__doc__)
    rng = np.random.default_rng(args.seed)

    section("1. The learning-rate threshold: lr < 2/L on a quadratic")
    # f(w) = 0.5 * w^T A w with eigenvalues mu=1, L=10 -> kappa = 10
    a = np.diag([1.0, 10.0])
    f = lambda w: 0.5 * w @ a @ w  # noqa: E731
    g = lambda w: a @ w  # noqa: E731
    w0 = np.array([1.0, 1.0])
    rows = []
    for lr in [0.01, 0.1, 0.18, 0.199, 0.201, 0.25]:
        path = gd(g, w0, lr, 200)
        rows.append({"lr": lr, "lr * L": lr * 10, "f after 200 steps": float(f(path[-1])),
                     "status": "diverged" if f(path[-1]) > f(w0) else "converged"})
    table(rows)
    kv("lr-threshold", below_2_over_L=rows[3]["f after 200 steps"], above_2_over_L=rows[4]["f after 200 steps"])

    section("2. Conditioning: steps to converge grow with kappa; scaling features fixes it")
    rows = []
    for kappa in [1, 10, 100, 1000]:
        a = np.diag([1.0, float(kappa)])
        fq = lambda w, a=a: 0.5 * w @ a @ w  # noqa: E731
        lr = 2 / (kappa + 1)                 # optimal fixed lr for GD
        p_gd = gd(lambda w, a=a: a @ w, w0, lr, 20000)
        p_m = momentum(lambda w, a=a: a @ w, w0, lr=1 / kappa, steps=20000,
                       beta=((np.sqrt(kappa) - 1) / (np.sqrt(kappa) + 1)) ** 2)  # heavy-ball tuned
        rows.append({"kappa": kappa, "GD steps to f<1e-8": steps_to(p_gd, fq, 1e-8),
                     "momentum steps": steps_to(p_m, fq, 1e-8)})
    table(rows)
    kv("conditioning", gd_k1000=rows[-1]["GD steps to f<1e-8"], momentum_k1000=rows[-1]["momentum steps"])
    # Real regression: one feature in metres, one in millimetres.
    n = 500
    x = np.c_[rng.normal(size=n), 1000 * rng.normal(size=n)]
    y = x @ np.array([2.0, 0.003]) + rng.normal(scale=0.1, size=n)
    def steps_lin(xm: np.ndarray) -> tuple[int | str, float]:
        h = xm.T @ xm / n * 2
        lam = np.linalg.eigvalsh(h)
        loss = lambda w: float(np.mean((xm @ w - y) ** 2))  # noqa: E731
        best = float(np.mean((xm @ np.linalg.lstsq(xm, y, rcond=None)[0] - y) ** 2))
        path = gd(lambda w: 2 * xm.T @ (xm @ w - y) / n, np.zeros(2), 1.9 / lam.max(), 100000)
        return steps_to(path, lambda w: loss(w) - best, 1e-6), float(lam.max() / lam.min())
    raw_steps, raw_kappa = steps_lin(x)
    xs = (x - x.mean(0)) / x.std(0)
    sc_steps, sc_kappa = steps_lin(xs)
    table([{"features": "raw (m and mm)", "kappa": raw_kappa, "GD steps": raw_steps},
           {"features": "standardised", "kappa": sc_kappa, "GD steps": sc_steps}])
    kv("scaling", raw_kappa=raw_kappa, scaled_kappa=sc_kappa, scaled_steps=sc_steps)

    section("3. Optimizers on the Rosenbrock valley f = (1-x)^2 + 100 (y - x^2)^2")
    fr = lambda w: (1 - w[0]) ** 2 + 100 * (w[1] - w[0] ** 2) ** 2  # noqa: E731
    gr = lambda w: np.array([-2 * (1 - w[0]) - 400 * w[0] * (w[1] - w[0] ** 2), 200 * (w[1] - w[0] ** 2)])  # noqa: E731
    start = np.array([-1.5, 2.0])
    steps = 3000 if args.quick else 20000
    runs = {
        "GD lr=1e-3": gd(gr, start, 1e-3, steps),
        "GD lr=2e-3 (near its edge)": gd(gr, start, 2e-3, steps),
        "momentum lr=3e-4 b=0.9": momentum(gr, start, 3e-4, steps),
        "RMSProp lr=1e-2": rmsprop(gr, start, 1e-2, steps),
        "Adam lr=2e-2": adam(gr, start, 2e-2, steps),
    }
    rows = [{"optimizer": k, f"f after {steps}": float(fr(p[-1])), "steps to f<1e-3": steps_to(p, fr, 1e-3),
             "f over last 4 steps": " ".join(f"{fr(q):.4f}" for q in p[-4:])} for k, p in runs.items()]
    table(rows)
    kv("rosenbrock", gd=float(fr(runs["GD lr=1e-3"][-1])), adam=float(fr(runs["Adam lr=2e-2"][-1])))
    print("  The minimum is (1, 1). GD's lr is capped by the steep walls (2/L), so it crawls along the flat floor;")
    print("  doubling it helps a lot here, but a bit more diverges. Momentum carries speed along the valley.")
    print("  RMSProp's step is ~lr whatever the gradient size, so with a constant lr it ends up bouncing between")
    print("  two points (see the last 4 values) instead of settling. That is one reason lr schedules decay to ~0.")

    section("4. SGD noise floor: constant lr stalls, decayed lr keeps improving")
    n, d = 5000, 20
    x = rng.normal(size=(n, d))
    w_true = rng.normal(size=d)
    y = x @ w_true + rng.normal(scale=1.0, size=n)
    w_star = np.linalg.lstsq(x, y, rcond=None)[0]
    excess = lambda w: float(np.mean((x @ w - y) ** 2) - np.mean((x @ w_star - y) ** 2))  # noqa: E731
    total = 2000 if args.quick else 10000
    rows = []
    for bs in [1, 32]:
        for sched in ["constant", "1/t decay"]:
            w = np.zeros(d)
            for t in range(total):
                idx = rng.integers(0, n, bs)
                lr = 0.02 if sched == "constant" else 0.02 / (1 + t / 200)
                w -= lr * 2 * x[idx].T @ (x[idx] @ w - y[idx]) / bs
            rows.append({"batch": bs, "lr schedule": sched, "excess loss": excess(w)})
    table(rows)
    kv("sgd-floor", bs1_const=rows[0]["excess loss"], bs1_decay=rows[1]["excess loss"],
       bs32_const=rows[2]["excess loss"])
    print("  Excess loss = loss - loss of the exact least-squares fit. Bigger batch or decayed lr = less noise.")


if __name__ == "__main__":
    main()
