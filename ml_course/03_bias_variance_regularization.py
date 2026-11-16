"""
03 - Bias vs variance, overfitting, L1 vs L2
============================================

Interview topics
----------------
- Expected test error = bias^2 + variance + irreducible noise.
    bias: error of the AVERAGE model over many training sets (too simple,
          systematically wrong -> underfitting).
    variance: how much the model changes when the training set changes
          (too flexible, fits noise -> overfitting).
  Section 2 measures both directly by refitting on 200 resampled datasets.
- Diagnose with train vs validation error:
    high train, high val (small gap)  -> high bias: bigger model, better
                                         features, less regularisation.
    low train, high val (big gap)     -> high variance: more data,
                                         regularisation, simpler model,
                                         early stopping, ensembling/bagging.
- Learning curves (error vs training-set size): more data closes a variance
  gap but does nothing for bias (section 4).
- L2 (ridge, weight decay): penalty lambda*||w||^2 shrinks all weights
  smoothly, never exactly to zero; good with many small correlated effects;
  Bayesian view = Gaussian prior. Closed form (X^T X + lambda I)^-1 X^T y,
  which is always invertible -> also fixes collinearity (lesson 01).
- L1 (lasso): lambda*||w||_1 drives weights to EXACTLY zero (the L1 ball has
  corners on the axes; the subgradient doesn't vanish near 0) -> feature
  selection; Laplace prior. With correlated features it picks one somewhat
  arbitrarily. Elastic net = both.
- Scale features before L1/L2: the penalty treats every coefficient alike.
- Modern caveat: very over-parameterised models (deep nets) can show
  "double descent": test error falls again past the interpolation point.
  The classic U-curve is the under-parameterised story.

Run:  python ml_course/03_bias_variance_regularization.py [--quick]
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from common import kv, parse_args, section, table

warnings.filterwarnings("ignore")


def true_f(x: np.ndarray) -> np.ndarray:
    return np.sin(np.pi * x)  # one full period on [-1, 1]


def sample(rng: np.random.Generator, n: int, noise: float = 0.3) -> tuple[np.ndarray, np.ndarray]:
    x = rng.uniform(-1, 1, n)  # [-1,1], not [0,1]: powers of x on [0,1] are nearly collinear
    return x, true_f(x) + rng.normal(scale=noise, size=n)


def poly(degree: int, alpha: float = 0.0):
    reg = Ridge(alpha=alpha) if alpha else LinearRegression()
    return make_pipeline(PolynomialFeatures(degree, include_bias=False), StandardScaler(), reg)


def main() -> None:
    args = parse_args(__doc__)
    rng = np.random.default_rng(args.seed)
    noise = 0.3

    section("1. Model complexity: train error keeps falling, test error is U-shaped")
    xtr, ytr = sample(rng, 30, noise)
    xte, yte = sample(rng, 2000, noise)
    rows = []
    for d in [1, 3, 5, 9, 12, 15]:
        m = poly(d).fit(xtr[:, None], ytr)
        rows.append({"degree": d, "train MSE": float(np.mean((m.predict(xtr[:, None]) - ytr) ** 2)),
                     "test MSE": float(np.mean((m.predict(xte[:, None]) - yte) ** 2))})
    table(rows)
    best = min(rows, key=lambda r: r["test MSE"])
    kv("complexity", best_degree=best["degree"], train_d15=rows[-1]["train MSE"], test_d15=rows[-1]["test MSE"],
       test_best=best["test MSE"])
    print(f"  Irreducible noise variance = {noise ** 2:.2f}: no model can get test MSE below that.")

    section("2. Bias^2 and variance measured by refitting on many training sets")
    grid = np.linspace(-0.95, 0.95, 50)
    reps = 100 if args.quick else 300
    rows = []
    for d in [1, 3, 12]:
        preds = np.empty((reps, len(grid)))
        for r in range(reps):
            x, y = sample(rng, 30, noise)
            preds[r] = poly(d).fit(x[:, None], y).predict(grid[:, None])
        bias2 = float(np.mean((preds.mean(0) - true_f(grid)) ** 2))
        var = float(np.mean(preds.var(0)))
        rows.append({"degree": d, "bias^2": bias2, "variance": var, "bias^2+var+noise": bias2 + var + noise ** 2})
    table(rows)
    kv("bias-variance", bias_d1=rows[0]["bias^2"], bias_d3=rows[1]["bias^2"], var_d3=rows[1]["variance"],
       var_d12=rows[2]["variance"])

    section("3. Regularisation tames the flexible model (degree 12 + ridge)")
    rows = []
    for alpha in [0.0, 1e-4, 1e-2, 1e-1, 1.0, 10.0]:
        test = []
        for _ in range(20 if args.quick else 100):
            x, y = sample(rng, 30, noise)
            test.append(np.mean((poly(12, alpha).fit(x[:, None], y).predict(xte[:, None]) - yte) ** 2))
        rows.append({"ridge alpha": alpha, "mean test MSE": float(np.mean(test))})
    table(rows)
    kv("ridge", alpha0=rows[0]["mean test MSE"], best=min(r["mean test MSE"] for r in rows))
    print("  Too little: variance wins. Too much: the model is forced flat and bias wins. Tune alpha on validation.")

    section("4. Learning curves: more data cures variance, not bias")
    rows = []
    for n in [15, 30, 100, 300, 1000]:
        row = {"n train": n}
        for d in [1, 12]:
            x, y = sample(rng, n, noise)
            m = poly(d).fit(x[:, None], y)
            row[f"d={d} train"] = float(np.mean((m.predict(x[:, None]) - y) ** 2))
            row[f"d={d} test"] = float(np.mean((m.predict(xte[:, None]) - yte) ** 2))
        rows.append(row)
    table(rows)
    kv("learning-curve", d1_test_n1000=rows[-1]["d=1 test"], d12_test_n15=rows[0]["d=12 test"],
       d12_test_n1000=rows[-1]["d=12 test"])
    print("  Degree 1 plateaus far above the noise floor (bias). Degree 12 converges to it once n is large.")

    section("5. L1 vs L2: sparsity")
    n, p, k = 200, 50, 5
    x = rng.normal(size=(n, p))
    w_true = np.zeros(p)
    w_true[:k] = rng.choice([-3, -2, 2, 3], size=k)
    y = x @ w_true + rng.normal(scale=1.0, size=n)
    xs = StandardScaler().fit_transform(x)
    rows = []
    for name, m in [("OLS", LinearRegression()), ("ridge a=10", Ridge(alpha=10)), ("lasso a=0.1", Lasso(alpha=0.1)),
                    ("lasso a=0.5", Lasso(alpha=0.5))]:
        w = m.fit(xs, y).coef_
        rows.append({"model": name, "exact zeros": int(np.sum(w == 0)), "|w|<0.05": int(np.sum(np.abs(w) < 0.05)),
                     "true features kept": int(np.sum(w[:k] != 0)), "noise weight L1": float(np.abs(w[k:]).sum())})
    table(rows)
    kv("l1-l2", ridge_zeros=rows[1]["exact zeros"], lasso_zeros=rows[3]["exact zeros"],
       lasso_true_kept=rows[3]["true features kept"])
    print(f"  {k} real features out of {p}. Lasso zeroes the noise features exactly; ridge only shrinks them.")
    # Why: proximal (soft-threshold) step of L1 vs shrink step of L2 on one weight.
    print("  One update with penalty strength 0.5 on weights [0.3, 2.0]:")
    w = np.array([0.3, 2.0])
    print(f"    L2 shrink  w/(1+0.5)            -> {np.round(w / 1.5, 3).tolist()}  (smaller, never 0)")
    print(f"    L1 soft-threshold sign(w)max(|w|-0.5, 0) -> {np.round(np.sign(w) * np.maximum(np.abs(w) - 0.5, 0), 3).tolist()}  (small weight -> exactly 0)")


if __name__ == "__main__":
    main()
