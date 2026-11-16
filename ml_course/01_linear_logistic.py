"""
01 - Linear and logistic regression from scratch
================================================

Interview topics
----------------
- Linear regression: y_hat = Xw + b, minimise MSE. Closed form (normal
  equation) w = (X^T X)^-1 X^T y, but in practice solve with QR/SVD
  (np.linalg.lstsq): forming X^T X squares the condition number, so
  collinear features blow it up. Gradient: (2/n) X^T (Xw - y).
- Assumptions worth naming: linear in parameters, independent errors,
  constant variance, no perfect collinearity. Collinearity doesn't hurt
  predictions much but makes individual coefficients meaningless (unstable,
  can flip sign) -> ridge, or drop/merge features.
- Logistic regression: p = sigmoid(Xw + b), minimise binary cross-entropy
  (= negative log-likelihood of a Bernoulli). Gradient has the same shape as
  linear regression: X^T (p - y) / n. Convex, so no bad local minima.
- Why not MSE on a sigmoid? The gradient w.r.t. the logit is
  2(p - y) p (1 - p): when the model is confidently WRONG, p(1 - p) ~ 0 and it
  stops learning. CE's gradient is (p - y): biggest exactly then. (Section 3.)
- Softmax regression = multiclass logistic regression. Compute softmax /
  log-softmax with the max-subtraction (log-sum-exp) trick or it overflows.
- Coefficients: in logistic regression exp(w_j) is the odds ratio for a
  one-unit increase in feature j (one std if standardised).
- Perfectly separable data -> weights grow without bound (no finite MLE);
  regularisation (C in sklearn) fixes it.

Run:  python ml_course/01_linear_logistic.py [--quick]
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_breast_cancer, load_diabetes, load_digits
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from common import kv, parse_args, section, table


def sigmoid(z: np.ndarray) -> np.ndarray:
    # Stable for large |z|: never calls exp on a large positive number.
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1 / (1 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1 + ez)
    return out


def bce(p: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def log_softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)  # log-sum-exp trick: shift doesn't change softmax
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


def fit_logreg_gd(x: np.ndarray, y: np.ndarray, lr: float, steps: int) -> tuple[np.ndarray, list[float]]:
    xb = np.c_[np.ones(len(x)), x]  # bias as a column of ones
    w = np.zeros(xb.shape[1])
    hist = []
    for _ in range(steps):
        p = sigmoid(xb @ w)
        w -= lr * xb.T @ (p - y) / len(y)
        hist.append(bce(p, y))
    return w, hist


def main() -> None:
    args = parse_args(__doc__)

    section("1. Linear regression three ways: normal equation, lstsq, gradient descent")
    x, y = load_diabetes(return_X_y=True)
    x = StandardScaler().fit_transform(x)
    xb = np.c_[np.ones(len(x)), x]
    w_normal = np.linalg.solve(xb.T @ xb, xb.T @ y)
    w_lstsq = np.linalg.lstsq(xb, y, rcond=None)[0]
    w_gd = np.zeros(xb.shape[1])
    lr = 0.1
    for _ in range(2000 if args.quick else 10000):
        w_gd -= lr * 2 * xb.T @ (xb @ w_gd - y) / len(y)
    sk = LinearRegression().fit(x, y)
    w_sk = np.r_[sk.intercept_, sk.coef_]
    rows = [{"method": m, "max|w - sklearn|": float(np.abs(w - w_sk).max()),
             "train MSE": float(np.mean((xb @ w - y) ** 2))}
            for m, w in [("normal equation", w_normal), ("lstsq (SVD)", w_lstsq), ("gradient descent", w_gd)]]
    table(rows)
    kv("linreg", normal_vs_sklearn=float(np.abs(w_normal - w_sk).max()), gd_vs_sklearn=float(np.abs(w_gd - w_sk).max()))
    print("  GD is slowest to agree: diabetes features are correlated (s1/s2 ~ 0.9), so the loss is a long valley.")

    section("2. Collinearity: predictions fine, coefficients meaningless")
    rng = np.random.default_rng(args.seed)
    n = 200
    a = rng.normal(size=n)
    rows = []
    for eps in [1.0, 1e-2, 1e-4, 1e-6]:
        b = a + eps * rng.normal(size=n)  # b is almost a copy of a
        xc = np.c_[np.ones(n), a, b]
        yc = 3 * a + rng.normal(scale=0.5, size=n)
        coefs = [np.linalg.lstsq(xc[idx], yc[idx], rcond=None)[0][1:] for idx in
                 (rng.choice(n, n, replace=True) for _ in range(20))]  # bootstrap resamples
        coefs = np.array(coefs)
        rows.append({"noise between a,b": eps, "cond(X^T X)": float(np.linalg.cond(xc.T @ xc)),
                     "w_a std over resamples": float(coefs[:, 0].std()), "w_a + w_b mean": float(coefs.sum(1).mean())})
    table(rows)
    kv("collinear", cond_max=rows[-1]["cond(X^T X)"], coef_std_max=rows[-1]["w_a std over resamples"],
       coef_std_min=rows[0]["w_a std over resamples"])
    print("  The SUM w_a + w_b stays ~3 (the prediction is stable); the split between them is arbitrary.")

    section("3. Why cross-entropy, not MSE, for a sigmoid output")
    print("  True label y=1. Gradient of the loss w.r.t. the logit z:")
    rows = []
    for z in [-10.0, -3.0, 0.0, 3.0]:
        p = 1 / (1 + np.exp(-z))
        rows.append({"logit z": z, "p": float(p), "dCE/dz = p-y": float(p - 1), "dMSE/dz = 2(p-y)p(1-p)": float(2 * (p - 1) * p * (1 - p))})
    table(rows)
    kv("ce-vs-mse", ce_grad_wrong=rows[0]["dCE/dz = p-y"], mse_grad_wrong=rows[0]["dMSE/dz = 2(p-y)p(1-p)"])
    print("  Confidently wrong (z=-10): CE pushes at full strength, MSE barely moves.")

    section("4. Logistic regression by gradient descent vs sklearn")
    xb_, yb_ = load_breast_cancer(return_X_y=True)
    xtr, xte, ytr, yte = train_test_split(xb_, yb_, test_size=0.3, stratify=yb_, random_state=args.seed)
    sc = StandardScaler().fit(xtr)  # fit on TRAIN only (lesson 00)
    xtr, xte = sc.transform(xtr), sc.transform(xte)
    w, hist = fit_logreg_gd(xtr, ytr, lr=0.5, steps=500 if args.quick else 3000)
    ours = float(((sigmoid(np.c_[np.ones(len(xte)), xte] @ w) > 0.5) == yte).mean())
    skl = LogisticRegression(C=1e6, max_iter=10000).fit(xtr, ytr)  # ~unregularised to match ours
    print(f"  loss: start {hist[0]:.3f} -> end {hist[-1]:.4f}")
    kv("logreg", ours_acc=ours, sklearn_acc=float(skl.score(xte, yte)))
    # Numerical gradient check on the BCE loss: analytic X^T(p-y)/n vs central differences.
    xb2 = np.c_[np.ones(len(xtr)), xtr]
    w0 = np.random.default_rng(1).normal(scale=0.1, size=xb2.shape[1])
    analytic = xb2.T @ (sigmoid(xb2 @ w0) - ytr) / len(ytr)
    h, numeric = 1e-6, np.zeros_like(w0)
    for j in range(len(w0)):
        e = np.zeros_like(w0)
        e[j] = h
        numeric[j] = (bce(sigmoid(xb2 @ (w0 + e)), ytr) - bce(sigmoid(xb2 @ (w0 - e)), ytr)) / (2 * h)
    rel = float(np.abs(analytic - numeric).max() / np.abs(analytic).max())
    kv("gradcheck", rel_err=rel)
    top = np.argsort(-np.abs(w[1:]))[:3]
    names = load_breast_cancer().feature_names
    for j in top:
        print(f"  {names[j]:<24} w={w[1 + j]:+.2f}  odds x{np.exp(w[1 + j]):.2f} per +1 std")
    print("  (Near-separable data, no regularisation: these weights keep growing with more steps.)")

    section("5. Softmax regression on digits, and the overflow trap")
    z = np.array([[1000.0, 999.0, 0.0]])
    with np.errstate(over="ignore", invalid="ignore"):
        naive = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
    stable = np.exp(log_softmax(z))
    print(f"  naive softmax of [1000, 999, 0] = {naive.round(3).tolist()}   stable = {stable.round(3).tolist()}")
    kv("softmax", naive_has_nan=bool(np.isnan(naive).any()), stable_sum=float(stable.sum()))
    xd, yd = load_digits(return_X_y=True)
    xtr, xte, ytr, yte = train_test_split(xd / 16.0, yd, test_size=0.3, stratify=yd, random_state=args.seed)
    k = 10
    w = np.zeros((xtr.shape[1] + 1, k))
    xtrb, xteb = np.c_[np.ones(len(xtr)), xtr], np.c_[np.ones(len(xte)), xte]
    onehot = np.eye(k)[ytr]
    for _ in range(300 if args.quick else 2000):
        p = np.exp(log_softmax(xtrb @ w))
        w -= 0.5 * xtrb.T @ (p - onehot) / len(ytr)  # same (p - y) form as the binary case
    acc = float((np.argmax(xteb @ w, axis=1) == yte).mean())
    kv("softmax-digits", test_acc=acc)


if __name__ == "__main__":
    main()
