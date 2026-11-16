"""
04 - Decision trees, random forests, gradient boosting (from scratch + sklearn)
===============================================================================

Interview topics
----------------
- Impurity: Gini = 1 - sum p_k^2, entropy = -sum p_k log2 p_k. A split is
  chosen greedily to maximise the impurity DECREASE (information gain),
  weighted by child sizes. Regression trees use variance (MSE) instead.
- Trees: no scaling needed, handle mixed feature types and interactions,
  interpretable when shallow. But high variance: an unlimited tree reaches
  100% train accuracy by memorising, and small data changes give a very
  different tree. Control with max_depth, min_samples_leaf, pruning
  (ccp_alpha).
- Random forest = bagging (each tree on a bootstrap resample, ~63.2% unique
  rows) + random feature subset at each split (max_features ~ sqrt(p) for
  classification). Averaging B trees with pairwise correlation rho leaves
  variance rho*s^2 + (1-rho)/B * s^2 -> decorrelating the trees is the whole
  point (section 3 measures rho). Mostly fixes VARIANCE. Free validation from
  out-of-bag (OOB) rows. More trees never overfit; they just stop helping.
- Gradient boosting: build trees SEQUENTIALLY, each fitting the negative
  gradient of the loss (for MSE = the residuals) of the current ensemble;
  add it scaled by a learning rate (shrinkage). Mostly reduces BIAS with
  shallow trees (depth 3-8). CAN overfit with too many rounds -> early
  stopping on validation. lr and n_rounds trade off (smaller lr, more trees).
- XGBoost / LightGBM / sklearn HistGradientBoosting: second-order (Newton)
  steps, L1/L2 on leaf values, histogram binning of features (fast), native
  missing-value handling, leaf-wise growth (LightGBM). On tabular data a
  tuned GBM is still the default strong baseline over deep nets.
- Feature importance: impurity-based importance is biased toward
  high-cardinality features; prefer permutation importance or SHAP.

Run:  python ml_course/04_trees_forests_boosting.py [--quick]
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import load_breast_cancer, make_friedman1
from sklearn.ensemble import (GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestClassifier,
                              RandomForestRegressor)
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from common import kv, parse_args, section, table


def gini(counts: np.ndarray) -> float:
    p = counts / counts.sum()
    return float(1 - np.sum(p ** 2))


def entropy(counts: np.ndarray) -> float:
    p = counts[counts > 0] / counts.sum()
    return float(-np.sum(p * np.log2(p))) + 0.0  # + 0.0 turns -0.0 into 0.0


@dataclass
class Node:
    pred: int
    feature: int = -1
    threshold: float = 0.0
    left: Node | None = None
    right: Node | None = None


class Tree:
    """Greedy CART classifier with Gini impurity. Exhaustive over features and midpoints, vectorised per feature."""

    def __init__(self, max_depth: int = 99, min_leaf: int = 1) -> None:
        self.max_depth, self.min_leaf = max_depth, min_leaf

    def fit(self, x: np.ndarray, y: np.ndarray) -> Tree:
        self.k = int(y.max()) + 1
        self.root = self._grow(x, y, 0)
        return self

    def _best_split(self, x: np.ndarray, y: np.ndarray) -> tuple[int, float, float]:
        n = len(y)
        best = (-1, 0.0, gini(np.bincount(y, minlength=self.k)))  # must beat the parent
        onehot = np.eye(self.k)[y]
        for j in range(x.shape[1]):
            order = np.argsort(x[:, j])
            xs, cum = x[order, j], np.cumsum(onehot[order], axis=0)  # class counts left of each cut
            left_n = np.arange(1, n)
            lc, rc = cum[:-1], cum[-1] - cum[:-1]
            g_l = 1 - np.sum((lc / left_n[:, None]) ** 2, axis=1)
            g_r = 1 - np.sum((rc / (n - left_n)[:, None]) ** 2, axis=1)
            weighted = (left_n * g_l + (n - left_n) * g_r) / n
            valid = (xs[1:] != xs[:-1]) & (left_n >= self.min_leaf) & (n - left_n >= self.min_leaf)
            if not valid.any():
                continue
            i = int(np.argmin(np.where(valid, weighted, np.inf)))
            if weighted[i] < best[2] - 1e-12:
                best = (j, float((xs[i] + xs[i + 1]) / 2), float(weighted[i]))
        return best

    def _grow(self, x: np.ndarray, y: np.ndarray, depth: int) -> Node:
        node = Node(pred=int(np.bincount(y, minlength=self.k).argmax()))
        if depth >= self.max_depth or len(np.unique(y)) == 1:
            return node
        j, t, _ = self._best_split(x, y)
        if j < 0:
            return node
        mask = x[:, j] <= t
        node.feature, node.threshold = j, t
        node.left, node.right = self._grow(x[mask], y[mask], depth + 1), self._grow(x[~mask], y[~mask], depth + 1)
        return node

    def predict(self, x: np.ndarray) -> np.ndarray:
        out = np.empty(len(x), dtype=int)
        for i, row in enumerate(x):
            n = self.root
            while n.left is not None:
                n = n.left if row[n.feature] <= n.threshold else n.right
            out[i] = n.pred
        return out


def boost(xtr: np.ndarray, ytr: np.ndarray, xte: np.ndarray, rounds: int, lr: float, depth: int) -> tuple[list[float], list[float]]:
    """Gradient boosting for squared error: each tree fits the residuals (= negative gradient)."""
    f_tr = np.full(len(ytr), ytr.mean())
    f_te = np.full(len(xte), ytr.mean())
    tr_hist, te_preds = [], []
    for _ in range(rounds):
        residual = ytr - f_tr
        t = DecisionTreeRegressor(max_depth=depth).fit(xtr, residual)
        f_tr += lr * t.predict(xtr)
        f_te += lr * t.predict(xte)
        tr_hist.append(float(np.mean((ytr - f_tr) ** 2)))
        te_preds.append(f_te.copy())
    return tr_hist, te_preds


def main() -> None:
    args = parse_args(__doc__)

    section("1. Impurity: which node is purer?")
    rows = [{"node": str(c), "gini": gini(np.array(c)), "entropy (bits)": entropy(np.array(c))}
            for c in ([50, 50], [99, 1], [100, 0], [34, 33, 33])]
    table(rows)
    kv("impurity", gini_50_50=rows[0]["gini"], gini_99_1=rows[1]["gini"], entropy_50_50=rows[0]["entropy (bits)"])

    section("2. Our tree vs sklearn, and depth vs overfitting")
    x, y = load_breast_cancer(return_X_y=True)
    xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.3, stratify=y, random_state=args.seed)
    rows = []
    for depth in [1, 2, 4, 8, None]:
        ours = Tree(max_depth=depth or 99).fit(xtr, ytr)
        sk = DecisionTreeClassifier(max_depth=depth, random_state=0).fit(xtr, ytr)
        rows.append({"max_depth": depth or "unlimited", "ours train": float((ours.predict(xtr) == ytr).mean()),
                     "ours test": float((ours.predict(xte) == yte).mean()), "sklearn test": float(sk.score(xte, yte))})
    table(rows)
    kv("tree", deep_train=rows[-1]["ours train"], deep_test=rows[-1]["ours test"],
       ours_vs_sklearn_depth4=abs(rows[2]["ours test"] - rows[2]["sklearn test"]))
    print("  Unlimited depth memorises the training set (train acc 1.0). Small test diffs vs sklearn come from")
    print("  tie-breaking between equally good splits.")

    section("3. Random forest: averaging DECORRELATED trees")
    # Noisy, harder version of the task so single trees have real variance.
    rng = np.random.default_rng(args.seed)
    xn = np.c_[xtr, rng.normal(size=(len(xtr), 30))]
    xnt = np.c_[xte, rng.normal(size=(len(xte), 30))]
    ytr_noisy = np.where(rng.random(len(ytr)) < 0.1, 1 - ytr, ytr)  # 10% label noise
    n_trees = 100 if args.quick else 300
    rows = []
    single = DecisionTreeClassifier(random_state=0).fit(xn, ytr_noisy).score(xnt, yte)
    rows.append({"model": "single deep tree", "test acc": float(single), "tree corr": "-", "OOB acc": "-"})
    for mf, label in [(1.0, "bagging only (all features)"), ("sqrt", "random forest (sqrt features)")]:
        rf = RandomForestClassifier(n_estimators=n_trees, max_features=mf, oob_score=True, random_state=0, n_jobs=-1)
        rf.fit(xn, ytr_noisy)
        probs = np.array([t.predict_proba(xnt)[:, 1] for t in rf.estimators_[:50]])
        corr = np.corrcoef(probs)
        rho = float(corr[np.triu_indices_from(corr, 1)].mean())
        rows.append({"model": label, "test acc": float(rf.score(xnt, yte)), "tree corr": rho, "OOB acc": float(rf.oob_score_)})
    table(rows)
    kv("forest", single=rows[0]["test acc"], bagging=rows[1]["test acc"], rf=rows[2]["test acc"],
       corr_bag=rows[1]["tree corr"], corr_rf=rows[2]["tree corr"])
    print("  OOB accuracy is measured on noisy labels, so it sits below test accuracy on clean labels.")

    section("4. Gradient boosting from scratch (fit residuals) vs sklearn")
    xf, yf = make_friedman1(n_samples=2000, noise=1.0, random_state=args.seed)
    xtr, xte, ytr, yte = train_test_split(xf, yf, test_size=0.5, random_state=args.seed)
    rounds = 300 if args.quick else 1000
    rows = []
    for lr in [1.0, 0.1]:
        tr_hist, te_preds = boost(xtr, ytr, xte, rounds, lr, depth=3)
        te_hist = [float(np.mean((p - yte) ** 2)) for p in te_preds]
        best = int(np.argmin(te_hist))
        rows.append({"learning rate": lr, "train MSE": tr_hist[-1], "test MSE @ end": te_hist[-1],
                     "best test MSE": te_hist[best], "best round": best + 1})
    sk = GradientBoostingRegressor(n_estimators=rounds, learning_rate=0.1, max_depth=3, random_state=0).fit(xtr, ytr)
    hgb = HistGradientBoostingRegressor(max_iter=rounds, learning_rate=0.1, early_stopping=True, random_state=0).fit(xtr, ytr)
    table(rows)
    sk_mse = float(np.mean((sk.predict(xte) - yte) ** 2))
    hgb_mse = float(np.mean((hgb.predict(xte) - yte) ** 2))
    print(f"  sklearn GradientBoosting lr=0.1: test MSE {sk_mse:.3f}   HistGradientBoosting (early stopping): {hgb_mse:.3f}"
          f" after {hgb.n_iter_} rounds")
    kv("boosting", ours_lr01=rows[1]["test MSE @ end"], sklearn=sk_mse, lr1_overfit_gap=rows[0]["test MSE @ end"] - rows[0]["best test MSE"])
    print("  Noise variance is 1.0. lr=1.0 overshoots and starts overfitting early; lr=0.1 is slower but better.")

    section("5. Feature importance: impurity-based vs permutation")
    xi = np.c_[xtr[:, :5], rng.integers(0, 1000, size=len(xtr))]  # 5 real features + a random ID-like column
    xit = np.c_[xte[:, :5], rng.integers(0, 1000, size=len(xte))]
    model = RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=-1).fit(xi, ytr)
    perm = permutation_importance(model, xit, yte, n_repeats=5, random_state=0).importances_mean
    names = ["x0", "x1", "x2", "x3", "x4", "random_id"]
    table([{"feature": n, "impurity importance": float(a), "permutation importance (test)": float(b)}
           for n, a, b in zip(names, model.feature_importances_, perm)])
    kv("importance", id_impurity=float(model.feature_importances_[-1]), id_permutation=float(perm[-1]))
    print("  The pure-noise ID column still gets clearly non-zero impurity importance (trees split on it to fit")
    print("  noise in the training set); permuting it on held-out data shows it is useless.")


if __name__ == "__main__":
    main()
