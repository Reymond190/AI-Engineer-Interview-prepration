"""
07 - Metrics, class imbalance, thresholds and calibration
=========================================================

Interview topics
----------------
- Accuracy paradox: with 1% positives, "always negative" scores 99%
  accuracy and catches nothing. Pick the metric from the cost of errors.
- Confusion matrix -> precision = TP/(TP+FP) ("when I say yes, am I right?"),
  recall = TP/(TP+FN) ("of all real yeses, how many did I catch?"),
  F1 = harmonic mean (punishes imbalance between the two), specificity,
  FPR = FP/(FP+TN). Fraud/medical screening: recall first; spam: precision.
- ROC-AUC = P(score of random positive > score of random negative); it is
  insensitive to class balance because TPR and FPR are each normalised
  within one class. Under heavy imbalance it can look great while precision
  is poor -> report PR-AUC (average precision), whose baseline equals the
  positive rate (section 3).
- The 0.5 threshold is arbitrary. The model outputs scores; choose the
  operating threshold on VALIDATION data for the metric/cost you care
  about (max F1, recall >= 95%, expected cost...). Never tune it on test.
- Imbalance fixes, cheapest first: right metric + threshold; class weights
  (scale the loss per class); resampling (undersample majority / oversample
  minority / SMOTE; only on the training fold!); collect more positives;
  anomaly-detection framing. Class weights / resampling shift probabilities,
  so they are no longer calibrated.
- Calibration: of all cases given p=0.8, ~80% should be positive. Measure
  with reliability curves, Brier score, ECE. Logistic regression is usually
  close; naive Bayes, boosted trees, SVM scores and re-weighted models are
  not. Fix with Platt scaling (sigmoid) or isotonic regression fitted on
  held-out data. Needed whenever probabilities drive decisions (expected
  value, risk scores, ranking mixed with thresholds).
- Regression metrics: MSE/RMSE (punish big errors, same units after root),
  MAE (robust to outliers, median), R^2 (vs predicting the mean), MAPE
  (breaks near 0).

Run:  python ml_course/07_metrics_imbalance.py [--quick]
"""

from __future__ import annotations

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

from common import kv, parse_args, section, table


def confusion(y: np.ndarray, pred: np.ndarray) -> dict[str, int]:
    return {"TP": int(((pred == 1) & (y == 1)).sum()), "FP": int(((pred == 1) & (y == 0)).sum()),
            "FN": int(((pred == 0) & (y == 1)).sum()), "TN": int(((pred == 0) & (y == 0)).sum())}


def prf(c: dict[str, int]) -> tuple[float, float, float]:
    p = c["TP"] / max(c["TP"] + c["FP"], 1)
    r = c["TP"] / max(c["TP"] + c["FN"], 1)
    return p, r, 2 * p * r / max(p + r, 1e-12)


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error: bin by predicted prob, average |accuracy - confidence| weighted by bin size."""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    return float(sum(abs(y[idx == b].mean() - p[idx == b].mean()) * (idx == b).mean() for b in range(bins) if np.any(idx == b)))


def main() -> None:
    args = parse_args(__doc__)

    def data(pos_rate: float, n: int = 20000, seed: int = args.seed):
        return make_classification(n_samples=n, n_features=20, n_informative=6, weights=[1 - pos_rate],
                                   class_sep=0.8, flip_y=0.0, random_state=seed)

    section("1. Accuracy paradox (2% positives)")
    x, y = data(0.02)
    xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.3, stratify=y, random_state=args.seed)
    always_neg = np.zeros_like(yte)
    model = LogisticRegression(max_iter=2000).fit(xtr, ytr)
    pred = model.predict(xte)
    rows = []
    for name, p in [("always negative", always_neg), ("logistic @0.5", pred)]:
        c = confusion(yte, p)
        pr, rc, f1 = prf(c)
        rows.append({"model": name, "accuracy": float((p == yte).mean()), **c, "precision": pr, "recall": rc, "F1": f1})
    table(rows)
    kv("paradox", always_neg_acc=rows[0]["accuracy"], always_neg_recall=rows[0]["recall"])
    pr, rc, f1 = prf(confusion(yte, pred))
    kv("prf-check", ours_f1=f1, sklearn_f1=float(f1_score(yte, pred)),
       match=bool(np.isclose(pr, precision_score(yte, pred)) and np.isclose(rc, recall_score(yte, pred))))

    section("2. Threshold tuning on VALIDATION (not test)")
    xtr2, xval, ytr2, yval = train_test_split(xtr, ytr, test_size=0.3, stratify=ytr, random_state=args.seed)
    m = LogisticRegression(max_iter=2000).fit(xtr2, ytr2)
    pv, pt = m.predict_proba(xval)[:, 1], m.predict_proba(xte)[:, 1]
    grid = np.linspace(0.02, 0.9, 89)
    f1s = [f1_score(yval, pv >= t) for t in grid]
    t_best = float(grid[int(np.argmax(f1s))])
    rows = []
    for name, t in [("default 0.5", 0.5), (f"best-F1 on val ({t_best:.2f})", t_best)]:
        pr, rc, f1 = prf(confusion(yte, (pt >= t).astype(int)))
        rows.append({"threshold": name, "test precision": pr, "test recall": rc, "test F1": f1})
    table(rows)
    kv("threshold", f1_at_05=rows[0]["test F1"], f1_tuned=rows[1]["test F1"], t_best=t_best)

    section("3. ROC-AUC vs PR-AUC as positives get rarer (the SAME scorer)")
    # Fix the classifier's quality: positive scores ~ N(2, 1), negative scores ~ N(0, 1). Only the class ratio changes.
    rng = np.random.default_rng(args.seed)
    rows = []
    for rate in [0.5, 0.1, 0.02, 0.005]:
        n_pos = 2000
        n_neg = int(n_pos * (1 - rate) / rate)
        s = np.r_[rng.normal(2, 1, n_pos), rng.normal(0, 1, n_neg)]
        d = np.r_[np.ones(n_pos), np.zeros(n_neg)]
        rows.append({"positive rate": rate, "ROC-AUC": float(roc_auc_score(d, s)), "PR-AUC (avg precision)": float(average_precision_score(d, s)),
                     "PR-AUC of random": float(d.mean())})
    table(rows)
    kv("auc", roc_rare=rows[-1]["ROC-AUC"], pr_rare=rows[-1]["PR-AUC (avg precision)"], pr_balanced=rows[0]["PR-AUC (avg precision)"])
    print("  Same separability, so ROC-AUC is constant. PR-AUC collapses: with 200 negatives per positive, even a")
    print("  small false-positive RATE is a large false-positive COUNT, so most flagged cases are false alarms.")

    section("4. Class weights: trade precision for recall (and lose calibration)")
    rows = []
    for cw in [None, "balanced"]:
        mm = LogisticRegression(max_iter=2000, class_weight=cw).fit(xtr, ytr)
        p = mm.predict_proba(xte)[:, 1]
        pr, rc, f1 = prf(confusion(yte, (p >= 0.5).astype(int)))
        rows.append({"class_weight": str(cw), "precision": pr, "recall": rc, "F1": f1,
                     "mean predicted p": float(p.mean()), "true positive rate": float(yte.mean())})
    table(rows)
    kv("class-weight", recall_none=rows[0]["recall"], recall_balanced=rows[1]["recall"],
       meanp_balanced=rows[1]["mean predicted p"])
    print("  'balanced' acts as if positives were 50%: recall jumps, and the probabilities are inflated ~10x.")

    section("5. Calibration: naive Bayes vs logistic, and fixing it")
    # 4 informative features, each copied 5 times with a little noise: the same evidence appears 5x.
    # Class-conditional Gaussians with shared covariance: exactly the model logistic regression assumes.
    y = (rng.random(20000) < 0.3).astype(int)
    x = rng.normal(size=(20000, 4)) + 0.5 * y[:, None]
    x = np.concatenate([x + 0.3 * rng.normal(size=x.shape) for _ in range(5)], axis=1)
    xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.5, stratify=y, random_state=args.seed)
    rows = []
    for name, mm in [("logistic regression", LogisticRegression(max_iter=2000)), ("Gaussian naive Bayes", GaussianNB()),
                     ("NB + isotonic (CV)", CalibratedClassifierCV(GaussianNB(), method="isotonic", cv=5)),
                     ("NB + Platt/sigmoid (CV)", CalibratedClassifierCV(GaussianNB(), method="sigmoid", cv=5))]:
        p = mm.fit(xtr, ytr).predict_proba(xte)[:, 1]
        rows.append({"model": name, "ECE": ece(yte, p), "Brier": float(np.mean((p - yte) ** 2)), "ROC-AUC": float(roc_auc_score(yte, p))})
    table(rows)
    kv("calibration", ece_lr=rows[0]["ECE"], ece_nb=rows[1]["ECE"], ece_nb_iso=rows[2]["ECE"])
    print("  Naive Bayes assumes independent features, so it multiplies correlated evidence -> overconfident.")
    print("  Calibration fixes the probabilities; ranking quality (AUC) barely changes.")


if __name__ == "__main__":
    main()
