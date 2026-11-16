"""
00 - Train/val/test, cross-validation and data leakage
======================================================

Interview topics
----------------
- Why three splits: train fits weights, validation picks hyperparameters /
  models / thresholds, test is touched ONCE to report. Every decision made by
  looking at a set turns it into training data for that decision ("selection
  on the test set" -> optimistic number). Section 1 measures how optimistic.
- k-fold CV: every point is validated exactly once; the mean has lower
  variance than one split, and the std across folds tells you how much to
  trust a difference between models. Stratify for classification. Nested CV
  (outer loop for the estimate, inner loop for tuning) if you tune AND report
  on the same data.
- Leakage = information from outside the training fold reaches the model.
    * preprocessing fitted on all data (feature selection, target encoding,
      imputation stats, scaling; scaling is usually small, selection is huge)
      -> fix: put EVERY fitted step in a Pipeline that is fit inside each fold.
    * group leakage: several rows per patient/user/session split across folds;
      the model recognises the entity, not the pattern -> GroupKFold.
    * time leakage: random split of time-ordered data uses the future ->
      TimeSeriesSplit / walk-forward validation with a gap.
    * target leakage: a feature that is only known after the label (e.g.
      "refund_issued" when predicting complaints).
- Red flag: validation much better than expected, or a single feature with
  near-perfect importance. Investigate before celebrating.

Run:  python ml_course/00_splits_cv_leakage.py [--quick]
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold, TimeSeriesSplit, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer

from common import kv, parse_args, section, table


def selection_on_test(rng: np.random.Generator, n_models: int) -> tuple[float, float]:
    """Pure-noise labels: no model can beat 50%. Pick the best of many by test score, then re-test on fresh data."""
    n, p = 200, 50
    x = rng.normal(size=(3 * n, p))
    y = rng.integers(0, 2, size=3 * n)
    xtr, ytr, xte, yte, xnew, ynew = x[:n], y[:n], x[n:2 * n], y[n:2 * n], x[2 * n:], y[2 * n:]
    best, best_model = -1.0, None
    for _ in range(n_models):
        cols = rng.choice(p, size=5, replace=False)
        m = LogisticRegression(max_iter=500).fit(xtr[:, cols], ytr)
        s = m.score(xte[:, cols], yte)
        if s > best:
            best, best_model = s, (m, cols)
    m, cols = best_model
    return best, m.score(xnew[:, cols], ynew)


def main() -> None:
    args = parse_args(__doc__)
    rng = np.random.default_rng(args.seed)

    section("1. Selecting a model on the TEST set makes the test score optimistic")
    best, fresh = selection_on_test(rng, n_models=50 if args.quick else 200)
    print("  Labels are coin flips, so the true accuracy of every model is 0.5.")
    kv("selection", best_on_test=best, fresh_holdout=fresh)
    print("  The 'best' model only won the lottery on that test set. This is why the test set is used once.")

    section("2. Leakage: feature selection done BEFORE cross-validation")
    # Classic example (Elements of Statistical Learning, 7.10.2): many noise features, random labels.
    n, p = 100, 5000
    x = rng.normal(size=(n, p))
    y = rng.integers(0, 2, size=n)
    cv = StratifiedKFold(5, shuffle=True, random_state=args.seed)
    xsel = SelectKBest(f_classif, k=20).fit_transform(x, y)          # looks at ALL labels: leak
    leaky = cross_val_score(LogisticRegression(max_iter=1000), xsel, y, cv=cv).mean()
    pipe = make_pipeline(SelectKBest(f_classif, k=20), LogisticRegression(max_iter=1000))
    honest = cross_val_score(pipe, x, y, cv=cv).mean()                 # selection refit inside each fold
    kv("leak-select", outside_cv=leaky, inside_pipeline=honest)
    print("  Same data, same model. Only where the selector was fitted differs.")

    section("3. Group leakage: several rows per entity")
    # 60 'patients', 10 rows each. Label is per patient and random; each patient has a fingerprint in the features.
    groups = np.repeat(np.arange(60), 10)
    fingerprint = rng.normal(size=(60, 10))
    xg = fingerprint[groups] + 0.3 * rng.normal(size=(600, 10))
    yg = rng.integers(0, 2, size=60)[groups]
    knn = KNeighborsClassifier(1)
    random_kf = cross_val_score(knn, xg, yg, cv=KFold(5, shuffle=True, random_state=args.seed)).mean()
    group_kf = cross_val_score(knn, xg, yg, cv=GroupKFold(5), groups=groups).mean()
    kv("leak-group", kfold=random_kf, group_kfold=group_kf)
    print("  KFold puts a patient's rows on both sides; 1-NN just recognises the patient.")

    section("4. Time leakage: shuffled split on a time series")
    # Random walk; predict tomorrow's level from today's lags. Shuffled CV interpolates between known neighbours.
    t = 1500
    series = np.cumsum(rng.normal(size=t + 5))
    lags = np.stack([series[i:i + t] for i in range(5)], axis=1)
    target = series[5:t + 5]
    rf = RandomForestRegressor(n_estimators=50, min_samples_leaf=5, random_state=args.seed, n_jobs=-1)
    shuffled = cross_val_score(rf, lags, target, cv=KFold(5, shuffle=True, random_state=0), scoring="r2").mean()
    walk_fwd = cross_val_score(rf, lags, target, cv=TimeSeriesSplit(5), scoring="r2").mean()
    kv("leak-time", shuffled_r2=shuffled, walk_forward_r2=walk_fwd)
    print("  Trees can't extrapolate to levels never seen before; shuffled CV hides that because the future")
    print("  is in the training folds. Walk-forward shows the real, much lower R^2.")

    section("5. One split vs k-fold: how noisy is a single validation score?")
    xb, yb = load_breast_cancer(return_X_y=True)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    seeds = range(10 if args.quick else 30)
    single, kfold = [], []
    for s in seeds:
        a, b, c, d = train_test_split(xb, yb, test_size=0.2, stratify=yb, random_state=s)
        single.append(model.fit(a, c).score(b, d))
        kfold.append(cross_val_score(model, xb, yb, cv=StratifiedKFold(5, shuffle=True, random_state=s)).mean())
    table([{"estimate": "single 80/20 split", "mean": float(np.mean(single)), "std over seeds": float(np.std(single)),
            "min": float(np.min(single)), "max": float(np.max(single))},
           {"estimate": "5-fold CV mean", "mean": float(np.mean(kfold)), "std over seeds": float(np.std(kfold)),
            "min": float(np.min(kfold)), "max": float(np.max(kfold))}])
    kv("cv-variance", single_std=float(np.std(single)), kfold_std=float(np.std(kfold)))
    print("  A 1-2 point difference between two models on one split is often just the split.")


if __name__ == "__main__":
    main()
