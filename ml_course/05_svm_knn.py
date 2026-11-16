"""
05 - SVMs, the kernel trick, kNN and the curse of dimensionality
================================================================

Interview topics
----------------
- Linear SVM: find the hyperplane with the widest MARGIN. Soft margin =
  minimise  0.5||w||^2 + C * sum hinge(y_i (w.x_i + b)),  hinge(m) = max(0, 1-m).
  Only points on or inside the margin (support vectors) affect the solution.
  Large C -> fewer violations allowed, narrower margin, more variance;
  small C -> wider margin, more support vectors, more bias.
- Hinge vs logistic loss: both are convex upper bounds on the 0/1 loss;
  hinge is exactly 0 once a point is past the margin (sparse solution, no
  probabilities); logistic never reaches 0 (gives calibrated-ish probabilities).
- Kernel trick: the dual only needs dot products x_i.x_j, so replace them with
  k(x_i, x_j) = phi(x_i).phi(x_j) for a feature map phi you never compute.
  Polynomial (x.z + c)^d, RBF exp(-gamma ||x - z||^2) (infinite-dimensional
  phi). RBF gamma = inverse width: large gamma -> each point is an island
  (overfit), small gamma -> almost linear.
- Kernel SVMs scale ~O(n^2)-O(n^3) in samples: fine to ~1e5, then use linear
  models, random features, or trees/nets.
- kNN: no training; predict by the k nearest labelled points. Small k = low
  bias / high variance; large k = smooth. Needs feature SCALING (distance is
  dominated by the largest-unit feature) and suffers from the curse of
  dimensionality: in high d, all points are almost equally far apart, so
  "nearest" stops meaning anything. Same reason distance-based methods need
  good embeddings / dimensionality reduction.

Run:  python ml_course/05_svm_knn.py [--quick]
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_breast_cancer, make_circles, make_moons
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

from common import kv, parse_args, section, table


def pegasos(x: np.ndarray, y: np.ndarray, lam: float, epochs: int, seed: int) -> tuple[np.ndarray, float]:
    """Linear SVM by stochastic subgradient descent on the primal (Pegasos, Shalev-Shwartz 2007). y in {-1, +1}."""
    rng = np.random.default_rng(seed)
    w, b, t = np.zeros(x.shape[1]), 0.0, 0
    for _ in range(epochs):
        for i in rng.permutation(len(y)):
            t += 1
            eta = 1 / (lam * t)
            margin = y[i] * (x[i] @ w + b)
            w *= 1 - eta * lam                    # gradient of the 0.5*lam*||w||^2 term
            if margin < 1:                        # hinge is active: point inside margin or misclassified
                w += eta * y[i] * x[i]
                b += eta * y[i]
    return w, b


def main() -> None:
    args = parse_args(__doc__)
    rng = np.random.default_rng(args.seed)

    section("1. Linear SVM from scratch (hinge loss, subgradient) vs sklearn")
    x, y = load_breast_cancer(return_X_y=True)
    xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.3, stratify=y, random_state=args.seed)
    sc = StandardScaler().fit(xtr)
    xtr, xte = sc.transform(xtr), sc.transform(xte)
    ys_tr, ys_te = 2 * ytr - 1, 2 * yte - 1
    w, b = pegasos(xtr, ys_tr, lam=1e-2, epochs=20 if args.quick else 100, seed=args.seed)
    ours = float((np.sign(xte @ w + b) == ys_te).mean())
    sk = LinearSVC(C=1 / (1e-2 * len(ytr)), max_iter=20000).fit(xtr, ytr)  # C ~ 1/(lambda n)
    margins = ys_tr * (xtr @ w + b)
    kv("linear-svm", ours_acc=ours, sklearn_acc=float(sk.score(xte, yte)),
       on_or_inside_margin=int((margins < 1 + 1e-6).sum()), n_train=len(ytr))
    print("  Only the points with margin < 1 carry gradient; the rest could be deleted without changing w.")

    section("2. C trades margin width against violations (RBF SVM on moons)")
    xm, ym = make_moons(400, noise=0.3, random_state=args.seed)
    xmtr, xmte, ymtr, ymte = train_test_split(xm, ym, test_size=0.5, random_state=args.seed)
    rows = []
    for c in [0.01, 1, 100, 10000]:
        m = SVC(C=c, kernel="rbf", gamma=1.0).fit(xmtr, ymtr)
        rows.append({"C": c, "support vectors": int(m.n_support_.sum()), "train acc": float(m.score(xmtr, ymtr)),
                     "test acc": float(m.score(xmte, ymte))})
    table(rows)
    kv("svm-C", sv_small_C=rows[0]["support vectors"], sv_large_C=rows[-1]["support vectors"])

    section("3. The kernel trick")
    a, z = rng.normal(size=3), rng.normal(size=3)
    def phi(v: np.ndarray) -> np.ndarray:  # explicit feature map of (v.z + 1)^2 in 3-d: 10 features
        quad = [v[i] * v[j] * (np.sqrt(2) if i != j else 1) for i in range(3) for j in range(i, 3)]
        return np.r_[1, np.sqrt(2) * v, quad]
    kernel, explicit = (a @ z + 1) ** 2, phi(a) @ phi(z)
    kv("kernel-identity", kernel=float(kernel), explicit_dot=float(explicit), abs_diff=float(abs(kernel - explicit)))
    print("  (x.z + 1)^2 equals a dot product in a 10-d feature space we never had to build. For RBF that space")
    print("  is infinite-dimensional, which is why the kernel form is the only practical one.")
    xc, yc = make_circles(400, noise=0.1, factor=0.4, random_state=args.seed)
    xctr, xcte, yctr, ycte = train_test_split(xc, yc, test_size=0.5, random_state=args.seed)
    rows = []
    for name, m in [("linear", SVC(kernel="linear")), ("poly d=2", SVC(kernel="poly", degree=2, coef0=1)),
                    ("rbf", SVC(kernel="rbf"))]:
        rows.append({"kernel": name, "test acc on circles": float(m.fit(xctr, yctr).score(xcte, ycte))})
    table(rows)
    kv("kernels", linear=rows[0]["test acc on circles"], rbf=rows[2]["test acc on circles"])
    rows = []
    for g in [0.1, 1, 10, 100, 1000]:
        m = SVC(kernel="rbf", gamma=g).fit(xmtr, ymtr)
        rows.append({"rbf gamma": g, "train acc": float(m.score(xmtr, ymtr)), "test acc": float(m.score(xmte, ymte))})
    table(rows)
    kv("gamma", train_g1000=rows[-1]["train acc"], test_g1000=rows[-1]["test acc"], test_g1=rows[1]["test acc"])

    section("4. kNN: k is the bias/variance knob, and scaling matters")
    rows = []
    for k in [1, 5, 25, 101]:
        m = KNeighborsClassifier(k).fit(xmtr, ymtr)
        rows.append({"k": k, "train acc": float(m.score(xmtr, ymtr)), "test acc": float(m.score(xmte, ymte))})
    table(rows)
    # Same moons, but add a useless feature measured in big units: it dominates every distance.
    big = rng.normal(scale=100, size=(len(xm), 1))
    xb = np.c_[xm, big]
    xbtr, xbte, ybtr, ybte = train_test_split(xb, ym, test_size=0.5, random_state=args.seed)
    raw = KNeighborsClassifier(15).fit(xbtr, ybtr).score(xbte, ybte)
    s = StandardScaler().fit(xbtr)
    scaled = KNeighborsClassifier(15).fit(s.transform(xbtr), ybtr).score(s.transform(xbte), ybte)
    kv("knn-scaling", raw=float(raw), scaled=float(scaled))
    print("  Scaling doesn't remove the useless feature, it just stops it from drowning the useful ones.")

    section("5. Curse of dimensionality")
    rows = []
    for d in [2, 10, 100, 1000]:
        pts = rng.uniform(size=(500, d))
        q = rng.uniform(size=d)
        dist = np.linalg.norm(pts - q, axis=1)
        rows.append({"dims": d, "nearest / farthest distance": float(dist.min() / dist.max()),
                     "std / mean of distances": float(dist.std() / dist.mean())})
    table(rows)
    kv("curse", ratio_d2=rows[0]["nearest / farthest distance"], ratio_d1000=rows[-1]["nearest / farthest distance"])
    # kNN with 2 informative dims + growing noise dims
    rows = []
    for extra in [0, 10, 50, 200]:
        xx = np.c_[xm, rng.normal(size=(len(xm), extra))] if extra else xm
        a_, b_, c_, d_ = train_test_split(xx, ym, test_size=0.5, random_state=args.seed)
        rows.append({"noise dims added": extra, "kNN k=15 test acc": float(KNeighborsClassifier(15).fit(a_, c_).score(b_, d_))})
    table(rows)
    kv("curse-knn", acc_0=rows[0]["kNN k=15 test acc"], acc_200=rows[-1]["kNN k=15 test acc"])
    print("  As d grows the nearest point is barely nearer than the farthest, so neighbours carry little signal.")


if __name__ == "__main__":
    main()
