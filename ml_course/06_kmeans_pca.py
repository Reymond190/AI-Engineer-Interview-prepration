"""
06 - Unsupervised: k-means, choosing k, and PCA
===============================================

Interview topics
----------------
- k-means (Lloyd): assign each point to the nearest centroid, move each
  centroid to the mean of its points, repeat. Minimises inertia (within-
  cluster sum of squares). Each step can only lower it, so it converges, but
  to a LOCAL minimum that depends on the init -> run several inits (n_init)
  and keep the lowest inertia.
- k-means++ init: pick the next centroid with probability proportional to
  squared distance from the nearest existing one -> spread-out seeds,
  provably O(log k)-competitive, fewer bad runs (section 2).
- k-means assumes convex, roughly spherical, similar-size clusters and
  Euclidean distance (scale features!). It fails on rings/moons -> DBSCAN
  (density), spectral clustering, or Gaussian mixtures (soft, elliptical).
- Choosing k: elbow on inertia (inertia ALWAYS falls with k, look for the
  bend), silhouette score (cohesion vs separation, -1..1, pick the max),
  or a downstream metric. With labels: adjusted Rand index (ARI) to compare.
- PCA: find orthogonal directions of maximum variance. Centre the data, SVD
  X = U S V^T; rows of V^T are the components; explained variance of
  component i = S_i^2 / (n-1). Equivalent to the eigenvectors of the
  covariance matrix. Uses: compression, de-noising, visualisation,
  decorrelating features, speeding up downstream models.
- PCA is linear and variance-based: standardise first or the feature with
  the biggest units wins (section 5); high variance != useful for the label
  (it's unsupervised). For non-linear structure: kernel PCA, t-SNE / UMAP
  (visualisation only: distances between far clusters are not meaningful).

Run:  python ml_course/06_kmeans_pca.py [--quick]
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import load_digits, load_wine, make_blobs, make_moons
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from common import kv, parse_args, section, table


def kmeans(x: np.ndarray, k: int, rng: np.random.Generator, init: str = "random", iters: int = 100) -> tuple[np.ndarray, np.ndarray, float]:
    if init == "random":
        c = x[rng.choice(len(x), k, replace=False)]
    else:  # k-means++
        c = [x[rng.integers(len(x))]]
        for _ in range(k - 1):
            d2 = np.min(((x[:, None] - np.array(c)[None]) ** 2).sum(-1), axis=1)
            c.append(x[rng.choice(len(x), p=d2 / d2.sum())])
        c = np.array(c)
    for _ in range(iters):
        labels = ((x[:, None] - c[None]) ** 2).sum(-1).argmin(1)
        new = np.array([x[labels == j].mean(0) if np.any(labels == j) else c[j] for j in range(k)])
        if np.allclose(new, c):
            break
        c = new
    inertia = float(((x - c[labels]) ** 2).sum())
    return c, labels, inertia


def main() -> None:
    args = parse_args(__doc__)
    rng = np.random.default_rng(args.seed)

    section("1. k-means from scratch vs sklearn")
    x, y = make_blobs(n_samples=1000, centers=5, cluster_std=1.0, random_state=args.seed)
    tries = [kmeans(x, 5, rng, init="++") for _ in range(10)]   # n_init=10: keep the lowest inertia
    _, labels, inertia = min(tries, key=lambda t: t[2])
    sk = KMeans(5, n_init=10, random_state=0).fit(x)
    kv("kmeans", ours_inertia=inertia, sklearn_inertia=float(sk.inertia_), ours_ari=float(adjusted_rand_score(y, labels)),
       single_run_worst=max(t[2] for t in tries))
    print("  Even k-means++ can land in a local minimum on one run (worst single run above); n_init fixes it.")

    section("2. Init matters: random vs k-means++ over many seeds (single run each, no n_init)")
    x, y = make_blobs(n_samples=1500, centers=12, cluster_std=0.6, random_state=3)
    runs = 20 if args.quick else 100
    res = {init: np.array([kmeans(x, 12, np.random.default_rng(s), init=init)[2] for s in range(runs)])
           for init in ["random", "++"]}
    best = min(r.min() for r in res.values())
    rows = [{"init": init, "best inertia": float(r.min()), "median": float(np.median(r)), "worst": float(r.max()),
             "runs within 5% of overall best": f"{int((r < best * 1.05).sum())}/{runs}"} for init, r in res.items()]
    table(rows)
    kv("init", random_median=rows[0]["median"], pp_median=rows[1]["median"])
    print("  Random init often puts two seeds in one blob and none in another; Lloyd's steps can't fix that.")

    section("3. Choosing k: elbow (inertia) and silhouette")
    x, y = make_blobs(n_samples=800, centers=4, cluster_std=0.8, random_state=args.seed)
    rows = []
    for k in range(2, 9):
        m = KMeans(k, n_init=10, random_state=0).fit(x)
        rows.append({"k": k, "inertia": float(m.inertia_), "silhouette": float(silhouette_score(x, m.labels_))})
    table(rows)
    best_k = max(rows, key=lambda r: r["silhouette"])["k"]
    kv("choose-k", true_k=4, silhouette_best_k=best_k)
    print("  Inertia falls for every extra k (k = n gives 0), so it can't pick k alone; look for the bend.")

    section("4. Where k-means fails: non-convex clusters")
    xm, ym = make_moons(600, noise=0.06, random_state=args.seed)
    km = KMeans(2, n_init=10, random_state=0).fit(xm)
    db = DBSCAN(eps=0.2, min_samples=5).fit(xm)
    kv("nonconvex", kmeans_ari=float(adjusted_rand_score(ym, km.labels_)), dbscan_ari=float(adjusted_rand_score(ym, db.labels_)))
    print("  k-means cuts the moons with a straight line (Voronoi cells are convex); DBSCAN follows density.")

    section("5. PCA via SVD from scratch vs sklearn")
    xd, yd = load_digits(return_X_y=True)
    xc = xd - xd.mean(0)
    _, s, vt = np.linalg.svd(xc, full_matrices=False)
    var = s ** 2 / (len(xd) - 1)
    ratio = var / var.sum()
    skp = PCA().fit(xd)
    sign_free = float(np.abs(np.abs(vt[:10]) - np.abs(skp.components_[:10])).max())  # components are defined up to sign
    kv("pca", max_component_diff=sign_free, max_ratio_diff=float(np.abs(ratio - skp.explained_variance_ratio_).max()))
    cum = np.cumsum(ratio)
    rows = []
    for k in [2, 10, 20, int(np.searchsorted(cum, 0.95)) + 1, 64]:
        recon = xc @ vt[:k].T @ vt[:k]
        rows.append({"components": k, "variance kept": float(cum[k - 1]), "reconstruction MSE": float(np.mean((recon - xc) ** 2))})
    table(rows)
    kv("pca-95", components_for_95=int(np.searchsorted(cum, 0.95)) + 1, of_dims=xd.shape[1])

    section("6. PCA without scaling: the biggest-unit feature wins")
    xw, _ = load_wine(return_X_y=True)
    names = load_wine().feature_names
    raw = PCA(1).fit(xw)
    std = PCA(1).fit(StandardScaler().fit_transform(xw))
    top_raw, top_std = int(np.argmax(np.abs(raw.components_[0]))), int(np.argmax(np.abs(std.components_[0])))
    print(f"  feature std devs range {xw.std(0).min():.2f} .. {xw.std(0).max():.0f} ('{names[int(np.argmax(xw.std(0)))]}')")
    print(f"  raw PC1:          {raw.explained_variance_ratio_[0]:.1%} of variance, dominated by '{names[top_raw]}'"
          f" (weight {raw.components_[0][top_raw]:+.3f})")
    print(f"  standardised PC1: {std.explained_variance_ratio_[0]:.1%} of variance, top feature '{names[top_std]}'"
          f" (weight {std.components_[0][top_std]:+.3f}); weights spread over many features")
    kv("pca-scaling", raw_pc1_ratio=float(raw.explained_variance_ratio_[0]), raw_top_weight=float(abs(raw.components_[0][top_raw])),
       std_top_weight=float(abs(std.components_[0][top_std])))


if __name__ == "__main__":
    main()
