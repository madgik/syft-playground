"""
federated_kmeans.py
-------------------
k-means clustering via federated EM (sum & count per cluster).

    poetry run python federated_kmeans.py
"""

from __future__ import annotations
import numpy as np
import syft as sy
from fed_utils import SITES, get_assets        # single source of truth


# ----------------------------------------------------------------------
# local E-step: return partial sums & counts
def _e_step(asset):
    @sy.syft_function_single_use(df=asset)
    def step(df, centers):
        import numpy as _np
        X = df.values
        d2 = ((X[:, None] - centers) ** 2).sum(2)     # squared Euclidean
        lbl = _np.argmin(d2, axis=1)

        K = len(centers)
        sums   = _np.zeros_like(centers)
        counts = _np.zeros(K, dtype=int)

        for k in range(K):
            m = lbl == k
            if m.any():
                sums[k]   = X[m].sum(0)
                counts[k] = m.sum()
        return sums, counts
    return step


# ----------------------------------------------------------------------
def kmeans_federated(k: int = 3, iters: int = 10, sites=SITES) -> np.ndarray:
    # one login per site
    assets, _ = get_assets()
    dim = assets[0].data.shape[1]

    rng = np.random.default_rng(0)
    centers = rng.normal(size=(k, dim))

    for _ in range(iters):
        sum_acc = np.zeros_like(centers)
        cnt_acc = np.zeros(k, dtype=int)

        for asset in assets:
            sums, cnts = _e_step(asset)(df=asset, centers=centers, blocking=True)
            sum_acc += np.asarray(sums)
            cnt_acc += np.asarray(cnts)

        mask = cnt_acc > 0
        centers[mask] = sum_acc[mask] / cnt_acc[mask][:, None]   # M-step

    return centers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("Cluster centers:\n", kmeans_federated())
