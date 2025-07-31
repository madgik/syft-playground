#!/usr/bin/env python3
"""
federated_kmeans.py
-------------------
k-means clustering via one E-step per datasite and a client-side M-step.

Run
----
    poetry run python federated_kmeans.py          # uses SITES from fed_utils
"""

from __future__ import annotations
import numpy as np
import syft as sy
from fed_utils import SITES, get_assets, to_native


# ----------------------------------------------------------------------
# local E-step: given current centres, return (partial_sums, counts)
def _make_e_step(asset):
    @sy.syft_function_single_use(df=asset)
    def e_step(df, centers):
        import numpy as _np
        X       = df.values
        centers = _np.asarray(centers)          # list → ndarray
        lbl     = _np.argmin(((X[:, None] - centers) ** 2).sum(2), axis=1)

        K       = len(centers)
        sums    = _np.zeros_like(centers)
        counts  = _np.zeros(K, dtype=int)
        for k in range(K):
            m = lbl == k
            if m.any():
                sums[k]   = X[m].sum(0)
                counts[k] = m.sum()
        return sums, counts
    return e_step


# ----------------------------------------------------------------------
def kmeans_federated(k: int = 3, iters: int = 10, sites=SITES) -> np.ndarray:
    # one login per site (only once)
    assets, _ = get_assets()
    dim       = assets[0].data.shape[1]

    # cache a compiled e_step function per site
    e_steps = [_make_e_step(asset) for asset in assets]

    # random initial centres
    rng      = np.random.default_rng(0)
    centers  = rng.normal(size=(k, dim))

    for _ in range(iters):
        sum_acc = np.zeros_like(centers)
        cnt_acc = np.zeros(k, dtype=int)

        for fn, asset in zip(e_steps, assets):
            sums, cnts = fn(df=asset, centers=centers.tolist(), blocking=True)
            sum_acc += np.asarray(sums)
            cnt_acc += np.asarray(cnts)

        mask          = cnt_acc > 0
        centers[mask] = sum_acc[mask] / cnt_acc[mask][:, None]  # M-step

    return centers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("Cluster centres:\n", kmeans_federated())
