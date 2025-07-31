#!/usr/bin/env python3
"""
federated_kmeans.py
-------------------
k-means clustering via federated EM (sum & count per cluster).

Call
----
    from federated_kmeans import kmeans_federated, SITES
    centers = kmeans_federated(k=3, iters=10, sites=SITES)
"""

from __future__ import annotations
from typing import List, Dict, Any
import numpy as np
import syft as sy

SITES: List[Dict[str, str | int]] = [
    {"host": "gaia2-vm-2.imsi.athenarc.gr", "port": 8080},
    {"host": "gaia2-vm-3.imsi.athenarc.gr", "port": 8090},
]
EMAIL, PASSWORD = "info@openmined.org", "changethis"


def _local_e_step(asset, centers):
    """Return (sum_vecs, counts) for current centers."""
    @sy.syft_function_single_use(df=asset, centers=None)
    def e_step(df, centers=None):
        import numpy as _np
        X = df.values
        dists = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = _np.argmin(dists, axis=1)

        sums   = _np.zeros_like(centers)
        counts = _np.zeros(len(centers), dtype=int)
        for k in range(len(centers)):
            mask = labels == k
            if mask.any():
                sums[k]   = X[mask].sum(axis=0)
                counts[k] = mask.sum()
        return sums, counts
    return e_step


def kmeans_federated(
    k: int,
    iters: int,
    sites: List[Dict[str, str | int]],
) -> np.ndarray:
    # login + asset handles
    assets, dim = [], None
    for ep in sites:
        url = f"http://{ep['host']}:{ep['port']}"
        c = sy.login(email=EMAIL, password=PASSWORD, url=url)
        ds = next(iter(c.datasets))
        assets.append(ds.assets[0])
        if dim is None:
            dim = ds.assets[0].data.shape[1]

    rng = np.random.default_rng(0)
    centers = rng.normal(size=(k, dim))  # random init

    for _ in range(iters):
        sum_acc = np.zeros_like(centers)
        cnt_acc = np.zeros(k, dtype=int)

        for asset in assets:
            s, c = _local_e_step(asset, centers)(df=asset, centers=centers, blocking=True)
            sum_acc += np.asarray(s)
            cnt_acc += np.asarray(c)

        # avoid division by zero
        mask = cnt_acc > 0
        centers[mask] = sum_acc[mask] / cnt_acc[mask][:, None]

    return centers

if __name__ == "__main__":
    centers = kmeans_federated(k=3, iters=15, sites=SITES)
    print("cluster centers:\n", centers)