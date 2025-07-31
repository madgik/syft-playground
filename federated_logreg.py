#!/usr/bin/env python3
"""
federated_logreg.py
-------------------
Mini-batch logistic regression with synchronous federated averaging.

Call
----
    from federated_logreg import train_federated_logreg, SITES
    w = train_federated_logreg(sites=SITES, epochs=20, lr=0.1, batch=32)
"""

from __future__ import annotations
from typing import List, Dict, Any
import numpy as np
import syft as sy

# ----------------------------------------------------------------------
# EDIT ONCE: remote datasite endpoints
SITES: List[Dict[str, str | int]] = [
    {"host": "gaia2-vm-2.imsi.athenarc.gr", "port": 8080},
    {"host": "gaia2-vm-3.imsi.athenarc.gr", "port": 8090},
]
EMAIL, PASSWORD = "info@openmined.org", "changethis"


# ----------------------------------------------------------------------
def _local_grad_fn(asset, batch_sz):
    """Return single-use fn that yields (grad, n) for current weights `w`."""
    @sy.syft_function_single_use(df=asset, w=None)
    def calc_grad(df, w=None):
        import numpy as _np
        X = _np.c_[ _np.ones(len(df)), df.drop("y", axis=1).values ]  # bias term
        y = df["y"].values.reshape(-1, 1)
        idx = _np.random.choice(len(df), size=batch_sz, replace=False)
        Xb, yb = X[idx], y[idx]
        preds = 1 / (1 + _np.exp(-Xb @ w))
        grad  = (Xb.T @ (preds - yb)) / batch_sz
        return grad, len(Xb)
    return calc_grad


def train_federated_logreg(
    sites: List[Dict[str, str | int]],
    epochs: int = 20,
    lr: float = 0.1,
    batch: int = 32,
) -> np.ndarray:
    """Return final weight vector w (including bias at index 0)."""
    # --- login & prepare assets -----------------------------------------
    clients, assets, dim = [], [], None
    for ep in sites:
        url = f"http://{ep['host']}:{ep['port']}"
        c = sy.login(email=EMAIL, password=PASSWORD, url=url)
        ds = next(iter(c.datasets))
        assets.append(ds.assets[0])
        clients.append(c)
        if dim is None:
            dim = ds.assets[0].data.columns.drop("y").size + 1  # +bias

    w = np.zeros((dim, 1))  # init weights

    # --- training loop ---------------------------------------------------
    for _ in range(epochs):
        grads = []
        for asset in assets:
            grad_fn = _local_grad_fn(asset, batch)
            g, _ = grad_fn(df=asset, w=w, blocking=True)
            grads.append(np.asarray(g))

        avg_grad = np.mean(np.stack(grads, axis=0), axis=0)
        w -= lr * avg_grad

    return w.flatten()

if __name__ == "__main__":
    w = train_federated_logreg(sites=SITES, epochs=25, lr=0.05)
    print("final weights:", w)
