#!/usr/bin/env python3
"""
federated_logreg.py  – synchronous FedAvg for binary logistic regression
"""

from __future__ import annotations
import numpy as np, syft as sy
from fed_utils import SITES, get_assets          # one source of config

EMAIL, PASSWORD = "info@openmined.org", "changethis"


# ----------------------------------------------------------------------
def _make_grad(asset, batch_sz: int):
    """Compile once per asset; returns dL/dw for a mini-batch."""
    @sy.syft_function_single_use(df=asset)
    def grad(df, w):
        import numpy as _np
        X = _np.c_[ _np.ones(len(df)), df.drop("y", axis=1).values ]   # bias
        y = df["y"].values.reshape(-1, 1)

        idx     = _np.random.choice(len(df), batch_sz, replace=False)
        Xb, yb  = X[idx], y[idx]

        p       = 1 / (1 + _np.exp(-Xb @ w))
        g       = (Xb.T @ (p - yb)) / batch_sz
        return g
    return grad


# ----------------------------------------------------------------------
def train_logreg_fed(
    epochs: int = 20,
    lr: float = 0.1,
    batch: int = 32,
    sites = SITES,
) -> np.ndarray:

    assets, dim = get_assets("y")            # one login per site
    w = np.zeros((dim, 1))

    grad_fns = [_make_grad(a, batch) for a in assets]    # cache functions

    for _ in range(epochs):
        grads = []
        for fn, asset in zip(grad_fns, assets):
            g = fn(df=asset, w=w.tolist(), blocking=True)   # send list
            grads.append(np.asarray(g))

        w -= lr * np.mean(np.stack(grads), axis=0)

    return w.flatten()


# ----------------------------------------------------------------------
if __name__ == "__main__":
    weights = train_logreg_fed(epochs=25, lr=0.05, batch=32)
    print("Final weights:", weights)
