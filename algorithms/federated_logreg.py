#!/usr/bin/env python3
"""
federated_logreg.py  – synchronous FedAvg for binary logistic regression
"""

from __future__ import annotations
import numpy as np, syft as sy
from fed_utils import SITES, get_assets

# ----------------------------------------------------------------------
def _make_grad(asset, batch_sz: int):
    """Compile once; returns mini-batch gradient."""

    @sy.syft_function_single_use(df=asset)
    def grad(df, w):
        import numpy as _np
        w = _np.asarray(w).reshape(-1, 1)            # ### FIX ###

        X = _np.c_[ _np.ones(len(df)), df.drop("y", axis=1).values ]
        y = df["y"].values.reshape(-1, 1)

        idx        = _np.random.choice(len(df), batch_sz, replace=False)
        Xb, yb     = X[idx], y[idx]
        p          = 1 / (1 + _np.exp(-Xb @ w))
        g          = (Xb.T @ (p - yb)) / batch_sz
        return g

    return grad


# ----------------------------------------------------------------------
def train_logreg_fed(epochs=20, lr=0.1, batch=32, sites=SITES) -> np.ndarray:
    assets, dim = get_assets("y")          # one login per site
    w = np.zeros((dim, 1))

    grad_fns = [_make_grad(a, batch) for a in assets]

    for _ in range(epochs):
        grads = [
            np.asarray(fn(df=a, w=w.tolist(), blocking=True))
            for fn, a in zip(grad_fns, assets)
        ]
        w -= lr * np.mean(np.stack(grads), axis=0)

    return w.flatten()


# ----------------------------------------------------------------------
if __name__ == "__main__":
    weights = train_logreg_fed(epochs=25, lr=0.05, batch=32)
    print("Final weights:", weights)
