#!/usr/bin/env python3
import syft as sy, numpy as np
from fed_utils import SITES, get_assets, to_native


def train(epochs=20, lr=.1, batch=32, sites=SITES):
    assets, dim = get_assets("y")          # unpack both values
    w = np.zeros((dim, 1))

    for _ in range(epochs):
        grads = []
        for asset in assets:
            grad_fn = _grad_fn(asset, batch)
            g, _ = grad_fn(df=asset, w=w.tolist(), blocking=True)
            grads.append(np.asarray(g))
        w -= lr * np.mean(np.stack(grads), axis=0)

    return w.flatten()

def _grad_fn(asset,b):
    @sy.syft_function_single_use(df=asset)
    def g(df,w):
        import numpy as _np
        X=_np.c_[ _np.ones(len(df)), df.drop("y",1) ].values
        y=df["y"].values.reshape(-1,1)
        idx=_np.random.choice(len(df),b,False); X,y=X[idx],y[idx]
        p=1/(1+_np.exp(-X@w)); return (X.T@(p-y))/b, len(idx)
    return g

if __name__ == "__main__":
    print("Weights:", train())
