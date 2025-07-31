#!/usr/bin/env python3
from __future__ import annotations
import numpy as np, syft as sy
from fed_utils import SITES, get_assets, to_native

def _grad_fn(asset,b):
    @sy.syft_function_single_use(df=asset)
    def g(df,w):
        import numpy as _np
        X=_np.c_[ _np.ones(len(df)), df.drop("y",1) ].values
        y=df["y"].values.reshape(-1,1)
        idx=_np.random.choice(len(df),b,False); X,y=X[idx],y[idx]
        p=1/(1+_np.exp(-X@w)); return (X.T@(p-y))/b, len(idx)
    return g

def train(epochs=20,lr=.1,batch=32,sites=SITES):
    d = get_assets("y")          # feature dim (+bias)
    w = np.zeros((d,1))
    for _ in range(epochs):
        grads=[]
        for _,asset in get_assets():
            g,_=_grad_fn(asset,batch)(df=asset,w=w.tolist(),blocking=True)
            grads.append(np.asarray(g))
        w -= lr*np.mean(np.stack(grads),0)
    return w.flatten()

if __name__ == "__main__":
    print("Weights:", train())
