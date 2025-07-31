#!/usr/bin/env python3
from __future__ import annotations
import numpy as np, syft as sy
from fed_utils import SITES, get_assets, to_native

def _e_step(asset):
    @sy.syft_function_single_use(df=asset)
    def step(df,centers):
        import numpy as _np
        X=df.values; K=len(centers)
        lbl=_np.argmin(((X[:,None]-centers)**2).sum(2),1)
        sums=_np.zeros_like(centers); cnt=_np.zeros(K,int)
        for k in range(K):
            m=lbl==k
            if m.any():
                sums[k]=X[m].sum(0); cnt[k]=m.sum()
        return sums,cnt
    return step

def kmeans(k=3,iters=10, sites=SITES):
    d = get_assets().__next__()[1].data.shape[1]
    C = np.random.default_rng(0).normal(size=(k,d))
    for _ in range(iters):
        s_acc=np.zeros_like(C); c_acc=np.zeros(k,int)
        for _,asset in get_assets():
            s,c=_e_step(asset)(df=asset,centers=C,blocking=True)
            s_acc+=np.asarray(s); c_acc+=np.asarray(c)
        m=c_acc>0; C[m]=s_acc[m]/c_acc[m,None]
    return C

if __name__ == "__main__":
    print("Centers:\n", kmeans())
