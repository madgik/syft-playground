#!/usr/bin/env python3
from __future__ import annotations
import syft as sy
import numpy as np
from fed_utils import SITES, get_assets, to_native

def _stats_fn(asset):
    @sy.syft_function_single_use(df=asset)
    def stats(df):
        import numpy as _np
        x,y = df["x"], df["y"]
        return len(df), float(x.sum()), float(y.sum()), \
               float((x**2).sum()), float((y**2).sum()), float((x*y).sum())
    return stats

def pearson(sites=SITES):
    results = []
    for _, asset in get_assets():
        res = _stats_fn(asset)(df=asset, blocking=True)
        results.append(to_native(res))
    n,sx,sy,sxx,syy,sxy = np.sum(results, axis=0)
    r = (sxy - sx*sy/n) / ((sxx - sx**2/n)**.5 * (syy - sy**2/n)**.5)
    return int(n), float(r)

if __name__ == "__main__":
    print("Total rows, r =", pearson())
