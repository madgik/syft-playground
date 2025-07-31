#!/usr/bin/env python3
from __future__ import annotations
import numpy as np
import syft as sy
from fed_utils import SITES, get_assets, to_native


# ----------------------------------------------------------------------
def _stats_fn(asset):
    @sy.syft_function_single_use(df=asset)
    def stats(df):
        import numpy as _np
        x, y = df["x"], df["y"]
        return len(df), float(x.sum()), float(y.sum()), \
               float((x ** 2).sum()), float((y ** 2).sum()), float((x * y).sum())
    return stats


def pearson(sites=SITES):
    # get_assets now returns (assets, dim) -> unpack the first element
    assets, _ = get_assets()

    # compile the remote function once per asset
    stats_fns = [_stats_fn(asset) for asset in assets]

    results = []
    for fn, asset in zip(stats_fns, assets):
        res = fn(df=asset, blocking=True)
        results.append(to_native(res))

    n, sx, sy, sxx, syy, sxy = np.sum(results, axis=0)
    r = (sxy - sx * sy / n) / (
        (sxx - sx ** 2 / n) ** 0.5 * (syy - sy ** 2 / n) ** 0.5
    )
    return int(n), float(r)


# ----------------------------------------------------------------------
if __name__ == "__main__":
    total_rows, corr = pearson()
    print(f"Total rows: {total_rows}")
    print(f"Pearson r : {corr:.6f}")
