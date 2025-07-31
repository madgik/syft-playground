#!/usr/bin/env python3
"""
pearson.py
----------
Compute a privacy-preserving (federated) Pearson correlation with PySyft 0.9.

Two modes
---------
1. Remote mode (default) – uses the hard-coded SITES list below.
2. Local test        – `python pearson.py --local N BASE_PORT`
                       connects to localhost:BASE_PORT..BASE_PORT+N-1.

Edit the SITES list to match your real datasite endpoints.
"""

from __future__ import annotations
from typing import List, Dict, Tuple, Any
import argparse

import numpy as np
import syft as sy


# ----------------------------------------------------------------------
# 1. Hard-coded endpoints  (edit as needed)
SITES: List[Dict[str, str | int]] = [
    {"host": "gaia2-vm-2.imsi.athenarc.gr", "port": 8090},
    {"host": "gaia2-vm-3.imsi.athenarc.gr", "port": 8090},
]
EMAIL = "info@openmined.org"
PASSWORD = "changethis"


# ----------------------------------------------------------------------
# helper: per-site stats function
def _stats_fn(asset):
    @sy.syft_function_single_use(df=asset)
    def local_stats(df):
        n   = len(df)
        sx  = float(df["x"].sum())
        sy_ = float(df["y"].sum())
        sxx = float((df["x"] ** 2).sum())
        syy = float((df["y"] ** 2).sum())
        sxy = float((df["x"] * df["y"]).sum())
        return n, sx, sy_, sxx, syy, sxy
    return local_stats


def _to_tuple(obj: Any):
    if isinstance(obj, tuple):
        return obj
    if hasattr(obj, "resolve"):
        return obj.resolve()
    if hasattr(obj, "get"):
        return obj.get()
    raise TypeError(f"Unexpected return type: {type(obj)}")


# ----------------------------------------------------------------------
def compute_global_pearson(sites: List[Dict[str, str | int]]) -> Tuple[int, float]:
    """
    Return (total_rows, Pearson r) across all `sites`.
    Each site dict needs {"host": ..., "port": ...}.
    """
    results: List[Tuple[int, float, float, float, float, float]] = []

    for site in sites:
        url = f"http://{site['host']}:{site['port']}"
        client = sy.login(email=EMAIL, password=PASSWORD, url=url)
        client.refresh()

        try:
            ds = next(iter(client.datasets))
        except StopIteration:
            raise RuntimeError(f"No dataset on {url}")

        asset = ds.assets[0]
        res   = _stats_fn(asset)(df=asset, blocking=True)
        results.append(_to_tuple(res))

    totals = np.sum(np.array(results, dtype=float), axis=0)
    N, sx, sy_, sxx, syy, sxy = totals

    num  = sxy - (sx * sy_) / N
    varx = sxx - (sx ** 2) / N
    vary = syy - (sy_ ** 2) / N
    r    = num / ((varx ** 0.5) * (vary ** 0.5))

    return int(N), float(r)


# ----------------------------------------------------------------------
# CLI entry-point
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local",
        metavar=("NUM_CLIENTS", "BASE_PORT"),
        nargs=2,
        type=int,
        help="Test against NUM_CLIENTS servers on localhost starting at BASE_PORT",
    )
    args = parser.parse_args()

    if args.local:
        num_clients, base_port = args.local
        sites = [{"host": "localhost", "port": base_port + i} for i in range(num_clients)]
    else:
        sites = SITES

    total, r = compute_global_pearson(sites)
    print(f"Total rows: {total}")
    print(f"Pearson r : {r:.6f}")


if __name__ == "__main__":
    main()
