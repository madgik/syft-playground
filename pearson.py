#!/usr/bin/env python3
"""
pearson.py
----------
Privacy-preserving (federated) Pearson correlation for PySyft 0.9.x.

How to call
-----------
1) Legacy (all-local) style
       total_rows, r = compute_global_pearson(num_clients=3, base_port=8080)

2) Explicit endpoint list (remote machines)
       SITES = [
           {"host": "localhost",                       "port": 8080},
           {"host": "gaia2-vm-2.imsi.athenarc.gr",     "port": 8080},
           {"host": "gaia2-vm-3.imsi.athenarc.gr",     "port": 8090},
       ]
       total_rows, r = compute_global_pearson(sites=SITES)
"""

from __future__ import annotations
from typing import Tuple, List, Dict, Any

import numpy as np
import syft as sy


# ----------------------------------------------------------------------
# helper: per-site stats function factory
def _stats_fn(asset):
    """Return a single-use Syft function that emits
       (n, Σx, Σy, Σx², Σy², Σxy) for its bound DataFrame.
    """
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


def _to_native(obj: Any):
    """Convert ActionObject → Python data."""
    if isinstance(obj, tuple):
        return obj
    if hasattr(obj, "resolve"):
        return obj.resolve()
    if hasattr(obj, "get"):
        return obj.get()
    raise TypeError(f"Cannot convert {type(obj)} to Python")


# ----------------------------------------------------------------------
# public API
def compute_global_pearson(
    num_clients: int | None = None,
    base_port: int | None = None,
    sites: List[Dict[str, str | int]] | None = None,
) -> Tuple[int, float]:
    """
    Return (total_row_count, Pearson r) computed across all datasites.

    Choose ONE of the call styles:
        • num_clients + base_port  –> local consecutive ports
        • sites = [{"host": ..., "port": ...}, ...]  –> explicit list
    """
    # -------------------------------------------------- build endpoint list
    if sites is not None:
        ENDPOINTS = sites
    elif num_clients is not None and base_port is not None:
        ENDPOINTS = [
            {"host": "localhost", "port": base_port + i}
            for i in range(num_clients)
        ]
    else:
        raise ValueError("Provide either (num_clients, base_port) OR sites list")

    # -------------------------------------------------- main loop
    results: List[Tuple[int, float, float, float, float, float]] = []

    for idx, ep in enumerate(ENDPOINTS, 1):
        url = f"http://{ep['host']}:{ep['port']}"
        client = sy.login(email="info@openmined.org",
                          password="changethis",
                          url=url)
        client.refresh()

        try:
            ds = next(iter(client.datasets))
        except StopIteration:
            raise RuntimeError(f"No dataset on site {url}; upload data first.")

        asset = ds.assets[0]
        res   = _stats_fn(asset)(df=asset, blocking=True)
        results.append(_to_native(res))

    # -------------------------------------------------- aggregate client-side
    totals = np.sum(np.array(results, dtype=float), axis=0)
    N, sx, sy_, sxx, syy, sxy = totals

    numerator = sxy - (sx * sy_) / N
    var_x     = sxx - (sx ** 2) / N
    var_y     = syy - (sy_ ** 2) / N
    r         = numerator / ((var_x ** 0.5) * (var_y ** 0.5))

    return int(N), float(r)
