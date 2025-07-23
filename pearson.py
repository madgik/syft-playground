#!/usr/bin/env python3
"""
pearson.py
----------
Privacy‑preserving (federated) Pearson correlation for PySyft 0.9.x.

API
---
    from pearson import compute_global_pearson
    total_rows, r = compute_global_pearson(num_clients=3, base_port=8080)
"""

from __future__ import annotations
from typing import Tuple, List, Any

import numpy as np
import syft as sy


# ----------------------------------------------------------------------
# helper: per‑site stats function factory
def _stats_fn(asset):
    """Return a single‑use Syft function that emits
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
def compute_global_pearson(num_clients: int, base_port: int) -> Tuple[int, float]:
    """Return (total_row_count, Pearson r) computed across all datasites."""

    results: List[Tuple[int, float, float, float, float, float]] = []

    for i in range(num_clients):
        client = sy.login(
            port=base_port + i,
            email="info@openmined.org",
            password="changethis",
        )
        client.refresh()

        try:
            ds = next(iter(client.datasets))
        except StopIteration:
            raise RuntimeError(
                f"No dataset on org{i + 1}; upload data first."
            )

        asset = ds.assets[0]
        res   = _stats_fn(asset)(df=asset, blocking=True)
        results.append(_to_native(res))

    totals = np.sum(np.array(results, dtype=float), axis=0)
    N, sx, sy_, sxx, syy, sxy = totals

    numerator = sxy - (sx * sy_) / N
    var_x     = sxx - (sx ** 2) / N
    var_y     = syy - (sy_ ** 2) / N
    r         = numerator / ((var_x ** 0.5) * (var_y ** 0.5))

    return int(N), float(r)
