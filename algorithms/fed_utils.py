#!/usr/bin/env python3
"""
fed_utils.py
------------
Shared helpers + single source of truth for federation settings.
Edit SITES / EMAIL / PASSWORD here once.
"""

from __future__ import annotations
from typing import List, Dict, Any
import json, os
import numpy as np
import syft as sy

# ------------------------------------------------------------------ config
# Option 1 : hard-code
SITES: List[Dict[str, str | int]] = [
    {"host": "gaia2-vm-2.imsi.athenarc.gr", "port": 8090},
    {"host": "gaia2-vm-3.imsi.athenarc.gr", "port": 8090},
]

# Option 2 : override with JSON env var FED_SITES='[{"host":...,"port":...}]'
if os.getenv("FED_SITES"):
    SITES = json.loads(os.environ["FED_SITES"])

EMAIL, PASSWORD = "info@openmined.org", "changethis"

# ------------------------------------------------------------------ helpers
def to_native(obj: Any):
    """ActionObject → Python."""
    if isinstance(obj, tuple):
        return obj
    if hasattr(obj, "resolve"):
        return obj.resolve()
    if hasattr(obj, "get"):
        return obj.get()
    raise TypeError(f"Cannot convert {type(obj)}")

def get_assets(label_col: str | None = None):
    """
    Yield (client, asset) for every site.
    If label_col is given, also return the feature dimension (d).
    """
    dim = None
    for site in SITES:
        url = f"http://{site['host']}:{site['port']}"
        c   = sy.login(email=EMAIL, password=PASSWORD, url=url)
        c.refresh()
        try:
            ds = next(iter(c.datasets))
        except StopIteration:
            raise RuntimeError(f"No dataset on {url}")
        asset = ds.assets[0]
        if label_col and dim is None:
            dim = asset.data.columns.drop(label_col).size + 1  # +bias
        yield c, asset
    if label_col:
        return dim
