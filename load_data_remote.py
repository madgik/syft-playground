#!/usr/bin/env python3
"""
load_data_remote.py
-------------------
Upload a toy DataFrame to every running Syft datasite listed below.
"""

from __future__ import annotations
import uuid
from typing import List, Dict

import numpy as np
import pandas as pd
import syft as sy


# ----------------------------------------------------------------------
# 1.  EDIT THESE ENDPOINTS ---------------------------------------------
SITES: List[Dict[str, str | int]] = [
    {"host": "gaia2-vm-1.imsi.athenarc.gr", "port": 8090, "name": "org0"},
    {"host": "gaia2-vm-2.imsi.athenarc.gr", "port": 8090, "name": "org1"},
    {"host": "gaia2-vm-3.imsi.athenarc.gr", "port": 8090, "name": "org2"},
]
# ----------------------------------------------------------------------

EMAIL = "info@openmined.org"
PASSWORD = "changethis"          # default dev password
rng = np.random.default_rng(0)


def upload(client: sy.Client, rows: int, idx: int) -> None:
    df = pd.DataFrame({"x": rng.normal(size=rows), "y": rng.normal(size=rows)})
    tag = f"site{idx+1}-{uuid.uuid4().hex[:6]}"
    asset = sy.Asset(name=f"{tag} asset", data=df, mock=df.head())
    dataset = sy.Dataset(name=tag, asset_list=[asset])
    client.upload_dataset(dataset)
    print(f"Dataset '{tag}' uploaded to {client.name} ({rows} rows)")


def main():
    for idx, site in enumerate(SITES):
        host, port = site["host"], site["port"]
        client = sy.login(host=host, port=port, email=EMAIL, password=PASSWORD)
        rows = 200 + idx * 100          # just for variety
        upload(client, rows, idx)


if __name__ == "__main__":
    main()
