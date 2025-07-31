#!/usr/bin/env python3
from __future__ import annotations
import uuid
from typing import List, Dict

import numpy as np
import pandas as pd
import syft as sy

# ----------------------------------------------------------------------
# EDIT your endpoints
SITES: List[Dict[str, str | int]] = [
    {"host": "gaia2-vm-1.imsi.athenarc.gr", "port": 8090},
    {"host": "gaia2-vm-2.imsi.athenarc.gr", "port": 8090},
    {"host": "gaia2-vm-3.imsi.athenarc.gr", "port": 8090},
]
# ----------------------------------------------------------------------

EMAIL = "info@openmined.org"
PASSWORD = "changethis"
rng = np.random.default_rng(0)


def upload(client: sy.Client, rows: int, idx: int) -> None:
    df = pd.DataFrame({"x": rng.normal(size=rows), "y": rng.normal(size=rows)})
    tag = f"site{idx+1}-{uuid.uuid4().hex[:6]}"
    asset = sy.Asset(name=f"{tag} asset", data=df, mock=df.head())
    client.upload_dataset(sy.Dataset(name=tag, asset_list=[asset]))
    print(f"Dataset '{tag}' uploaded to {client.name} ({rows} rows)")


def main():
    for idx, site in enumerate(SITES):
        host, port = site["host"], site["port"]
        url = f"http://{host}:{port}"           # ← build URL here
        client = sy.login(url, email=EMAIL, password=PASSWORD)
        rows = 200 + idx * 100
        upload(client, rows, idx)


if __name__ == "__main__":
    main()
