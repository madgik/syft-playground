#!/usr/bin/env python3
"""
load_data_local.py
------------------
Run this on *each* machine that already has a Syft server listening on
<port>.  It creates a small (x, y) DataFrame and uploads it as a dataset.

Usage
-----
    poetry run python load_data_local.py --port 8080 --rows 300
"""

from __future__ import annotations
import argparse, uuid

import numpy as np
import pandas as pd
import syft as sy


DEFAULT_EMAIL    = "info@openmined.org"
DEFAULT_PASSWORD = "changethis"


def main(port: int, rows: int):
    # 1) admin login to the local datasite
    client = sy.login(
        email=DEFAULT_EMAIL,
        password=DEFAULT_PASSWORD,
        url=f"http://localhost:{port}",
    )

    # 2) create toy data
    rng = np.random.default_rng()
    df  = pd.DataFrame({"x": rng.normal(size=rows), "y": rng.normal(size=rows)})

    # 3) build dataset + asset
    tag   = f"localdata-{uuid.uuid4().hex[:6]}"
    asset = sy.Asset(name=f"{tag} asset", data=df, mock=df.head())
    client.upload_dataset(sy.Dataset(name=tag, asset_list=[asset]))

    print(f"Uploaded dataset '{tag}' ({rows} rows) to {client.name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True,
                    help="Port your local Syft server is listening on")
    ap.add_argument("--rows", type=int, default=200,
                    help="Number of rows to generate (default 200)")
    args = ap.parse_args()
    main(args.port, args.rows)
