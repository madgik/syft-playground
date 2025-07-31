# --- keep previous imports / config here -------------------------------
from typing import List, Dict, Any
import json, os
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
    if isinstance(obj, tuple):
        return obj
    if hasattr(obj, "resolve"):
        return obj.resolve()
    if hasattr(obj, "get"):
        return obj.get()
    raise TypeError(f"Cannot convert {type(obj)}")


def get_assets(label_col: str | None = None):
    """
    Return (assets, feature_dim)
    • assets      – list of Syft Asset objects (one per site)
    • feature_dim – d  (+1 for bias)  if label_col is given, else None
    """
    assets: List[Any] = []
    dim = None

    for site in SITES:
        url = f"http://{site['host']}:{site['port']}"
        client = sy.login(email=EMAIL, password=PASSWORD, url=url)
        client.refresh()

        try:
            ds = next(iter(client.datasets))
        except StopIteration:
            raise RuntimeError(f"No dataset on {url}")

        asset = ds.assets[0]
        assets.append(asset)

        if label_col and dim is None:
            dim = asset.data.columns.drop(label_col).size + 1  # +bias

    return (assets, dim) if label_col else (assets, None)
