from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import time
import uuid
from pathlib import Path

import invoke
import numpy as np
import pandas as pd
import psutil
import syft as sy

import logging
for name in ("syft", "syft.server.server"):
    logging.getLogger(name).setLevel(logging.ERROR)


from pearson import compute_global_pearson

# ------------------------------------------------------------------
# Configuration -----------------------------------------------------
CFG = json.loads(Path("config.json").read_text())
NUM, BASE = CFG["num_clients"], CFG["base_port"]

LOG_DIR  = Path("syft_logs")
PID_FILE = Path(".syft_pids")
MARKER   = "Application startup complete"

# ------------------------------------------------------------------
# Helper functions --------------------------------------------------
def _launch_server(idx: int) -> subprocess.Popen:
    org = f"org{idx + 1}"
    port = BASE + idx
    log_path = LOG_DIR / f"site{idx}.log"
    cmd = ["syft", "launch", f"--name={org}", f"--port={port}", "--reset=True"]
    proc = subprocess.Popen(cmd, stdout=open(log_path, "w"), stderr=subprocess.STDOUT)
    print(f"Started {org} on port {port}. Logs: {log_path}")
    return proc


def _wait_until_ready(timeout: float = 30.0) -> None:
    start = time.time()
    logs = [LOG_DIR / f"site{i}.log" for i in range(NUM)]
    print("Waiting for datasites to start …")
    while time.time() - start < timeout:
        if all(p.exists() and MARKER in p.read_text(errors="ignore") for p in logs):
            print("All datasites are up.")
            return
        time.sleep(0.5)
    _kill_all_syft()
    print("Timeout waiting for datasites.")


def _store_pids(procs):
    PID_FILE.write_text("\n".join(str(p.pid) for p in procs))


def _read_pids():
    return [int(pid) for pid in PID_FILE.read_text().splitlines()] if PID_FILE.exists() else []

def _upload_dataset(client: sy.Client, idx: int) -> None:
    rng = np.random.default_rng(idx)
    rows = 200 + idx * 100
    df = pd.DataFrame({"x": rng.normal(size=rows), "y": rng.normal(size=rows)})
    name = f"site{idx + 1}-toy-{uuid.uuid4().hex[:6]}"
    asset = sy.Asset(name=f"{name} asset", data=df, mock=df.head())
    client.upload_dataset(sy.Dataset(name=name, asset_list=[asset]))
    print(f"Uploaded dataset '{name}' to {client.name}")

def _syft_running() -> bool:
    """Return True if *any* syft‑launch process is alive."""
    for p in psutil.process_iter(["cmdline"]):
        cmd = p.info["cmdline"] or []
        if len(cmd) >= 2 and cmd[0].endswith("syft") and cmd[1] == "launch":
            return True
    return False

def _kill_all_syft():
    """Return number of processes killed."""
    targets = []
    for p in psutil.process_iter(["pid", "cmdline"]):
        cmd = p.info["cmdline"] or []
        if len(cmd) >= 2 and cmd[0].endswith("syft") and "launch" in cmd:
            targets.append(p)

    if not targets:
        return 0

    # first try graceful terminate
    for p in targets:
        try:
            p.terminate()
        except psutil.NoSuchProcess:
            pass

    gone, alive = psutil.wait_procs(targets, timeout=5)

    # hard kill anything still alive
    for p in alive:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass

# ------------------------------------------------------------------

@invoke.task
def cleanup(c):
    """Stop datasites and remove logs."""
    # tidy up PIDs from last deploy (if any)
    for pid in _read_pids():
        try:
            os.kill(pid, signal.SIGINT)
        except ProcessLookupError:
            pass
    PID_FILE.unlink(missing_ok=True)

    # kill *all* syft‑launch processes, even from older runs
    _kill_all_syft()

    # delete logs directory
    if LOG_DIR.exists():
        shutil.rmtree(LOG_DIR, ignore_errors=True)

    print("Cleanup complete.")


@invoke.task
def deploy(c):
    """Launch datasites; auto‑cleanup if something is already running."""
    if _syft_running():
        print("Existing Syft servers detected – cleaning up first.")
        cleanup(c)

    LOG_DIR.mkdir(exist_ok=True)
    procs = [_launch_server(i) for i in range(NUM)]
    _store_pids(procs)
    _wait_until_ready()

@invoke.task()
def load_data(c):
    """Upload toy data to every datasite."""
    for i in range(NUM):
        client = sy.login(port=BASE + i, email="info@openmined.org", password="changethis")
        _upload_dataset(client, i)

@invoke.task()
def run(c):
    """Call pearson.compute_global_pearson and print the result."""
    total_rows, r = compute_global_pearson(NUM, BASE)
    print(f"\nGlobal Pearson r over {total_rows} rows, {NUM} sites: {r:.6f}")
