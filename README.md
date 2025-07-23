# Syft‑Playground – Federated Pearson Demo

A lean sandbox that spins up **multiple local Syft datasites**, uploads toy
data, then computes a **privacy‑preserving Pearson correlation** (rows never
leave their home server – only 6 summary numbers are shared).

```
.
├── config.json          # how many sites & which base port
├── tasks.py             # Invoke CLI entry‑points (deploy, load‑data, run …)
├── pearson.py           # algorithm module (federated aggregates → r)
└── pyproject.toml       # Poetry env: syft‑cli 0.9.x, torch‑cpu, numpy, …
```

---

## 1 . Set‑up (Poetry, CPU‑only)

```bash
pip install --user poetry
poetry install          # resolves deps from pyproject.toml
poetry shell            # activate the venv
```

Dependencies (CPU wheels):

```
syft 0.9.5
syft-cli 0.9.*
torch 2.2.2+cpu
numpy 1.24–2.2
scipy 1.10–1.11
invoke, psutil  (dev utilities)
```

---

## 2 . Configuration

`config.json`

```json
{
  "num_clients": 3,
  "base_port": 8080
}
```

Change either value as you like; all scripts/tasks pick it up automatically.

---

## 3 . Core tasks

The CLI entry‑point is **`inv`** (Invoke’s short alias).

| Command         | What it does                                                                                                                                                                                 |
| --------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `inv deploy`    | Launches *N* Syft servers on consecutive ports (\<base\_port> … +N‑1). If any Syft servers are already running, it auto‑cleans them first (SIGTERM → SIGKILL fallback) and removes old logs. |
| `inv load-data` | Logs in as admin on every server and uploads a toy `(x, y)` DataFrame (size = 200 + 100 × index).                                                                                            |
| `inv run`       | Imports **`pearson.compute_global_pearson`** – each site returns `(n, Σx, Σy, Σx², Σy², Σxy)`; the client merges and prints the global Pearson *r*.                                          |
| `inv cleanup`   | Stops *all* `syft launch` processes (tracked and untracked) and deletes `syft_logs/`.                                                                                                        |

Example workflow:

```bash
inv deploy          # start datasites
inv load-data       # upload toy data
inv run             # compute Pearson r
inv cleanup         # tear down & wipe logs
```

---

## 4 . Privacy model

* Each datasite executes a local stats function that emits just six
  aggregates.
* No raw rows cross server boundaries.
* The central client combines aggregates → global correlation.
* Good fit for demos of federated analytics or as a template for adding DP/MPC layers later.

---

## 5 . Files in detail

| File             | Role                                                                                                                                                                      |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `tasks.py`       | Invoke tasks, server management, data upload. Uses `psutil` to kill leftover Syft servers robustly, stores child PIDs in `.syft_pids`, and logs to `syft_logs/site*.log`. |
| `pearson.py`     | Pure algorithm: single‑use per‑site stats function, aggregate merge, Pearson *r* math. Can be imported by other code/tests.                                               |
| `config.json`    | Central place to set `num_clients` and `base_port`.                                                                                                                       |
| `pyproject.toml` | Poetry env (+ PyTorch CPU index URL).                                                                                                                                     |

Logs live in `syft_logs/` and are wiped by `inv cleanup`.

---

## 6 . Troubleshooting

| Issue                            | Fix                                                                                                                 |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| Ports already in use             | Edit `base_port` in *config.json*.                                                                                  |
| “No dataset found on orgX” error | Run `inv load-data` after `inv deploy`.                                                                             |
| Stale Syft processes remain      | `inv cleanup` runs a Python `psutil` sweep; if anything survives, run `kill -9 $(pgrep -f "syft launch")` manually. |

---

## 7 . License

MIT – do whatever you like, but no warranty.
