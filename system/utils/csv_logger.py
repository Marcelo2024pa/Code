"""
CSV logger com escrita segura (fcntl lock) para uso concorrente entre processos.

Gera em results/csv/:
  training_rounds.csv - uma linha por round de FL (todos os treinos)
"""

import csv
import fcntl
import json
import os
from datetime import datetime
from typing import List, Optional

CSV_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "results", "csv")
)

# ── Schemas ───────────────────────────────────────────────────────────────────

TRAINING_ROUNDS_COLS = [
    "timestamp", "run_id", "goal", "dataset", "algorithm",
    "round", "total_rounds",
    "test_acc", "train_loss", "epsilon",
    "sigma_mean", "sigma_min", "sigma_max",
    "sigma_per_client",
]


# ── Internal helpers ──────────────────────────────────────────────────────────

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _append_row(filepath: str, cols: List[str], row: dict) -> None:
    """Append one row to a CSV, creating header if the file is new. Thread/process-safe."""
    _ensure_dir(os.path.dirname(filepath))
    lock_path = filepath + ".lock"

    with open(lock_path, "w") as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)
        try:
            new_file = not os.path.exists(filepath) or os.path.getsize(filepath) == 0
            with open(filepath, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
                if new_file:
                    writer.writeheader()
                writer.writerow(row)
        finally:
            fcntl.flock(lock_f, fcntl.LOCK_UN)


def _sigma_stats(sigma_list) -> dict:
    """Return mean/min/max and JSON string from a sigma list."""
    if not sigma_list:
        return {
            "sigma_mean": None,
            "sigma_min": None,
            "sigma_max": None,
            "sigma_json": None,
        }
    import numpy as np
    arr = [float(x) for x in sigma_list]
    return {
        "sigma_mean": round(float(np.mean(arr)), 4),
        "sigma_min":  round(float(np.min(arr)),  4),
        "sigma_max":  round(float(np.max(arr)),  4),
        "sigma_json": json.dumps([round(x, 4) for x in arr]),
    }


# ── Public API ────────────────────────────────────────────────────────────────

def log_training_round(
    run_id: str,
    goal: str,
    dataset: str,
    algorithm: str,
    round_num: int,
    total_rounds: int,
    test_acc: Optional[float],
    train_loss: Optional[float],
    epsilon: Optional[float],
    sigma_per_client: Optional[List[float]],
    csv_dir: str = CSV_DIR,
) -> None:
    """Log one FL training round."""
    try:
        stats = _sigma_stats(sigma_per_client)
        row = {
            "timestamp":      datetime.now().isoformat(timespec="seconds"),
            "run_id":         run_id,
            "goal":           goal,
            "dataset":        dataset,
            "algorithm":      algorithm,
            "round":          round_num,
            "total_rounds":   total_rounds,
            "test_acc":       round(test_acc,  6) if test_acc  is not None else "",
            "train_loss":     round(train_loss, 6) if train_loss is not None else "",
            "epsilon":        round(epsilon,   6) if epsilon   is not None else "",
            "sigma_mean":     stats["sigma_mean"],
            "sigma_min":      stats["sigma_min"],
            "sigma_max":      stats["sigma_max"],
            "sigma_per_client": stats["sigma_json"],
        }
        _append_row(os.path.join(csv_dir, "training_rounds.csv"), TRAINING_ROUNDS_COLS, row)
    except Exception:
        pass
