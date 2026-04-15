"""
CSV logger com escrita segura (fcntl lock) para uso concorrente entre processos.

Três CSVs gerados em results/csv/:
  training_rounds.csv      — uma linha por round de FL (todos os treinos)
  optimization_individuals.csv — uma linha por indivíduo avaliado no AG
  optimization_generations.csv — uma linha por geração (resumo do melhor)
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

OPT_INDIVIDUALS_COLS = [
    "timestamp", "generation", "total_generations",
    "individual_idx", "run_id",
    "fitness", "accuracy", "epsilon",
    "sigma_mean", "sigma_std", "sigma_var", "sigma_min", "sigma_max",
    "sigma_per_client",
]

OPT_GENERATIONS_COLS = [
    "timestamp", "generation", "total_generations",
    "best_fitness", "best_accuracy", "best_epsilon",
    "avg_fitness", "std_fitness", "var_fitness",
    "avg_accuracy", "std_accuracy", "var_accuracy",
    "avg_epsilon", "std_epsilon", "var_epsilon",
    "sigma_mean_best", "sigma_std_best", "sigma_var_best",
    "sigma_min_best", "sigma_max_best",
    "sigma_best",
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
    """Return mean/std/var/min/max and JSON string from a sigma list."""
    if not sigma_list:
        return {
            "sigma_mean": None, "sigma_std": None, "sigma_var": None,
            "sigma_min": None, "sigma_max": None, "sigma_json": None,
        }
    import numpy as np
    arr = [float(x) for x in sigma_list]
    return {
        "sigma_mean": round(float(np.mean(arr)), 4),
        "sigma_std":  round(float(np.std(arr)),  4),
        "sigma_var":  round(float(np.var(arr)),  6),
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
            "sigma_mean":       stats["sigma_mean"],
            "sigma_std":        stats["sigma_std"],
            "sigma_var":        stats["sigma_var"],
            "sigma_min":        stats["sigma_min"],
            "sigma_max":        stats["sigma_max"],
            "sigma_per_client": stats["sigma_json"],
        }
        _append_row(os.path.join(csv_dir, "training_rounds.csv"), TRAINING_ROUNDS_COLS, row)
    except Exception:
        pass


def log_opt_individual(
    generation: int,
    total_generations: int,
    individual_idx: int,
    run_id: str,
    fitness: float,
    accuracy: float,
    epsilon: float,
    sigma_vector: List[float],
    csv_dir: str = CSV_DIR,
) -> None:
    """Log one AG individual evaluation."""
    try:
        stats = _sigma_stats(sigma_vector)
        row = {
            "timestamp":         datetime.now().isoformat(timespec="seconds"),
            "generation":        generation,
            "total_generations": total_generations,
            "individual_idx":    individual_idx,
            "run_id":            run_id,
            "fitness":           round(fitness,  6),
            "accuracy":          round(accuracy, 6),
            "epsilon":           round(epsilon,  6),
            "sigma_mean":        stats["sigma_mean"],
            "sigma_std":         stats["sigma_std"],
            "sigma_var":         stats["sigma_var"],
            "sigma_min":         stats["sigma_min"],
            "sigma_max":         stats["sigma_max"],
            "sigma_per_client":  stats["sigma_json"],
        }
        _append_row(os.path.join(csv_dir, "optimization_individuals.csv"), OPT_INDIVIDUALS_COLS, row)
    except Exception:
        pass


def log_opt_generation(
    generation: int,
    total_generations: int,
    best_fitness: float,
    best_accuracy: float,
    best_epsilon: float,
    avg_fitness: float,
    std_fitness: float,
    var_fitness: float,
    avg_accuracy: float,
    std_accuracy: float,
    var_accuracy: float,
    avg_epsilon: float,
    std_epsilon: float,
    var_epsilon: float,
    best_sigma: List[float],
    csv_dir: str = CSV_DIR,
) -> None:
    """Log one AG generation summary."""
    try:
        stats = _sigma_stats(best_sigma)
        row = {
            "timestamp":         datetime.now().isoformat(timespec="seconds"),
            "generation":        generation,
            "total_generations": total_generations,
            "best_fitness":      round(best_fitness,  6),
            "best_accuracy":     round(best_accuracy, 6),
            "best_epsilon":      round(best_epsilon,  6),
            "avg_fitness":       round(avg_fitness,   6),
            "std_fitness":       round(std_fitness,   6),
            "var_fitness":       round(var_fitness,   6),
            "avg_accuracy":      round(avg_accuracy,  6),
            "std_accuracy":      round(std_accuracy,  6),
            "var_accuracy":      round(var_accuracy,  6),
            "avg_epsilon":       round(avg_epsilon,   6),
            "std_epsilon":       round(std_epsilon,   6),
            "var_epsilon":       round(var_epsilon,   6),
            "sigma_mean_best":   stats["sigma_mean"],
            "sigma_std_best":    stats["sigma_std"],
            "sigma_var_best":    stats["sigma_var"],
            "sigma_min_best":    stats["sigma_min"],
            "sigma_max_best":    stats["sigma_max"],
            "sigma_best":        stats["sigma_json"],
        }
        _append_row(os.path.join(csv_dir, "optimization_generations.csv"), OPT_GENERATIONS_COLS, row)
    except Exception:
        pass
