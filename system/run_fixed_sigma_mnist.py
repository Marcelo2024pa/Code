#!/usr/bin/env python
"""
Experimento: sigmas fixos — FedAvg / SCAFFOLD / FedALA — MNIST

Configuração:
  - Dataset  : MNIST  (non-IID Dirichlet, alpha=0.2, 20 clientes)
  - Algoritmos: FedAvg, SCAFFOLD, FedALA
  - Sigmas   : [1, 4, 12, 20]  (DP com Opacus)
  - Rounds   : 20
  - Clientes : 20  (join_ratio=1.0 → todos participam)

Saídas (em results/csv/):
  fixed_sigma_mnist_rounds_<ts>.csv    — uma linha por round × configuração
  fixed_sigma_mnist_summary_<ts>.csv   — uma linha por configuração
  fixed_sigma_mnist_rounds_latest.csv  — overwrite conveniente
  fixed_sigma_mnist_summary_latest.csv

Uso:
  cd system
  python run_fixed_sigma_mnist.py [--rounds 20] [--clients 20] [--device_id 0]
"""

import argparse
import copy
import csv
import logging
import os
import sys
import warnings
from datetime import datetime

import numpy as np
import torch

# ── paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'dataset'))

sys.path.insert(0, BASE_DIR)
sys.path.insert(0, DATASET_DIR)
sys.path.insert(0, os.path.join(DATASET_DIR, 'utils'))

warnings.simplefilter("ignore")
logging.getLogger().setLevel(logging.ERROR)

# ── imports do projeto ────────────────────────────────────────────────────────
from flcore.servers.serveravg      import FedAvg
from flcore.clients.clientavg      import clientAVG
from flcore.servers.serverscaffold import SCAFFOLD
from flcore.clients.clientscaffold import clientSCAFFOLD
from flcore.servers.serverala      import FedALA
from flcore.clients.clientala      import clientALA
from flcore.trainmodel.models      import FedAvgCNN


# ── colunas do CSV único ──────────────────────────────────────────────────────
CSV_COLS = [
    "algorithm", "sigma", "dataset",
    "round", "total_rounds",
    "test_acc", "train_loss", "epsilon",
]


# ── argparse helper ───────────────────────────────────────────────────────────
class _Args:
    pass


def make_args(algorithm: str, sigma: float, rounds: int, num_clients: int,
              device: torch.device, privacy: bool = True) -> object:
    a = _Args()
    a.dataset                   = "mnist"
    a.num_clients               = num_clients
    a.global_rounds             = rounds
    a.local_epochs              = 1
    a.local_learning_rate       = 0.005
    a.batch_size                = 32
    a.join_ratio                = 1.0        # todos os clientes por round
    a.random_join_ratio         = False
    a.num_classes               = 10
    a.privacy                   = privacy
    a.dp_sigma                  = sigma
    a.max_grad_norm             = 1.0
    a.top_cnt                   = 100
    a.auto_break                = False
    a.eval_gap                  = 1
    a.save_folder_name          = "results"
    a.algorithm                 = algorithm
    a.goal                      = f"sigma_{int(sigma)}" if privacy else "sigma_0"
    a.times                     = 0
    a.num_new_clients           = 0
    a.fine_tuning_epoch_new     = 0
    a.dlg_eval                  = False
    a.dlg_gap                   = 100
    a.batch_num_per_client      = 2
    a.client_drop_rate          = 0.0
    a.train_slow_rate           = 0.0
    a.send_slow_rate            = 0.0
    a.time_select               = False
    a.time_threthold            = 10000
    a.server_learning_rate      = 1.0        # usado pelo SCAFFOLD
    a.verbose                   = False
    a.learning_rate_decay       = False
    a.learning_rate_decay_gamma = 0.99
    a.sigma_per_client          = [sigma] * num_clients
    a.device                    = device
    a.device_id                 = str(device).replace("cuda:", "")
    a.tensorboard               = False

    # FedAvgCNN para MNIST: in_features=1 (grayscale), dim=1024 (4×4×64)
    a.model = FedAvgCNN(in_features=1, num_classes=10, dim=1024).to(device)
    return a


# ── runner de um experimento ──────────────────────────────────────────────────
ALGO_MAP = {
    "FedAvg":   (FedAvg,   clientAVG),
    "SCAFFOLD": (SCAFFOLD, clientSCAFFOLD),
    "FedALA":   (FedALA,   clientALA),
}


def run_experiment(algorithm: str, sigma: float,
                   rounds: int, num_clients: int,
                   device: torch.device) -> dict:
    """
    Executa um experimento e retorna dicionário com rows e summary.
    """
    privacy = sigma > 0
    args = make_args(algorithm, sigma, rounds, num_clients, device, privacy)

    label = f"sigma_{int(sigma)}" if privacy else "sigma_0"
    print(f"\n{'='*65}")
    print(f"  {algorithm:12s} | σ={sigma:5.1f} | {'DP' if privacy else 'SEM DP'} | {rounds} rounds | MNIST")
    print(f"{'='*65}")

    ServerClass, _ = ALGO_MAP[algorithm]
    server = ServerClass(args, times=0)

    server.train()

    acc  = server.rs_test_acc
    loss = server.rs_train_loss
    eps  = getattr(server, "rs_epsilon", None) or [0.0] * len(acc)

    n = min(len(acc), len(loss), len(eps))
    acc, loss, eps = acc[:n], loss[:n], eps[:n]

    rows = []
    for i, (a, l, e) in enumerate(zip(acc, loss, eps)):
        rows.append({
            "algorithm":   algorithm,
            "sigma":       sigma,
            "dataset":     "mnist",
            "round":       i,
            "total_rounds": rounds,
            "test_acc":    round(float(a), 6),
            "train_loss":  round(float(l), 6),
            "epsilon":     round(float(e), 6),
        })

    del server
    torch.cuda.empty_cache()

    return rows


# ── escrita de CSV ─────────────────────────────────────────────────────────────
def write_csv(path: str, cols: list, rows: list) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  -> Salvo: {path}  ({len(rows)} linhas)")


# ── geração do dataset MNIST  ─────────────────────────────────────────────────
def ensure_mnist_dataset(num_clients: int = 20):
    """
    Gera o dataset MNIST (Dirichlet alpha=0.2) se ainda não existir
    com a configuração correcta.
    """
    from generate_mnist import generate_mnist

    mnist_dir = os.path.join(DATASET_DIR, "mnist") + os.sep
    print(f"\n[Dataset] Verificando MNIST em {mnist_dir} ...")
    generate_mnist(mnist_dir, num_clients, niid=True, balance=False, partition="dir")


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Experimento sigmas fixos — FedAvg / SCAFFOLD / FedALA — MNIST"
    )
    parser.add_argument("--rounds",    type=int, default=20,  help="Rounds de FL")
    parser.add_argument("--clients",   type=int, default=20,  help="Número de clientes")
    parser.add_argument("--device_id", type=int, default=0,   help="GPU ID")
    parser.add_argument(
        "--sigmas", type=float, nargs="+", default=[1.0, 4.0, 12.0, 20.0],
        help="Valores de sigma fixo (ex: 1 4 12 20)"
    )
    parser.add_argument(
        "--algorithms", type=str, nargs="+",
        default=["FedAvg", "SCAFFOLD", "FedALA"],
        choices=list(ALGO_MAP.keys()),
        help="Algoritmos a comparar"
    )
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    print(f"Rounds: {args.rounds}  |  Clientes: {args.clients}")
    print(f"Algoritmos: {args.algorithms}")
    print(f"Sigmas    : {args.sigmas}")

    # Gera / verifica dataset MNIST com alpha=0.2
    ensure_mnist_dataset(args.clients)

    out_dir = os.path.abspath(os.path.join(BASE_DIR, '..', 'results', 'csv'))
    os.makedirs(out_dir, exist_ok=True)

    all_rows = []

    # ── loop principal ────────────────────────────────────────────────────────
    for algo in args.algorithms:
        for sigma in args.sigmas:
            torch.manual_seed(42)
            np.random.seed(42)
            rows = run_experiment(algo, sigma, args.rounds, args.clients, device)
            all_rows.extend(rows)

    # ── salva CSV único ───────────────────────────────────────────────────────
    ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path  = os.path.join(out_dir, f"fixed_sigma_mnist_{ts}.csv")
    csv_latest = os.path.join(out_dir, "fixed_sigma_mnist_latest.csv")

    write_csv(csv_path,   CSV_COLS, all_rows)
    write_csv(csv_latest, CSV_COLS, all_rows)

    # ── tabela resumo (última linha por config) ───────────────────────────────
    # Agrupa por (algorithm, sigma) para mostrar métricas finais
    from itertools import groupby
    from operator  import itemgetter

    print(f"\n{'='*72}")
    print("  RESUMO — MNIST  (sigma fixo, Dirichlet alpha=0.2)")
    print(f"{'='*72}")
    print(f"{'Algoritmo':<12} {'σ':>5}  {'Acc_max':>8}  {'Acc_final':>10}  {'ε_final':>9}")
    print("-" * 72)

    sorted_rows = sorted(all_rows, key=itemgetter("algorithm", "sigma"))
    for (algo, sig), grp in groupby(sorted_rows, key=lambda r: (r["algorithm"], r["sigma"])):
        grp = list(grp)
        best_acc  = max(r["test_acc"]  for r in grp)
        last      = grp[-1]
        final_acc = last["test_acc"]
        final_eps = last["epsilon"]
        print(
            f"  {algo:<10}  "
            f"{sig:>5.1f}  "
            f"{best_acc*100:>7.2f}%  "
            f"{final_acc*100:>9.2f}%  "
            f"{final_eps:>8.4f}"
        )
    print(f"{'='*72}")
    print(f"\nCSV salvo em: {csv_path}")


if __name__ == "__main__":
    main()
