#!/usr/bin/env python
"""
Comparação completa — SCAFFOLD / CIFAR-10  (FL-DP2: sigma schedule + join_ratio<1)

Experimentos:
  - sigma=0             (sem DP / sem ruído, baseline superior)
  - sigma=12            (melhor fixo da Fase 1, baseline de referência)
  - sched_20_12         (schedule alto→médio: reduz 40% ε vs sigma=12 fixo)
  - sched_18_10         (schedule intermediário)
  - sched_20_8          (schedule agressivo em ε)
  - sigma_schedule_AG   (AG otimiza [σ_start, σ_end] — 2 genes globais)

join_ratio = 0.5 → amplificação por subsampling: ε_efetivo ≈ q·ε

Saídas (em results/csv/):
  comparison_rounds_<ts>.csv   — uma linha por round × configuração
  comparison_summary_<ts>.csv  — uma linha por configuração (métricas finais)

Uso:
  cd system
  python run_full_comparison.py [--rounds 50] [--clients 20] [--device_id 0]
"""

import argparse
import copy
import csv
import json
import logging
import os
import sys
import warnings
from datetime import datetime

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.simplefilter("ignore")
logging.getLogger().setLevel(logging.ERROR)

from flcore.servers.serverscaffold import SCAFFOLD
from flcore.clients.clientscaffold import clientSCAFFOLD
from flcore.trainmodel.models import FedAvgCNN
from optimization.genetic_algorithm import GeneticAlgorithm
from optimization.fitness import FitnessEvaluator

# ── Constantes ────────────────────────────────────────────────────────────────

COMPARISON_ROUNDS_COLS = [
    "sigma_label", "run_type", "algorithm", "dataset",
    "round", "total_rounds",
    "test_acc", "train_loss", "epsilon",
    "sigma_mean", "sigma_std", "sigma_var", "sigma_min", "sigma_max",
    "acc_per_epsilon",
]

COMPARISON_SUMMARY_COLS = [
    "sigma_label", "run_type", "algorithm", "dataset",
    "best_acc", "final_acc", "final_epsilon", "acc_per_epsilon_final",
    "sigma_mean", "sigma_std", "sigma_var", "sigma_min", "sigma_max",
    "total_rounds",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

class _Args: pass  # module-level so multiprocessing can pickle it


def make_args(sigma: float, rounds: int, num_clients: int, device: torch.device,
              privacy: bool = True) -> object:
    a = _Args()
    a.dataset                   = "Cifar10"
    a.num_clients               = num_clients
    a.global_rounds             = rounds
    a.local_epochs              = 1
    a.local_learning_rate       = 0.005
    a.batch_size                = 32
    a.join_ratio                = 0.5
    a.random_join_ratio         = False
    a.num_classes               = 10
    a.privacy                   = privacy
    a.dp_sigma                  = sigma
    a.max_grad_norm             = 1.0
    a.top_cnt                   = 100
    a.auto_break                = False
    a.eval_gap                  = 1
    a.save_folder_name          = "results"
    a.algorithm                 = "SCAFFOLD"
    a.goal                      = f"sigma_{sigma}" if privacy else "sigma_0"
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
    a.server_learning_rate      = 1.0
    a.verbose                   = False
    a.learning_rate_decay       = False
    a.learning_rate_decay_gamma = 0.99
    a.sigma_per_client          = [sigma] * num_clients
    a.device                    = device
    a.device_id                 = str(device).replace("cuda:", "")
    # GA extras (needed by optimize_sigma internals)
    a.population                = 10
    a.ga_generations            = 10
    a.sigma_min                 = 1.0
    a.sigma_max                 = 20.0
    a.mutation_rate             = 0.2
    a.mutation_sigma            = 1.9   # 10% do range [1,20]
    a.crossover_rate            = 0.8
    a.elitism                   = 2
    a.tournament_size           = 3
    a.seed                      = 42
    a.fitness_type              = "linear"
    a.lambda_weight             = 0.1
    a.device_ids                = [int(str(device).replace("cuda:", "") or 0)]
    a.model = FedAvgCNN(in_features=3, num_classes=10, dim=1600).to(device)
    return a


def sigma_stats(sigmas, sigma_start=None, sigma_end=None):
    """
    Compute sigma statistics.
    When sigma_start/sigma_end are given (schedule experiments), uses them
    to describe the range instead of the sigma_per_client list.
    """
    if sigma_start is not None and sigma_end is not None:
        lo = min(sigma_start, sigma_end)
        hi = max(sigma_start, sigma_end)
        mid = (sigma_start + sigma_end) / 2.0
        return {
            "sigma_mean": mid,
            "sigma_std":  abs(sigma_end - sigma_start) / 2.0,
            "sigma_var":  ((sigma_end - sigma_start) / 2.0) ** 2,
            "sigma_min":  lo,
            "sigma_max":  hi,
        }
    arr = np.array([float(s) for s in sigmas])
    return {
        "sigma_mean": float(np.mean(arr)),
        "sigma_std":  float(np.std(arr)),
        "sigma_var":  float(np.var(arr)),
        "sigma_min":  float(np.min(arr)),
        "sigma_max":  float(np.max(arr)),
    }


def run_experiment(label: str, run_type: str,
                   sigma_per_client: list, privacy: bool,
                   rounds: int, num_clients: int,
                   device: torch.device,
                   sigma_schedule: dict = None,
                   sigma_start: float = None,
                   sigma_end: float = None) -> dict:
    """
    Roda um experimento SCAFFOLD e retorna dict com curvas por round.

    sigma_schedule (optional): {client_id: (sigma_start, sigma_end)} for
    per-round linear decay. If None, sigma_per_client is used as fixed noise.

    sigma_start / sigma_end (optional): used only for sigma_stats display when
    running schedule experiments (shows true range instead of start values only).
    """
    sigma_val = sigma_per_client[0] if len(set(sigma_per_client)) == 1 else 0.0
    args = make_args(sigma_val, rounds, num_clients, device, privacy)
    args.sigma_per_client = sigma_per_client
    args.sigma_schedule   = sigma_schedule   # None = fixed; dict = per-round decay
    args.privacy = privacy
    args.goal = label   # garante que o serverbase loga com o label correto

    print(f"\n{'='*60}")
    print(f"  {label}  |  {'DP' if privacy else 'SEM DP'}  |  {rounds} rounds")
    if sigma_schedule and sigma_start is not None and sigma_end is not None:
        print(f"  σ schedule: {sigma_start:.1f} → {sigma_end:.1f}  (linear, todos os clientes)")
    else:
        print(f"  σ range: [{min(sigma_per_client):.2f}, {max(sigma_per_client):.2f}]")
    print(f"{'='*60}")

    server = SCAFFOLD(args, times=0)
    for client in server.clients:
        client.dp_sigma = sigma_per_client[client.id]

    server.train()

    acc   = server.rs_test_acc
    loss  = server.rs_train_loss
    eps   = server.rs_epsilon if server.rs_epsilon else [0.0] * len(acc)

    # Alinha comprimentos (avaliação pode ser a cada eval_gap)
    n = min(len(acc), len(loss), len(eps))
    acc, loss, eps = acc[:n], loss[:n], eps[:n]

    stats = sigma_stats(sigma_per_client, sigma_start=sigma_start, sigma_end=sigma_end)

    # Constrói linhas por round
    rows = []
    for i, (a, l, e) in enumerate(zip(acc, loss, eps)):
        rows.append({
            "sigma_label":    label,
            "run_type":       run_type,
            "algorithm":      "SCAFFOLD",
            "dataset":        "Cifar10",
            "round":          i,
            "total_rounds":   rounds,
            "test_acc":       round(float(a), 6),
            "train_loss":     round(float(l), 6),
            "epsilon":        round(float(e), 6),
            "sigma_mean":     round(stats["sigma_mean"], 4),
            "sigma_std":      round(stats["sigma_std"],  4),
            "sigma_var":      round(stats["sigma_var"],  6),
            "sigma_min":      round(stats["sigma_min"],  4),
            "sigma_max":      round(stats["sigma_max"],  4),
            "acc_per_epsilon": round(float(a) / (float(e) + 1e-6), 4),
        })

    # Resumo
    summary = {
        "sigma_label":         label,
        "run_type":            run_type,
        "algorithm":           "SCAFFOLD",
        "dataset":             "Cifar10",
        "best_acc":            round(max(acc), 6) if acc else 0.0,
        "final_acc":           round(float(acc[-1]), 6) if acc else 0.0,
        "final_epsilon":       round(float(eps[-1]), 6) if eps else 0.0,
        "acc_per_epsilon_final": round(float(acc[-1]) / (float(eps[-1]) + 1e-6), 4) if acc else 0.0,
        "sigma_mean":          round(stats["sigma_mean"], 4),
        "sigma_std":           round(stats["sigma_std"],  4),
        "sigma_var":           round(stats["sigma_var"],  6),
        "sigma_min":           round(stats["sigma_min"],  4),
        "sigma_max":           round(stats["sigma_max"],  4),
        "total_rounds":        rounds,
    }

    del server
    torch.cuda.empty_cache()

    return {"rows": rows, "summary": summary}


def run_ag_optimization(rounds: int, num_clients: int, device: torch.device,
                        population: int, generations: int,
                        n_evals: int = 1,
                        patience: int = 5,
                        tol: float = 1e-3,
                        min_diversity: float = 0.02) -> tuple:
    """
    Roda o AG e retorna (sigma_start, sigma_end) do melhor schedule encontrado.

    Abordagem FL-DP2:
    - 2 genes globais [σ_start, σ_end] — schedule linear aplicado a todos os clientes
    - σ(t) = σ_start + (σ_end − σ_start) · (t/T)  — começa alto (≤ ε) e pode cair
    - join_ratio = 0.5 → amplificação por subsampling (ε_efetivo ≈ 0.5·ε)
    - λ = 0.05 (calibrado: ε ~3x maior em runs longos)
    - Seeds cobrem os candidatos mais promissores da teoria
    """
    SIGMA_MIN = 1.0
    SIGMA_MAX = 20.0
    MUTATION_SIGMA = (SIGMA_MAX - SIGMA_MIN) * 0.15   # ±2.85 por gene
    LAMBDA_WEIGHT = 0.05   # calibrado: ε ~3x mais alto que runs curtos

    args = make_args(1.0, rounds, num_clients, device, privacy=True)
    args.goal                 = "optimization"
    args.sigma_min            = SIGMA_MIN
    args.sigma_max            = SIGMA_MAX
    args.mutation_rate        = 0.3
    args.mutation_sigma       = MUTATION_SIGMA
    args.crossover_rate       = 0.8
    args.elitism              = 1
    args.tournament_size      = 3
    args.seed                 = 42
    args.fitness_type         = "linear"
    args.lambda_weight        = LAMBDA_WEIGHT
    args.device_ids           = [int(str(device).replace("cuda:", "") or 0)]
    args.algorithm            = "SCAFFOLD"
    args.verbose              = False
    args.sigma_schedule_type  = "linear"   # communicated to FitnessEvaluator

    fitness_evaluator = FitnessEvaluator(
        args=args,
        server_class=SCAFFOLD,
        client_class=clientSCAFFOLD,
        fitness_type=args.fitness_type,
        lambda_weight=LAMBDA_WEIGHT,
        use_cache=True,
        device_ids=args.device_ids,
        n_evals=n_evals,
        base_seed=args.seed,
        schedule_type="linear",
    )

    # client_groups placeholder — not used with 2-gene global schedule,
    # but FitnessEvaluator.evaluate() still receives it
    class _DummyGroups:
        pass

    dummy_groups = _DummyGroups()

    def fitness_func(sigma_vector):
        return fitness_evaluator.evaluate(sigma_vector, dummy_groups, verbose=False)

    # Seeds: pares [σ_start, σ_end] que exploram o espaço teoria-motivados
    # Alto→baixo: gasta pouco ε no início (σ grande) e aceita mais ruído no fim
    seed_individuals = [
        [20.0, 12.0],   # teoria: alto início, médio fim → -40% ε vs σ=12 fixo
        [18.0, 10.0],   # intermediário
        [20.0,  8.0],   # agressivo em ε
        [16.0,  8.0],   # moderado
        [12.0, 12.0],   # σ=12 fixo (baseline)
        [20.0, 20.0],   # σ=20 fixo (mais privado)
        [15.0, 10.0],   # gradiente suave
        [20.0,  4.0],   # muito agressivo
    ]

    _csv_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "results", "csv")
    )

    ga = GeneticAlgorithm(
        num_genes=2,
        sigma_min=SIGMA_MIN,
        sigma_max=SIGMA_MAX,
        population_size=population,
        generations=generations,
        mutation_rate=args.mutation_rate,
        mutation_sigma=MUTATION_SIGMA,
        crossover_rate=args.crossover_rate,
        elitism=args.elitism,
        tournament_size=args.tournament_size,
        seed=args.seed,
        csv_dir=_csv_dir,
        seed_individuals=seed_individuals,
        patience=patience,
        tol=tol,
        min_diversity=min_diversity,
    )

    print(f"\n{'='*60}")
    print(f"  AG SCAFFOLD FL-DP2 — pop={population}  gen={generations}  rounds={rounds}")
    print(f"  Genes: 2 [σ_start, σ_end] — schedule global (todos os clientes)")
    print(f"  Range: [{SIGMA_MIN}, {SIGMA_MAX}]  mutation_σ={MUTATION_SIGMA:.2f}")
    print(f"  join_ratio=0.5 → amplificação por subsampling")
    print(f"  n_evals={n_evals} (média sobre {n_evals} run(s) por indivíduo)")
    print(f"  Fitness: Acc - {LAMBDA_WEIGHT}·ε  (λ calibrado)")
    print(f"  Early stopping: patience={patience}  tol={tol:.0e}  min_diversity={min_diversity}")
    print(f"  Init: {len(seed_individuals)} indivíduos semente + aleatórios")
    print(f"{'='*60}")

    best = ga.evolve(fitness_func=fitness_func, verbose=True)

    sigma_start = float(best.sigma_vector[0])
    sigma_end   = float(best.sigma_vector[1])
    return sigma_start, sigma_end


def write_csv(path: str, cols: list, rows: list) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  -> Salvo: {path}  ({len(rows)} linhas)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Comparação completa — SCAFFOLD FL-DP2")
    parser.add_argument("--rounds",     type=int, default=150, help="Rounds de FL por experimento")
    parser.add_argument("--clients",    type=int, default=20,  help="Número de clientes")
    parser.add_argument("--device_id",  type=int, default=0,   help="GPU ID")
    parser.add_argument("--population", type=int, default=10,  help="Tamanho da população AG")
    parser.add_argument("--generations",type=int, default=10,  help="Gerações do AG")
    parser.add_argument("--n_evals",      type=int,   default=1,    help="Avaliações por indivíduo no AG")
    parser.add_argument("--patience",     type=int,   default=5,    help="Gerações sem melhora para early stopping")
    parser.add_argument("--tol",          type=float, default=1e-3, help="Melhora mínima de fitness para resetar patience")
    parser.add_argument("--min_diversity",type=float, default=0.02, help="Fração do range sigma abaixo da qual para por colapso")
    parser.add_argument("--skip_ag",    action="store_true",   help="Pula otimização AG")
    parser.add_argument("--skip_fixed", action="store_true",   help="Pula runs de sigma fixo e schedules manuais")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    print(f"Rounds: {args.rounds}  |  Clientes: {args.clients}  |  join_ratio=0.5")

    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "csv"))
    os.makedirs(out_dir, exist_ok=True)

    all_round_rows = []
    all_summaries  = []

    # ── 1. Baselines: sigma fixo (sem DP + melhor da Fase 1) ─────────────────
    fixed_configs = [
        ("sigma_0",  0.0,  False),   # sem DP — teto de desempenho
        ("sigma_12", 12.0, True),    # melhor fixo da Fase 1
    ]

    # ── 2. Schedules manuais: teoria σ_alto→σ_baixo ──────────────────────────
    # σ(t) = σ_start + (σ_end − σ_start) * t/T
    schedule_configs = [
        ("sched_20_12", 20.0, 12.0),   # -40% ε vs sigma_12 fixo (teórico)
        ("sched_18_10", 18.0, 10.0),   # intermediário
        ("sched_20_8",  20.0,  8.0),   # agressivo em ε
    ]

    if args.skip_fixed:
        print("\n[skip_fixed] Pulando runs de sigma fixo e schedules manuais.")
    else:
        for label, sigma, privacy in fixed_configs:
            sigma_per_client = [sigma] * args.clients
            result = run_experiment(
                label=label,
                run_type="fixed",
                sigma_per_client=sigma_per_client,
                privacy=privacy,
                rounds=args.rounds,
                num_clients=args.clients,
                device=device,
            )
            all_round_rows.extend(result["rows"])
            all_summaries.append(result["summary"])

        for label, s_start, s_end in schedule_configs:
            sigma_per_client = [s_start] * args.clients
            schedule_dict    = {i: (s_start, s_end) for i in range(args.clients)}
            result = run_experiment(
                label=label,
                run_type="schedule",
                sigma_per_client=sigma_per_client,
                privacy=True,
                rounds=args.rounds,
                num_clients=args.clients,
                device=device,
                sigma_schedule=schedule_dict,
                sigma_start=s_start,
                sigma_end=s_end,
            )
            all_round_rows.extend(result["rows"])
            all_summaries.append(result["summary"])

    # ── 3. Otimização AG — busca melhor [σ_start, σ_end] ─────────────────────
    best_sigma_start, best_sigma_end = None, None

    if args.skip_ag:
        print("\n[skip_ag] Pulando otimização AG.")
        # Usa o melhor schedule manual como fallback
        best_sigma_start, best_sigma_end = 20.0, 12.0
        print(f"  Fallback: σ_start={best_sigma_start}  σ_end={best_sigma_end}")
    else:
        best_sigma_start, best_sigma_end = run_ag_optimization(
            rounds=args.rounds,
            num_clients=args.clients,
            device=device,
            population=args.population,
            generations=args.generations,
            n_evals=args.n_evals,
            patience=args.patience,
            tol=args.tol,
            min_diversity=args.min_diversity,
        )

    # ── 4. Avaliação final com melhor schedule do AG ──────────────────────────
    best_sigma_per_client = [best_sigma_start] * args.clients
    best_schedule_dict    = {i: (best_sigma_start, best_sigma_end) for i in range(args.clients)}

    result = run_experiment(
        label="sigma_schedule_AG",
        run_type="adaptive",
        sigma_per_client=best_sigma_per_client,
        privacy=True,
        rounds=args.rounds,
        num_clients=args.clients,
        device=device,
        sigma_schedule=best_schedule_dict,
        sigma_start=best_sigma_start,
        sigma_end=best_sigma_end,
    )
    all_round_rows.extend(result["rows"])
    all_summaries.append(result["summary"])

    # ── 4. Escreve CSVs unificados ────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    rounds_path  = os.path.join(out_dir, f"comparison_rounds_{ts}.csv")
    summary_path = os.path.join(out_dir, f"comparison_summary_{ts}.csv")

    # Também sobrescreve a versão "latest" para facilitar leitura
    rounds_latest  = os.path.join(out_dir, "comparison_rounds_latest.csv")
    summary_latest = os.path.join(out_dir, "comparison_summary_latest.csv")

    print(f"\n{'='*60}")
    print("  ESCREVENDO CSVs")
    print(f"{'='*60}")

    write_csv(rounds_path,   COMPARISON_ROUNDS_COLS,  all_round_rows)
    write_csv(summary_path,  COMPARISON_SUMMARY_COLS, all_summaries)
    write_csv(rounds_latest, COMPARISON_ROUNDS_COLS,  all_round_rows)
    write_csv(summary_latest,COMPARISON_SUMMARY_COLS, all_summaries)

    # ── 5. Imprime tabela resumo ──────────────────────────────────────────────
    print(f"\n{'='*75}")
    print("  RESUMO COMPARATIVO  (join_ratio=0.5)")
    print(f"{'='*75}")
    print(f"{'Config':<22} {'Acc_max':>8} {'Acc_final':>10} {'ε_final':>9} {'Acc/ε':>8}  {'σ_min':>6}  {'σ_max':>6}")
    print("-" * 75)
    for s in all_summaries:
        print(
            f"  {s['sigma_label']:<20} "
            f"{s['best_acc']*100:>7.2f}%  "
            f"{s['final_acc']*100:>8.2f}%  "
            f"{s['final_epsilon']:>8.4f}  "
            f"{s['acc_per_epsilon_final']:>7.2f}  "
            f"{s['sigma_min']:>6.1f}  "
            f"{s['sigma_max']:>6.1f}"
        )
    print(f"{'='*75}")
    print(f"\nAG best schedule: σ_start={best_sigma_start:.2f}  σ_end={best_sigma_end:.2f}")
    print(f"CSVs salvos em: {out_dir}")


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    main()
