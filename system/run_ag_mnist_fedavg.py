#!/usr/bin/env python
"""
Otimização AG — FedAvg / MNIST

Configuração base (consistente com run_fixed_sigma_mnist.py):
  - Dataset   : MNIST  (non-IID Dirichlet, alpha=0.2, 20 clientes)
  - Algoritmo : FedAvg
  - join_ratio: 1.0   (todos os clientes por round)
  - Cromossomo: [σ_start, σ_end, C] — 3 genes
  - σ(t)      = σ_start + (σ_end − σ_start) · t/T
  - C         = max_grad_norm (clipping, fixo)
  - Fitness   : Acc − λ·ε   (λ=0.1)
  - σ range   : [0.5, 20.0]  |  C range: [0.1, 5.0]

Saídas em tempo real (results/csv/):
  ag_mnist_fedavg_latest.csv         — rounds, append imediato após cada exp.
  ag_mnist_fedavg_summary_latest.csv — summaries, append imediato
  ag_mnist_fedavg_rounds_<ts>.csv    — cópia timestampada ao final
  ag_mnist_fedavg_summary_<ts>.csv   — cópia timestampada ao final

Uso:
  cd system
  python run_ag_mnist_fedavg.py [--rounds 50] [--ag_rounds 15]
                                 [--clients 20] [--population 8]
                                 [--generations 8] [--device_id 0]
"""

import argparse
import copy
import csv
import fcntl
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

from flcore.servers.serveravg   import FedAvg
from flcore.clients.clientavg   import clientAVG
from flcore.trainmodel.models   import FedAvgCNN
from optimization.genetic_algorithm import GeneticAlgorithm
from optimization.fitness           import FitnessEvaluator


class _DummyGroups:
    """Placeholder picklable pelo multiprocessing (sem estado necessário)."""
    pass


# ── Colunas CSV ───────────────────────────────────────────────────────────────

ROUNDS_COLS = [
    "timestamp", "label", "run_type", "algorithm", "dataset",
    "round", "total_rounds",
    "test_acc", "train_loss", "epsilon",
    "sigma_start", "sigma_end", "clip",
    "sigma_mean", "sigma_std", "sigma_min", "sigma_max",
    "acc_per_epsilon",
]

SUMMARY_COLS = [
    "timestamp", "label", "run_type", "algorithm", "dataset",
    "best_acc", "final_acc", "final_epsilon", "acc_per_epsilon_final",
    "sigma_start", "sigma_end", "clip",
    "sigma_mean", "sigma_std", "sigma_min", "sigma_max",
    "total_rounds",
]


# ── Args ──────────────────────────────────────────────────────────────────────

class _Args:
    pass


def make_args(sigma: float, rounds: int, num_clients: int,
              device: torch.device, privacy: bool = True) -> object:
    """Constrói args para FedAvg / MNIST — espelhado em run_fixed_sigma_mnist.py."""
    a = _Args()
    a.dataset                   = "mnist"
    a.num_clients               = num_clients
    a.global_rounds             = rounds
    a.local_epochs              = 1
    a.local_learning_rate       = 0.005
    a.batch_size                = 32
    a.join_ratio                = 1.0          # todos os clientes
    a.random_join_ratio         = False
    a.num_classes               = 10
    a.privacy                   = privacy
    a.dp_sigma                  = sigma
    a.max_grad_norm             = 1.0
    a.top_cnt                   = 100
    a.auto_break                = False
    a.eval_gap                  = 1
    a.save_folder_name          = "results"
    a.algorithm                 = "FedAvg"
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
    a.sigma_schedule            = None
    a.device                    = device
    a.device_id                 = str(device).replace("cuda:", "")
    a.tensorboard               = False
    # FedAvgCNN para MNIST: grayscale, dim=1024 (4×4×64)
    a.model = FedAvgCNN(in_features=1, num_classes=10, dim=1024).to(device)
    return a


# ── Sigma stats ───────────────────────────────────────────────────────────────

def sigma_stats(sigma_per_client, sigma_start=None, sigma_end=None):
    if sigma_start is not None and sigma_end is not None:
        lo  = min(sigma_start, sigma_end)
        hi  = max(sigma_start, sigma_end)
        mid = (sigma_start + sigma_end) / 2.0
        return {"sigma_mean": mid, "sigma_std": abs(sigma_end - sigma_start) / 2.0,
                "sigma_min": lo, "sigma_max": hi}
    arr = np.array([float(s) for s in sigma_per_client])
    return {"sigma_mean": float(np.mean(arr)), "sigma_std": float(np.std(arr)),
            "sigma_min": float(np.min(arr)),   "sigma_max": float(np.max(arr))}


# ── Run experiment ────────────────────────────────────────────────────────────

def run_experiment(label: str, run_type: str,
                   sigma_per_client: list, privacy: bool,
                   rounds: int, num_clients: int,
                   device: torch.device,
                   sigma_schedule: dict = None,
                   sigma_start: float = None,
                   sigma_end: float = None,
                   clip: float = None) -> dict:
    """Roda um experimento FedAvg/MNIST e retorna dict com rows e summary."""
    sigma_val = sigma_per_client[0] if len(set(sigma_per_client)) == 1 else sigma_per_client[0]
    args = make_args(sigma_val, rounds, num_clients, device, privacy)
    args.sigma_per_client = sigma_per_client
    args.sigma_schedule   = sigma_schedule
    args.privacy          = privacy
    args.goal             = label
    if clip is not None:
        args.max_grad_norm = clip

    print(f"\n{'='*65}")
    print(f"  {label}  |  FedAvg  |  {'DP' if privacy else 'SEM DP'}  |  {rounds} rounds  |  MNIST")
    if sigma_schedule and sigma_start is not None and sigma_end is not None:
        print(f"  σ schedule: {sigma_start:.1f} → {sigma_end:.1f}  (linear, todos os clientes)")
    else:
        print(f"  σ range: [{min(sigma_per_client):.2f}, {max(sigma_per_client):.2f}]")
    print(f"{'='*65}")

    server = FedAvg(args, times=0)
    for client in server.clients:
        client.dp_sigma = sigma_per_client[client.id]

    server.train()

    acc  = server.rs_test_acc
    loss = server.rs_train_loss
    eps  = server.rs_epsilon if server.rs_epsilon else [0.0] * len(acc)

    n = min(len(acc), len(loss), len(eps))
    acc, loss, eps = acc[:n], loss[:n], eps[:n]

    stats = sigma_stats(sigma_per_client, sigma_start=sigma_start, sigma_end=sigma_end)
    clip_used = clip if clip is not None else 1.0
    ts = datetime.now().isoformat(timespec="seconds")

    rows = []
    for i, (a, l, e) in enumerate(zip(acc, loss, eps)):
        rows.append({
            "timestamp":    ts,
            "label":        label,
            "run_type":     run_type,
            "algorithm":    "FedAvg",
            "dataset":      "mnist",
            "round":        i,
            "total_rounds": rounds,
            "test_acc":     round(float(a), 6),
            "train_loss":   round(float(l), 6),
            "epsilon":      round(float(e), 6),
            "sigma_start":  round(sigma_start, 4) if sigma_start is not None else round(sigma_per_client[0], 4),
            "sigma_end":    round(sigma_end,   4) if sigma_end   is not None else round(sigma_per_client[0], 4),
            "clip":         round(clip_used, 4),
            "sigma_mean":   round(stats["sigma_mean"], 4),
            "sigma_std":    round(stats["sigma_std"],  4),
            "sigma_min":    round(stats["sigma_min"],  4),
            "sigma_max":    round(stats["sigma_max"],  4),
            "acc_per_epsilon": round(float(a) / (float(e) + 1e-6), 4),
        })

    summary = {
        "timestamp":           ts,
        "label":               label,
        "run_type":            run_type,
        "algorithm":           "FedAvg",
        "dataset":             "mnist",
        "best_acc":            round(max(acc), 6)           if acc else 0.0,
        "final_acc":           round(float(acc[-1]), 6)     if acc else 0.0,
        "final_epsilon":       round(float(eps[-1]), 6)     if eps else 0.0,
        "acc_per_epsilon_final": round(float(acc[-1]) / (float(eps[-1]) + 1e-6), 4) if acc else 0.0,
        "sigma_start":         round(sigma_start, 4) if sigma_start is not None else round(sigma_per_client[0], 4),
        "sigma_end":           round(sigma_end,   4) if sigma_end   is not None else round(sigma_per_client[0], 4),
        "clip":                round(clip_used, 4),
        "sigma_mean":          round(stats["sigma_mean"], 4),
        "sigma_std":           round(stats["sigma_std"],  4),
        "sigma_min":           round(stats["sigma_min"],  4),
        "sigma_max":           round(stats["sigma_max"],  4),
        "total_rounds":        rounds,
    }

    del server
    torch.cuda.empty_cache()

    return {"rows": rows, "summary": summary}


# ── AG Optimization ───────────────────────────────────────────────────────────

def run_ag_optimization(ag_rounds: int, num_clients: int, device: torch.device,
                        population: int, generations: int,
                        device_ids: list = None,
                        patience: int = 6, tol: float = 5e-4,
                        min_diversity: float = 0.01) -> tuple:
    """
    Roda o AG para FedAvg/MNIST e retorna (sigma_start, sigma_end, clip).

    Cromossomo: 3 genes [σ_start, σ_end, C]
      - σ(t) = σ_start + (σ_end − σ_start) · t/T  (schedule linear global)
      - C = max_grad_norm (clipping threshold, fixo ao longo dos rounds)

    Calibrado com resultados fixos (MNIST, 20 rounds, 20 clientes):
      σ=1 → Acc=47.6%  σ=4 → Acc=48.1% (sweet spot)  σ=12 → 37.9%  σ=20 → 28.5%
      ⇒ busca restrita a σ ∈ [0.5, 10.0] — acima de 10 destrói accuracy.
      C ∈ [0.3, 2.0] cobre intervalo prático da literatura (Abadi 2016, Andrew 2021).

    λ = 0.1  →  Fitness = Acc − 0.1·ε
    Avaliação paralela: 2 GPUs via multiprocessing (imap_unordered).
    """
    # ── Espaço de busca (calibrado com resultados fixos) ──────────────────────
    SIGMA_MIN     = 0.5
    SIGMA_MAX     = 10.0   # σ>10 destrói accuracy; não faz sentido buscar acima
    CLIP_MIN      = 0.3
    CLIP_MAX      = 2.0    # range prático Andrew et al. 2021
    LAMBDA_WEIGHT = 0.1

    # Bounds e mutation_sigma por gene: [σ_start, σ_end, C]
    GENE_BOUNDS = [
        (SIGMA_MIN, SIGMA_MAX),   # σ_start
        (SIGMA_MIN, SIGMA_MAX),   # σ_end
        (CLIP_MIN,  CLIP_MAX),    # C
    ]
    # mutation_sigma = 15% do range de cada gene → perturbações úteis sem fugir dos bounds
    MUTATION_SIGMAS = [
        (SIGMA_MAX - SIGMA_MIN) * 0.15,   # ≈1.43 para σ_start
        (SIGMA_MAX - SIGMA_MIN) * 0.15,   # ≈1.43 para σ_end
        (CLIP_MAX  - CLIP_MIN)  * 0.20,   # ≈0.34 para C
    ]

    _device_ids = device_ids if device_ids else [int(str(device).replace("cuda:", "") or 0)]

    args = make_args(1.0, ag_rounds, num_clients, device, privacy=True)
    args.goal                = "optimization"
    args.fitness_type        = "linear"
    args.lambda_weight       = LAMBDA_WEIGHT
    args.device_ids          = _device_ids
    args.sigma_schedule_type = "linear"

    fitness_evaluator = FitnessEvaluator(
        args=args,
        server_class=FedAvg,
        client_class=clientAVG,
        fitness_type="linear",
        lambda_weight=LAMBDA_WEIGHT,
        use_cache=True,
        device_ids=_device_ids,
        n_evals=1,
        base_seed=42,
        schedule_type="linear",
    )

    # Função sequencial (usada dentro do mesmo processo quando parallel=False)
    def fitness_func(sigma_vector):
        return fitness_evaluator.evaluate(sigma_vector, _DummyGroups(), verbose=False)

    # Função paralela: avalia lote de indivíduos via multiprocessing (2 GPUs)
    def parallel_fitness_func(sigma_vectors, on_result=None):
        return fitness_evaluator.evaluate_parallel(
            sigma_vectors, _DummyGroups(), verbose=False, on_result=on_result
        )

    # ── Indivíduos semente ────────────────────────────────────────────────────
    # Estratégia para superar o melhor fixo: σ=4, C=1.0 → Fitness=0.422
    #
    # Alavancas do AG sobre o fixo:
    #   1. C menor → noise_std = σ·C menor → melhor accuracy sem mudar ε
    #   2. Schedule decrescente → rounds iniciais (alto σ) gastam menos ε;
    #      rounds finais (baixo σ) convergem com menos ruído → tradeoff melhor
    #   3. Combinação ótima σ,C que o grid fixo [1,4,12,20] com C=1.0 não cobre
    #
    # Fitness alvo: > 0.422 (melhor fixo σ=4, C=1.0)
    # noise_std = σ·C → reduzir C é tão eficaz quanto reduzir σ para ruído,
    # mas com menos impacto em ε (ε depende de σ, não de C diretamente).
    seed_individuals = [
        # ── Vizinhança do melhor fixo (σ=4, C=1.0) ──────────────────────────
        [4.0,  4.0, 0.5],   # igual ao fixo mas C apertado: noise=2.0 vs 4.0 → melhor acc
        [4.0,  4.0, 0.7],   # fixo σ=4, C levemente apertado
        [4.0,  1.0, 0.5],   # schedule decrescente do sweet spot + C apertado
        [4.0,  2.0, 0.7],   # decaimento moderado + C apertado
        # ── Schedule alto→baixo (hipótese central de tradeoff) ───────────────
        [8.0,  2.0, 0.5],   # alto→médio, C apertado — ε baixo + acc razoável
        [6.0,  2.0, 0.7],   # médio→baixo, C moderado
        [8.0,  4.0, 0.5],   # alto→sweet spot + C apertado
        [10.0, 2.0, 0.5],   # máximo range σ + C apertado
        # ── Exploração de C muito baixo (máximo benefício noise_std) ─────────
        [4.0,  2.0, 0.3],   # C mínimo — noise_std = 1.2 vs 4.0 no fixo
        [6.0,  1.0, 0.5],   # decaimento total + C apertado
    ]

    _csv_dir = os.path.abspath(
        os.path.join(BASE_DIR, '..', 'results', 'csv')
    )

    use_parallel = len(_device_ids) > 1

    ga = GeneticAlgorithm(
        num_genes=3,
        sigma_min=SIGMA_MIN,
        sigma_max=SIGMA_MAX,
        population_size=population,
        generations=generations,
        mutation_rate=0.3,           # mais alto p/ manter diversidade em 20 gerações
        mutation_sigma=MUTATION_SIGMAS[0],
        crossover_rate=0.8,
        elitism=2,                   # preserva os 2 melhores a cada geração
        tournament_size=3,
        seed=42,
        csv_dir=_csv_dir,
        seed_individuals=seed_individuals,
        patience=patience,
        tol=tol,
        min_diversity=min_diversity,
        gene_bounds=GENE_BOUNDS,
        mutation_sigmas=MUTATION_SIGMAS,
    )

    print(f"\n{'='*65}")
    print(f"  AG FedAvg/MNIST — pop={population}  gen={generations}  ag_rounds={ag_rounds}")
    print(f"  Genes: [σ_start, σ_end, C]")
    print(f"  σ ∈ [{SIGMA_MIN}, {SIGMA_MAX}]  |  C ∈ [{CLIP_MIN}, {CLIP_MAX}]")
    print(f"  mutation_rate=0.3  |  mutation_σ: {[f'{m:.2f}' for m in MUTATION_SIGMAS]}")
    print(f"  elitism=2  |  crossover=0.8  |  tournament=3")
    print(f"  Fitness: Acc − {LAMBDA_WEIGHT}·ε  (λ=0.1)")
    print(f"  Early stopping: patience={patience}  tol={tol:.0e}  min_diversity={min_diversity}")
    print(f"  GPUs: {_device_ids}  ({'paralelo' if use_parallel else 'sequencial'})")
    print(f"  {len(seed_individuals)} indivíduos semente")
    print(f"{'='*65}")

    pfunc = parallel_fitness_func if use_parallel else None
    best = ga.evolve(fitness_func=fitness_func, parallel_fitness_func=pfunc, verbose=True)
    return float(best.sigma_vector[0]), float(best.sigma_vector[1]), float(best.sigma_vector[2])


# ── CSV helpers ───────────────────────────────────────────────────────────────

def write_csv(path: str, cols: list, rows: list) -> None:
    """Sobrescreve o arquivo inteiro (usado para cópias timestampadas no final)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  -> Salvo: {path}  ({len(rows)} linhas)")


def init_csv(path: str, cols: list) -> None:
    """Cria o arquivo com cabeçalho (ou limpa se já existir) no início da run."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=cols, extrasaction="ignore").writeheader()


def append_csv(path: str, cols: list, rows: list) -> None:
    """
    Append imediato de rows no CSV, com fcntl lock (process-safe).
    Chamado logo após cada run_experiment() terminar — garante salvamento
    mesmo se o processo crashar antes do final.
    """
    if not rows:
        return
    lock_path = path + ".lock"
    with open(lock_path, "w") as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)
        try:
            with open(path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
                writer.writerows(rows)
        finally:
            fcntl.flock(lock_f, fcntl.LOCK_UN)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AG otimização FedAvg/MNIST")
    parser.add_argument("--rounds",      type=int,   default=20,   help="Rounds para avaliação final do melhor cromossomo")
    parser.add_argument("--ag_rounds",   type=int,   default=20,   help="Rounds inner-loop do AG (mesmo dos fixos → sinal fiel)")
    parser.add_argument("--clients",     type=int,   default=20,   help="Número de clientes")
    parser.add_argument("--population",  type=int,   default=10,   help="Tamanho da população AG")
    parser.add_argument("--generations", type=int,   default=20,   help="Gerações do AG")
    parser.add_argument("--patience",    type=int,   default=6,    help="Gerações sem melhora para early stop")
    parser.add_argument("--tol",         type=float, default=5e-4, help="Melhora mínima de fitness")
    parser.add_argument("--device_ids",  type=int,   nargs="+",    default=[0, 1],
                        help="IDs das GPUs para avaliação paralela (ex: --device_ids 0 1)")
    cli = parser.parse_args()

    device = torch.device(f"cuda:{cli.device_ids[0]}" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice principal: {device}  |  GPUs para AG: {cli.device_ids}")
    print(f"Rounds finais: {cli.rounds}  |  AG inner rounds: {cli.ag_rounds}")
    print(f"Clientes: {cli.clients}  |  Pop: {cli.population}  |  Gen: {cli.generations}")

    out_dir = os.path.abspath(os.path.join(BASE_DIR, '..', 'results', 'csv'))
    os.makedirs(out_dir, exist_ok=True)

    latest_rounds_csv   = os.path.join(out_dir, "ag_mnist_fedavg_latest.csv")
    latest_summary_csv  = os.path.join(out_dir, "ag_mnist_fedavg_summary_latest.csv")

    # Inicializa CSVs com cabeçalho no começo da run (limpa corridas anteriores)
    init_csv(latest_rounds_csv,  ROUNDS_COLS)
    init_csv(latest_summary_csv, SUMMARY_COLS)
    print(f"  CSVs inicializados: {latest_rounds_csv}")

    all_rows      = []
    all_summaries = []

    def _save(result: dict) -> None:
        """Append imediato após cada experimento — sobrevive a crash."""
        all_rows.extend(result["rows"])
        all_summaries.append(result["summary"])
        append_csv(latest_rounds_csv,  ROUNDS_COLS,  result["rows"])
        append_csv(latest_summary_csv, SUMMARY_COLS, [result["summary"]])
        print(f"  [CSV] {result['summary']['label']} salvo ({len(result['rows'])} rounds)")

    # ── 1. Baseline fixo de referência (σ=4, C=1.0) — carregado dos resultados já existentes ──
    # Adicionamos o melhor fixo diretamente ao CSV para comparação visual no dashboard.
    # Não roda treinamento — usa os valores conhecidos dos experimentos fixos.
    BASELINE_FITNESS = 0.4222   # Acc=0.4810 − 0.1×0.5881
    baseline_summary = {
        "timestamp":             datetime.now().isoformat(timespec="seconds"),
        "label":                 "sigma_4_fixed_ref",
        "run_type":              "fixed_reference",
        "algorithm":             "FedAvg",
        "dataset":               "mnist",
        "best_acc":              0.4810,
        "final_acc":             0.4543,
        "final_epsilon":         0.5881,
        "acc_per_epsilon_final": round(0.4543 / (0.5881 + 1e-6), 4),
        "sigma_start":           4.0,
        "sigma_end":             4.0,
        "clip":                  1.0,
        "sigma_mean":            4.0,
        "sigma_std":             0.0,
        "sigma_min":             4.0,
        "sigma_max":             4.0,
        "total_rounds":          20,
    }
    all_summaries.append(baseline_summary)
    append_csv(latest_summary_csv, SUMMARY_COLS, [baseline_summary])
    print(f"\n  [Referência] σ=4 fixo: Acc={baseline_summary['best_acc']*100:.2f}%  "
          f"ε={baseline_summary['final_epsilon']:.4f}  Fitness={BASELINE_FITNESS:.4f}")
    print(f"  O AG precisa superar Fitness > {BASELINE_FITNESS:.4f}")

    # ── 2. AG — busca [σ_start, σ_end, C] ────────────────────────────────────
    torch.manual_seed(42)
    np.random.seed(42)

    best_start, best_end, best_clip = run_ag_optimization(
        ag_rounds=cli.ag_rounds,
        num_clients=cli.clients,
        device=device,
        population=cli.population,
        generations=cli.generations,
        device_ids=cli.device_ids,
        patience=cli.patience,
        tol=cli.tol,
    )

    print(f"\nMelhor schedule: σ_start={best_start:.4f}  σ_end={best_end:.4f}  C={best_clip:.4f}")

    # ── 3. Avaliação final com melhor cromossomo ───────────────────────────────
    torch.manual_seed(42)
    np.random.seed(42)

    schedule_dict = {i: (best_start, best_end) for i in range(cli.clients)}
    _save(run_experiment(
        label="sigma_schedule_AG",
        run_type="adaptive",
        sigma_per_client=[best_start] * cli.clients,
        privacy=True,
        rounds=cli.rounds,
        num_clients=cli.clients,
        device=device,
        sigma_schedule=schedule_dict,
        sigma_start=best_start,
        sigma_end=best_end,
        clip=best_clip,
    ))

    # ── 4. Cópias timestampadas (arquivo completo, para histórico) ─────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    write_csv(os.path.join(out_dir, f"ag_mnist_fedavg_rounds_{ts}.csv"),  ROUNDS_COLS,  all_rows)
    write_csv(os.path.join(out_dir, f"ag_mnist_fedavg_summary_{ts}.csv"), SUMMARY_COLS, all_summaries)

    # ── 5. Tabela resumo ──────────────────────────────────────────────────────
    lambda_w = 0.1
    print(f"\n{'='*80}")
    print("  RESUMO — FedAvg / MNIST  |  Fitness = Acc − 0.1·ε")
    print(f"{'='*80}")
    print(f"{'Config':<25} {'Acc_max':>8} {'Acc_final':>10} {'ε_final':>9} {'Fitness':>8}  C     σ_range")
    print("-" * 80)
    for s in all_summaries:
        fitness_val = s['best_acc'] - lambda_w * s['final_epsilon']
        marker = " ◄ MELHOR" if s['label'] == "sigma_schedule_AG" and fitness_val > BASELINE_FITNESS else ""
        print(
            f"  {s['label']:<23} "
            f"{s['best_acc']*100:>7.2f}%  "
            f"{s['final_acc']*100:>8.2f}%  "
            f"{s['final_epsilon']:>8.4f}  "
            f"{fitness_val:>8.4f}  "
            f"{s['clip']:>4.2f}  "
            f"[{s['sigma_min']:.1f}→{s['sigma_max']:.1f}]"
            f"{marker}"
        )
    print(f"{'='*80}")

    ag_fitness = next(
        (s['best_acc'] - lambda_w * s['final_epsilon']
         for s in all_summaries if s['label'] == "sigma_schedule_AG"), None
    )
    if ag_fitness is not None:
        delta = ag_fitness - BASELINE_FITNESS
        status = f"SUPEROU o fixo em +{delta:.4f}" if delta > 0 else f"abaixo do fixo em {delta:.4f}"
        print(f"\n  AG vs referência (σ=4 fixo, Fitness={BASELINE_FITNESS:.4f}): {status}")
    print(f"\n  AG best: σ_start={best_start:.4f}  σ_end={best_end:.4f}  C={best_clip:.4f}")
    print(f"  CSVs em: {out_dir}")


if __name__ == "__main__":
    main()
