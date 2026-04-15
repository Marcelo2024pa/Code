#!/usr/bin/env python
"""
Resume AG optimization from cached results in fl_curves_history.json.

Strategy:
  1. Load all AG evaluations from curves_history (sigma_per_client with variance > 0)
  2. Pre-populate the FitnessEvaluator cache with those results
  3. Re-run GA with seed=42 — since it's deterministic, all previously evaluated
     individuals will be cache hits (instant), and GA resumes from where it stopped
  4. Run final evaluation with best sigma found
"""

import copy
import hashlib
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
from utils.data_utils import read_client_data
from optimization.genetic_algorithm import GeneticAlgorithm
from optimization.fitness import FitnessEvaluator
from optimization.client_groups import ClientGrouper
from run_full_comparison import make_args, run_experiment, write_csv
from run_full_comparison import COMPARISON_ROUNDS_COLS, COMPARISON_SUMMARY_COLS

CURVES_HISTORY_FILE = "/tmp/fl_curves_history.json"
INDIVIDUALS_CSV = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "results", "csv", "optimization_individuals.csv")
)


def _sigma_to_key(sigma_vector):
    rounded = [round(s, 4) for s in sigma_vector]
    return hashlib.md5(json.dumps(rounded).encode()).hexdigest()


def load_cache_from_csv():
    """Build fitness cache from optimization_individuals.csv (survives reboots)."""
    if not os.path.exists(INDIVIDUALS_CSV):
        return {}

    import csv
    cache = {}
    with open(INDIVIDUALS_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                spc = json.loads(row["sigma_per_client"])
                fitness = float(row["fitness"])
                accuracy = float(row["accuracy"])
                epsilon = float(row["epsilon"])
                key = _sigma_to_key(spc)
                # Keep best fitness if same sigma appears multiple times
                if key not in cache or fitness > cache[key][0]:
                    cache[key] = (fitness, accuracy, epsilon)
            except (KeyError, ValueError, json.JSONDecodeError):
                continue

    print(f"[resume] Pré-carregados {len(cache)} resultados do CSV de indivíduos.")
    return cache


def load_cache_from_history():
    """Build fitness cache from fl_curves_history.json (may not exist after reboot)."""
    if not os.path.exists(CURVES_HISTORY_FILE):
        return {}

    with open(CURVES_HISTORY_FILE) as f:
        history = json.load(f)

    def _is_ag_eval(run_dict):
        goal = run_dict.get("goal", "")
        if goal == "optimization":
            return True
        if goal and goal != "optimization":
            return False
        spc = run_dict.get("sigma_per_client")
        if not spc or len(spc) < 2:
            return False
        return len(set(round(s, 4) for s in spc)) > 1

    cache = {}
    for run_id, run in history.items():
        if not _is_ag_eval(run):
            continue
        spc = run.get("sigma_per_client")
        acc_list = run.get("acc", [])
        eps_list = run.get("epsilon", [])
        if not spc or not acc_list or not eps_list:
            continue
        acc = max(acc_list)
        epsilon = eps_list[-1]
        fitness = acc - 0.1 * epsilon
        key = _sigma_to_key(spc)
        cache[key] = (fitness, acc, epsilon)

    print(f"[resume] Pré-carregados {len(cache)} resultados do histórico JSON.")
    return cache


def load_last_generation_population(individuals_csv, target_generation):
    """
    Reads optimization_individuals.csv and reconstructs the population
    of `target_generation` as a list of Individual objects with fitness set.
    Returns (population, generation_found) or (None, 0) if not found.
    """
    import csv as _csv
    from optimization.genetic_algorithm import Individual

    if not os.path.exists(individuals_csv):
        return None, 0

    rows_by_gen = {}
    with open(individuals_csv, newline="") as f:
        for row in _csv.DictReader(f):
            try:
                gen = int(row["generation"])
                rows_by_gen.setdefault(gen, []).append(row)
            except (KeyError, ValueError):
                continue

    if not rows_by_gen:
        return None, 0

    # Use the requested generation, or the latest completed one
    if target_generation > 0 and target_generation in rows_by_gen:
        gen = target_generation
    else:
        gen = max(rows_by_gen.keys())

    rows = sorted(rows_by_gen[gen], key=lambda r: int(r.get("individual_idx", 0)))
    population = []
    for row in rows:
        try:
            spc = json.loads(row["sigma_per_client"])
            ind = Individual(spc)
            ind.fitness  = float(row["fitness"])
            ind.accuracy = float(row["accuracy"])
            ind.epsilon  = float(row["epsilon"])
            population.append(ind)
        except (KeyError, ValueError, json.JSONDecodeError):
            continue

    print(f"[resume] Geração {gen} carregada: {len(population)} indivíduos com fitness já computado.")
    return population, gen


def run_ag_optimization_true_resume(rounds, num_clients, device, population_size,
                                     total_generations, resume_from_gen=None):
    """
    True resume: loads population from `resume_from_gen` and continues
    the GA from the next generation, skipping already-evaluated work.
    """
    args = make_args(1.0, rounds, num_clients, device, privacy=True)
    args.goal                = "optimization"
    args.population          = population_size
    args.ga_generations      = total_generations
    args.sigma_min           = 0.1
    args.sigma_max           = 20.0
    args.mutation_rate       = 0.2
    args.mutation_sigma      = 1.9
    args.crossover_rate      = 0.8
    args.elitism             = 3
    args.tournament_size     = 5
    args.seed                = None
    args.fitness_type        = "linear"
    args.lambda_weight       = 0.1
    args.device_ids          = [0, 1]   # both GPUs
    args.algorithm           = "SCAFFOLD"
    args.verbose             = False
    # Per-round sigma schedule: GA optimizes (sigma_start, sigma_end) per client
    args.sigma_schedule_type = "linear"

    _csv_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "results", "csv")
    )

    # Load the last completed generation's population
    initial_pop, completed_gen = load_last_generation_population(
        INDIVIDUALS_CSV,
        target_generation=resume_from_gen or 0,
    )

    if initial_pop is None or completed_gen == 0:
        print("[resume] Nenhuma geração anterior encontrada. Iniciando do zero.")
        start_gen = 0
        initial_pop = None
    else:
        start_gen = completed_gen  # evolve() will start from gen index `completed_gen`
        print(f"[resume] Continuando a partir da geração {completed_gen + 1}/{total_generations}")

    client_samples = [
        len(read_client_data(args.dataset, i, is_train=True))
        for i in range(num_clients)
    ]

    fitness_evaluator = FitnessEvaluator(
        args=args,
        server_class=SCAFFOLD,
        client_class=clientSCAFFOLD,
        fitness_type=args.fitness_type,
        lambda_weight=args.lambda_weight,
        use_cache=True,
        device_ids=args.device_ids,
        schedule_type=args.sigma_schedule_type,
    )

    # Also pre-populate cache so any re-evaluation attempts are instant
    preloaded = load_cache_from_csv()
    preloaded.update(load_cache_from_history())
    fitness_evaluator.cache.update(preloaded)
    print(f"[resume] Cache auxiliar: {len(fitness_evaluator.cache)} entradas.")

    client_grouper = ClientGrouper(num_groups=3)
    client_grouper.fit(client_samples)

    def fitness_func(sigma_vector):
        return fitness_evaluator.evaluate(sigma_vector, client_grouper, verbose=False)

    def parallel_fitness_func(sigma_vectors, on_result=None):
        return fitness_evaluator.evaluate_parallel(sigma_vectors, client_grouper,
                                                    verbose=False, on_result=on_result)

    # Each chromosome encodes [sigma_start_0..N-1, sigma_end_0..N-1]
    ga = GeneticAlgorithm(
        num_genes=num_clients * 2,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        population_size=population_size,
        generations=total_generations,
        mutation_rate=args.mutation_rate,
        mutation_sigma=args.mutation_sigma,
        crossover_rate=args.crossover_rate,
        elitism=args.elitism,
        tournament_size=args.tournament_size,
        seed=None,
        csv_dir=_csv_dir,
    )

    best = ga.evolve(
        fitness_func=fitness_func,
        parallel_fitness_func=parallel_fitness_func,
        verbose=True,
        initial_population=initial_pop,
        start_generation=start_gen,
    )
    return best.sigma_vector


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Resume AG optimization from last completed generation")
    parser.add_argument("--rounds",      type=int, default=25)
    parser.add_argument("--clients",     type=int, default=20)
    parser.add_argument("--population",  type=int, default=30)
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--from_gen",    type=int, default=0,
                        help="Resume from this specific generation (0=auto-detect last)")
    cli = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  OTIMIZAÇÃO AG — GPUs: 0 + 1 (paralelo)")
    print(f"  Rounds: {cli.rounds} | Clientes: {cli.clients}")
    print(f"  Pop: {cli.population} | Gerações: {cli.generations}")
    print(f"  sigma∈[0.1, 20.0] | elitism=3 | tournament=5")
    print(f"{'='*60}\n")

    # True resume: load last generation's population and continue from next gen
    best_sigma = run_ag_optimization_true_resume(
        rounds=cli.rounds,
        num_clients=cli.clients,
        device=device,
        population_size=cli.population,
        total_generations=cli.generations,
        resume_from_gen=cli.from_gen,
    )

    N = cli.clients
    print(f"\nMelhor schedule encontrado:")
    print(f"  σ_start = {[round(s,2) for s in best_sigma[:N]]}")
    print(f"  σ_end   = {[round(s,2) for s in best_sigma[N:]]}")

    # Final evaluation with best schedule (sigma_start used as sigma_per_client)
    sigma_start = best_sigma[:N]
    sigma_end   = best_sigma[N:]
    sigma_schedule = {i: (sigma_start[i], sigma_end[i]) for i in range(N)}

    print("\n" + "="*60)
    print("  AVALIAÇÃO FINAL com schedule adaptativo do AG")
    print("="*60)
    result = run_experiment(
        label="sigma_adaptive_AG",
        run_type="adaptive",
        sigma_per_client=sigma_start,
        sigma_schedule=sigma_schedule,
        privacy=True,
        rounds=cli.rounds,
        num_clients=cli.clients,
        device=device,
    )

    out_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "results", "csv")
    )
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    write_csv(
        os.path.join(out_dir, f"ag_final_rounds_{ts}.csv"),
        COMPARISON_ROUNDS_COLS,
        result["rows"],
    )
    write_csv(
        os.path.join(out_dir, f"ag_final_summary_{ts}.csv"),
        COMPARISON_SUMMARY_COLS,
        [result["summary"]],
    )

    print("\nPronto!")


if __name__ == "__main__":
    main()
