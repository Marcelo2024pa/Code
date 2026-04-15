# Genetic Algorithm for sigma optimization
# Based on arq_AG.md specification

import numpy as np
from typing import List, Tuple, Optional, Callable
import copy
import time
import os
import sys
# csv_logger is in system/utils — add system/ to path if needed
_system_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _system_dir not in sys.path:
    sys.path.insert(0, _system_dir)
from utils.csv_logger import log_opt_individual, log_opt_generation, CSV_DIR


class Individual:
    """Represents a candidate solution (chromosome)."""

    def __init__(self, sigma_vector: List[float]):
        self.sigma_vector = sigma_vector
        self.fitness: Optional[float] = None
        self.accuracy: Optional[float] = None
        self.epsilon: Optional[float] = None

    def __repr__(self):
        return f"Individual(σ={self.sigma_vector}, fitness={self.fitness:.4f})" if self.fitness else f"Individual(σ={self.sigma_vector})"

    def copy(self) -> 'Individual':
        ind = Individual(self.sigma_vector.copy())
        ind.fitness = self.fitness
        ind.accuracy = self.accuracy
        ind.epsilon = self.epsilon
        return ind


class GeneticAlgorithm:
    """
    Genetic Algorithm for optimizing sigma allocation.

    Chromosome: [σ_start, σ_end, C]  — global linear schedule + clipping threshold.
    σ decays (or increases) linearly from σ_start (round 0) to σ_end (round T).
    C (max_grad_norm) controls gradient sensitivity and is fixed across rounds.

    Parameters from arq_AG.md:
    - population_size: 5-10
    - generations: 5-10
    - mutation_rate: 0.1
    - crossover_rate: 0.8
    - elitism: 1-2
    """

    def __init__(
        self,
        num_genes: int = 2,
        sigma_min: float = 0.1,
        sigma_max: float = 2.0,
        population_size: int = 10,
        generations: int = 10,
        mutation_rate: float = 0.2,
        mutation_sigma: float = 0.1,
        crossover_rate: float = 0.8,
        elitism: int = 2,
        tournament_size: int = 3,
        seed: Optional[int] = None,
        csv_dir: Optional[str] = None,
        seed_individuals: Optional[List[List[float]]] = None,
        patience: int = 3,
        tol: float = 1e-4,
        min_diversity: float = 0.05,
        gene_bounds: Optional[List[Tuple[float, float]]] = None,
        mutation_sigmas: Optional[List[float]] = None,
    ):
        """
        Args:
            num_genes: Number of genes in the chromosome
            sigma_min: Default minimum value (used if gene_bounds not provided)
            sigma_max: Default maximum value (used if gene_bounds not provided)
            population_size: Number of individuals
            generations: Number of generations
            mutation_rate: Probability of mutation per gene
            mutation_sigma: Default std for Gaussian mutation (used if mutation_sigmas not provided)
            crossover_rate: Probability of crossover
            elitism: Number of best individuals to keep
            tournament_size: Size for tournament selection
            seed: Random seed for reproducibility
            patience: Early stopping — max generations without improvement.
            tol: Minimum absolute improvement in best fitness to reset patience.
            min_diversity: Early stopping — stop if normalized std across all genes
                falls below this threshold (population collapsed to a single point).
            gene_bounds: Per-gene (min, max) bounds. If None, uses
                [(sigma_min, sigma_max)] * num_genes for all genes.
            mutation_sigmas: Per-gene std for Gaussian mutation. If None, uses
                [mutation_sigma] * num_genes for all genes.
        """
        self.num_genes = num_genes
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.mutation_sigma = mutation_sigma
        self.crossover_rate = crossover_rate
        self.elitism = elitism
        self.tournament_size = tournament_size
        self.patience = patience
        self.tol = tol
        self.min_diversity = min_diversity

        # Per-gene bounds and mutation scales
        self.gene_bounds: List[Tuple[float, float]] = (
            gene_bounds if gene_bounds is not None
            else [(sigma_min, sigma_max)] * num_genes
        )
        self.mutation_sigmas: List[float] = (
            mutation_sigmas if mutation_sigmas is not None
            else [mutation_sigma] * num_genes
        )

        if seed is not None:
            np.random.seed(seed)

        self.csv_dir = csv_dir or CSV_DIR
        self.seed_individuals = seed_individuals or []
        self.population: List[Individual] = []
        self.best_individual: Optional[Individual] = None
        self.history: List[dict] = []

    def initialize_population(self) -> None:
        """Initialize population with seed individuals + random fill."""
        self.population = []

        # Place seed individuals first — clip each gene to its own bounds
        for sv in self.seed_individuals:
            # Pad or truncate seed to num_genes
            sv_padded = list(sv) + [0.0] * max(0, self.num_genes - len(sv))
            sv_padded = sv_padded[:self.num_genes]
            clipped = [
                float(np.clip(s, lo, hi))
                for s, (lo, hi) in zip(sv_padded, self.gene_bounds)
            ]
            self.population.append(Individual(clipped))
            if len(self.population) >= self.population_size:
                break

        # Fill remaining slots with random individuals — per-gene uniform
        while len(self.population) < self.population_size:
            sigma_vector = [
                float(np.random.uniform(lo, hi))
                for lo, hi in self.gene_bounds
            ]
            self.population.append(Individual(sigma_vector))

    def evaluate_population(
        self,
        fitness_func: Callable[[List[float]], Tuple[float, float, float]],
        parallel: bool = False,
        generation: int = 0,
        total_generations: int = 0,
    ) -> None:
        """
        Evaluate fitness for all individuals.

        Args:
            fitness_func: Function that takes sigma_vector and returns (fitness, acc, epsilon)
            parallel: Whether to use parallel evaluation
            generation: Current generation number (for immediate CSV checkpointing)
            total_generations: Total generations planned (for immediate CSV checkpointing)
        """
        # Filter individuals without fitness
        to_evaluate = [ind for ind in self.population if ind.fitness is None]

        if not to_evaluate:
            return

        # Map ind → its index in self.population for logging
        pop_indices = {id(ind): i for i, ind in enumerate(self.population)}

        if parallel:
            # Get sigma vectors for parallel evaluation
            sigma_vectors = [ind.sigma_vector for ind in to_evaluate]

            # Callback fires as each worker finishes (imap_unordered),
            # enabling real-time CSV checkpointing and dashboard updates.
            def _on_result(task_idx, result):
                fitness, acc, epsilon = result
                ind = to_evaluate[task_idx]
                ind.fitness = fitness
                ind.accuracy = acc
                ind.epsilon = epsilon
                if generation > 0:
                    log_opt_individual(
                        generation=generation,
                        total_generations=total_generations,
                        individual_idx=pop_indices.get(id(ind), -1),
                        run_id=getattr(ind, "_run_id", ""),
                        fitness=fitness,
                        accuracy=acc,
                        epsilon=epsilon,
                        sigma_vector=ind.sigma_vector,
                        csv_dir=self.csv_dir,
                    )

            fitness_func(sigma_vectors, on_result=_on_result)
        else:
            for ind in to_evaluate:
                fitness, acc, epsilon = fitness_func(ind.sigma_vector)
                ind.fitness = fitness
                ind.accuracy = acc
                ind.epsilon = epsilon
                # Checkpoint immediately — survives mid-generation shutdown
                if generation > 0:
                    log_opt_individual(
                        generation=generation,
                        total_generations=total_generations,
                        individual_idx=pop_indices.get(id(ind), -1),
                        run_id=getattr(ind, "_run_id", ""),
                        fitness=fitness,
                        accuracy=acc,
                        epsilon=epsilon,
                        sigma_vector=ind.sigma_vector,
                        csv_dir=self.csv_dir,
                    )

    def tournament_selection(self) -> Individual:
        """Select an individual using tournament selection."""
        tournament = np.random.choice(
            self.population, self.tournament_size, replace=False
        )
        winner = max(tournament, key=lambda x: x.fitness)
        return winner.copy()

    def uniform_crossover(
        self,
        parent1: Individual,
        parent2: Individual
    ) -> Tuple[Individual, Individual]:
        """Apply uniform crossover to two parents."""
        if np.random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()

        child1_sigma = []
        child2_sigma = []

        for i in range(self.num_genes):
            if np.random.random() < 0.5:
                child1_sigma.append(parent1.sigma_vector[i])
                child2_sigma.append(parent2.sigma_vector[i])
            else:
                child1_sigma.append(parent2.sigma_vector[i])
                child2_sigma.append(parent1.sigma_vector[i])

        return Individual(child1_sigma), Individual(child2_sigma)

    def mutate(self, individual: Individual) -> Individual:
        """Apply Gaussian mutation with per-gene bounds and mutation scale."""
        mutated_sigma = []
        for gene, (lo, hi), mut_sig in zip(
            individual.sigma_vector, self.gene_bounds, self.mutation_sigmas
        ):
            if np.random.random() < self.mutation_rate:
                new_val = gene + np.random.normal(0, mut_sig)
                new_val = float(np.clip(new_val, lo, hi))
                mutated_sigma.append(new_val)
            else:
                mutated_sigma.append(gene)

        return Individual(mutated_sigma)

    def get_elite(self) -> List[Individual]:
        """Get the best individuals for elitism."""
        sorted_pop = sorted(
            self.population, key=lambda x: x.fitness, reverse=True
        )
        return [ind.copy() for ind in sorted_pop[:self.elitism]]

    def _check_convergence(
        self,
        no_improve_count: int,
        verbose: bool,
    ) -> Tuple[bool, str]:
        """
        Check two convergence criteria:

        1. Fitness plateau — best fitness did not improve by more than `tol`
           for `patience` consecutive generations.
        2. Population diversity collapse — the std of all sigma genes across
           the population dropped below `min_diversity` × sigma_range,
           meaning every individual encodes virtually the same solution.

        Returns:
            (converged: bool, reason: str)
        """
        # ── Criterion 1: fitness plateau ─────────────────────────────────────
        if no_improve_count >= self.patience:
            return True, (
                f"fitness plateau: no improvement > {self.tol:.1e} "
                f"for {self.patience} consecutive generations"
            )

        # ── Criterion 2: population diversity collapse ────────────────────────
        # Normalize each gene by its range before computing std,
        # so genes with different scales contribute equally.
        evaluated = [ind for ind in self.population if ind.fitness is not None]
        if len(evaluated) >= 2:
            normalized = []
            for ind in evaluated:
                row = []
                for g, (lo, hi) in zip(ind.sigma_vector, self.gene_bounds):
                    r = hi - lo
                    row.append((g - lo) / r if r > 0 else 0.0)
                normalized.append(row)
            diversity = float(np.std(normalized))
            if diversity < self.min_diversity:
                return True, (
                    f"population collapsed: normalized_std={diversity:.4f} "
                    f"< {self.min_diversity:.4f}"
                )

        return False, ""

    def evolve(
        self,
        fitness_func: Callable[[List[float]], Tuple[float, float, float]],
        parallel_fitness_func: Optional[Callable] = None,
        verbose: bool = True,
        callback: Optional[Callable] = None,
        initial_population: Optional[List['Individual']] = None,
        start_generation: int = 0,
    ) -> Individual:
        """
        Run the genetic algorithm.

        Args:
            fitness_func: Function to evaluate single individual
            parallel_fitness_func: Optional function for parallel evaluation
            verbose: Print progress
            callback: Optional callback after each generation

        Returns:
            Best individual found
        """
        start_time = time.time()

        # Initialize
        if verbose:
            print("=" * 60)
            print("GENETIC ALGORITHM - Sigma Optimization")
            print("=" * 60)
            print(f"Population: {self.population_size}")
            print(f"Generations: {self.generations}")
            print(f"Sigma range: [{self.sigma_min}, {self.sigma_max}]")
            print(f"Genes: {self.num_genes}")
            print(f"Early stopping: patience={self.patience}  tol={self.tol:.1e}  min_diversity={self.min_diversity}")
            if start_generation > 0:
                print(f"Resuming from generation {start_generation + 1}/{self.generations}")
            print("=" * 60)

        if initial_population is not None:
            # Resume: use injected pre-evaluated population
            self.population = [ind.copy() for ind in initial_population]
            # Set best from the injected population
            evaluated = [ind for ind in self.population if ind.fitness is not None]
            if evaluated:
                self.best_individual = max(evaluated, key=lambda x: x.fitness).copy()
        else:
            self.initialize_population()

        no_improve_count = 0
        prev_best_fitness: Optional[float] = (
            self.best_individual.fitness if self.best_individual else None
        )

        for gen in range(start_generation, self.generations):
            gen_start = time.time()

            if verbose:
                print(f"\n--- Generation {gen + 1}/{self.generations} ---")

            # Evaluate population (each individual is checkpointed to CSV immediately)
            if parallel_fitness_func is not None:
                self.evaluate_population(parallel_fitness_func, parallel=True,
                                         generation=gen + 1, total_generations=self.generations)
            else:
                self.evaluate_population(fitness_func, parallel=False,
                                         generation=gen + 1, total_generations=self.generations)

            # Update best and track improvement for early stopping
            current_best = max(self.population, key=lambda x: x.fitness)
            if self.best_individual is None or current_best.fitness > self.best_individual.fitness:
                self.best_individual = current_best.copy()

            if prev_best_fitness is None or (current_best.fitness - prev_best_fitness) > self.tol:
                no_improve_count = 0
            else:
                no_improve_count += 1
            prev_best_fitness = current_best.fitness

            # Record history
            fitnesses = [ind.fitness for ind in self.population]
            accuracies = [ind.accuracy for ind in self.population]
            epsilons = [ind.epsilon for ind in self.population]

            gen_record = {
                'generation': gen + 1,
                'best_fitness': current_best.fitness,
                'best_sigma': current_best.sigma_vector.copy(),
                'best_accuracy': current_best.accuracy,
                'best_epsilon': current_best.epsilon,
                'avg_fitness':  np.mean(fitnesses),
                'std_fitness':  np.std(fitnesses),
                'var_fitness':  np.var(fitnesses),
                'avg_accuracy': np.mean(accuracies),
                'std_accuracy': np.std(accuracies),
                'var_accuracy': np.var(accuracies),
                'avg_epsilon':  np.mean(epsilons),
                'std_epsilon':  np.std(epsilons),
                'var_epsilon':  np.var(epsilons),
                'time': time.time() - gen_start
            }
            self.history.append(gen_record)

            if verbose:
                print(f"  Best: σ={current_best.sigma_vector}")
                print(f"        Acc={current_best.accuracy:.4f}, ε={current_best.epsilon:.4f}")
                print(f"        Fitness={current_best.fitness:.4f}  (no_improve={no_improve_count}/{self.patience})")
                print(f"  Fitness  — avg={gen_record['avg_fitness']:.4f}  std={gen_record['std_fitness']:.4f}  var={gen_record['var_fitness']:.6f}")
                print(f"  Accuracy — avg={gen_record['avg_accuracy']:.4f}  std={gen_record['std_accuracy']:.4f}  var={gen_record['var_accuracy']:.6f}")
                print(f"  Epsilon  — avg={gen_record['avg_epsilon']:.4f}  std={gen_record['std_epsilon']:.4f}  var={gen_record['var_epsilon']:.6f}")
                print(f"  Time: {gen_record['time']:.1f}s")

            # ── CSV: resumo da geração ────────────────────────────────────────
            log_opt_generation(
                generation        = gen + 1,
                total_generations = self.generations,
                best_fitness      = gen_record["best_fitness"],
                best_accuracy     = gen_record["best_accuracy"],
                best_epsilon      = gen_record["best_epsilon"],
                avg_fitness       = gen_record["avg_fitness"],
                std_fitness       = gen_record["std_fitness"],
                var_fitness       = gen_record["var_fitness"],
                avg_accuracy      = gen_record["avg_accuracy"],
                std_accuracy      = gen_record["std_accuracy"],
                var_accuracy      = gen_record["var_accuracy"],
                avg_epsilon       = gen_record["avg_epsilon"],
                std_epsilon       = gen_record["std_epsilon"],
                var_epsilon       = gen_record["var_epsilon"],
                best_sigma        = current_best.sigma_vector,
                csv_dir           = self.csv_dir,
            )

            # Callback
            if callback:
                callback(gen, gen_record)

            # ── Early stopping check ──────────────────────────────────────────
            converged, reason = self._check_convergence(no_improve_count, verbose)
            if converged:
                if verbose:
                    print(f"\n  [Early stopping] {reason}")
                    print(f"  Stopping at generation {gen + 1}/{self.generations}.")
                break

            # Skip evolution on last generation
            if gen == self.generations - 1:
                break

            # Elitism
            elite = self.get_elite()

            # Create new population
            new_population = elite.copy()

            # Fill rest with offspring
            while len(new_population) < self.population_size:
                # Selection
                parent1 = self.tournament_selection()
                parent2 = self.tournament_selection()

                # Crossover
                child1, child2 = self.uniform_crossover(parent1, parent2)

                # Mutation
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)

                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)

            self.population = new_population

        total_time = time.time() - start_time

        if verbose:
            print("\n" + "=" * 60)
            print("OPTIMIZATION COMPLETE")
            print("=" * 60)
            print(f"Best solution: σ = {self.best_individual.sigma_vector}")
            print(f"Best accuracy: {self.best_individual.accuracy:.4f}")
            print(f"Best epsilon: {self.best_individual.epsilon:.4f}")
            print(f"Best fitness: {self.best_individual.fitness:.4f}")
            print(f"Total time: {total_time / 60:.2f} minutes")
            print("=" * 60)

        return self.best_individual

    def get_convergence_data(self) -> dict:
        """Get convergence data for plotting."""
        return {
            'generations':  [h['generation']  for h in self.history],
            'best_fitness': [h['best_fitness'] for h in self.history],
            'avg_fitness':  [h['avg_fitness']  for h in self.history],
            'std_fitness':  [h['std_fitness']  for h in self.history],
            'var_fitness':  [h['var_fitness']  for h in self.history],
            'best_accuracy':[h['best_accuracy']for h in self.history],
            'avg_accuracy': [h['avg_accuracy'] for h in self.history],
            'std_accuracy': [h['std_accuracy'] for h in self.history],
            'var_accuracy': [h['var_accuracy'] for h in self.history],
            'best_epsilon': [h['best_epsilon'] for h in self.history],
            'avg_epsilon':  [h['avg_epsilon']  for h in self.history],
            'std_epsilon':  [h['std_epsilon']  for h in self.history],
            'var_epsilon':  [h['var_epsilon']  for h in self.history],
        }
