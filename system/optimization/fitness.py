# Fitness evaluation for sigma optimization
# f(σ) = Acc - λ·ε  or  f(σ) = Acc / (ε + 1e-6)

import copy
import torch
import numpy as np
from typing import Tuple, List, Dict, Optional, Callable
import hashlib
import json


def _mp_evaluate_worker(task):
    """
    Worker function for multiprocessing-based parallel GPU evaluation.

    CUDA context is pinned at process start via CUDA_VISIBLE_DEVICES so that
    each worker is hard-locked to its assigned GPU regardless of Pool scheduling.
    """
    import sys, os, copy as _copy
    import torch as _torch

    (idx, sigma_vector, device_id, args, server_class, client_class,
     fitness_type, lambda_weight, client_groups, verbose) = task

    # Pin this process to a single GPU via env var — must happen before any
    # CUDA call so the CUDA context is created on the right device.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)

    _project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _project_dir not in sys.path:
        sys.path.insert(0, _project_dir)

    # With CUDA_VISIBLE_DEVICES set to a single GPU, that GPU becomes cuda:0
    # inside this process.
    args = _copy.deepcopy(args)
    args.device    = "cuda:0"
    args.device_id = "0"
    args.model     = args.model.to(args.device)

    schedule_type = getattr(args, 'sigma_schedule_type', 'fixed')
    evaluator = FitnessEvaluator(
        args, server_class, client_class,
        fitness_type, lambda_weight, use_cache=False,
        schedule_type=schedule_type,
    )
    result = evaluator.evaluate(sigma_vector, client_groups, verbose)
    return idx, result


class FitnessEvaluator:
    """
    Evaluates fitness of a sigma configuration by running FL training.

    Fitness functions:
    - 'linear': f(σ) = Acc - λ·ε
    - 'ratio': f(σ) = Acc / (ε + 1e-6)

    When n_evals > 1, each individual is evaluated multiple times with
    different random seeds and the mean is used as fitness, reducing the
    impact of stochastic noise in FL training.
    """

    def __init__(
        self,
        args,
        server_class,
        client_class,
        fitness_type: str = 'linear',
        lambda_weight: float = 0.1,
        use_cache: bool = True,
        device_ids: List[int] = None,
        n_evals: int = 1,
        base_seed: int = 42,
        schedule_type: str = 'fixed',
    ):
        """
        Args:
            args: Training arguments
            server_class: Server class (e.g., FedAvg)
            client_class: Client class (e.g., clientAVG)
            fitness_type: 'linear' or 'ratio'
            lambda_weight: Weight for epsilon in linear fitness
            use_cache: Whether to cache fitness evaluations
            device_ids: List of GPU device IDs for parallel evaluation
            n_evals: Number of independent FL runs per individual (>=1).
                     Results are averaged to reduce stochastic noise.
            base_seed: Base random seed; each eval i uses base_seed+i.
            schedule_type: 'fixed' (one sigma per client, static) or
                           'linear' (chromosome encodes sigma_start + sigma_end
                           per client; sigma decays linearly over rounds).
        """
        self.args = args
        self.server_class = server_class
        self.client_class = client_class
        self.fitness_type = fitness_type
        self.lambda_weight = lambda_weight
        self.use_cache = use_cache
        self.cache: Dict[str, Tuple[float, float, float]] = {}
        self.device_ids = device_ids or [0]
        self.n_evals = max(1, n_evals)
        self.base_seed = base_seed
        self.schedule_type = schedule_type
        self.evaluation_count = 0

    def _sigma_to_key(self, sigma_vector: List[float]) -> str:
        """Convert sigma vector to cache key."""
        # Round to 4 decimal places for cache stability
        rounded = [round(s, 4) for s in sigma_vector]
        return hashlib.md5(json.dumps(rounded).encode()).hexdigest()

    def evaluate(
        self,
        sigma_vector: List[float],
        client_groups: 'ClientGrouper',
        verbose: bool = False
    ) -> Tuple[float, float, float]:
        """
        Evaluate fitness for a sigma configuration.

        When self.n_evals > 1, runs FL training n_evals times with different
        seeds and returns mean (fitness, accuracy, epsilon), reducing noise.

        Args:
            sigma_vector: [sigma_low, sigma_mid, sigma_high]
            client_groups: ClientGrouper instance
            verbose: Print progress

        Returns:
            (fitness, accuracy, epsilon)  — averaged over n_evals runs
        """
        # Check cache
        cache_key = self._sigma_to_key(sigma_vector)
        if self.use_cache and cache_key in self.cache:
            if verbose:
                print(f"  [Cache hit] σ={sigma_vector}")
            return self.cache[cache_key]

        self.evaluation_count += 1
        if verbose:
            print(f"  [Eval #{self.evaluation_count}  n_evals={self.n_evals}] σ={sigma_vector}")

        args_copy = copy.deepcopy(self.args)
        args_copy.privacy = True
        args_copy.goal = "optimization"
        N = args_copy.num_clients

        if self.schedule_type == 'linear':
            # Chromosome layout: [sigma_start, sigma_end, C]
            # C (max_grad_norm) is optional — falls back to args default if not present.
            sigma_start_val = float(sigma_vector[0])
            sigma_end_val   = float(sigma_vector[1])
            if len(sigma_vector) >= 3:
                args_copy.max_grad_norm = float(sigma_vector[2])
            # sigma_schedule: client_id -> (start, end) for server to apply per round
            args_copy.sigma_schedule  = {i: (sigma_start_val, sigma_end_val) for i in range(N)}
            args_copy.sigma_per_client = [sigma_start_val] * N  # used for logging/display
            sigma_per_client = [sigma_start_val] * N
            if verbose:
                c_val = args_copy.max_grad_norm
                print(f"    σ_start={sigma_start_val:.2f}  σ_end={sigma_end_val:.2f}  C={c_val:.2f}  (global schedule, {N} clients)")
        else:
            # Fixed: one sigma per client, unchanged across rounds
            args_copy.sigma_schedule = None
            if len(sigma_vector) == N:
                sigma_per_client = list(sigma_vector)
            else:
                sigma_per_client = client_groups.apply_sigma_vector(sigma_vector, N)
            args_copy.sigma_per_client = sigma_per_client

        # Run n_evals independent FL trainings with distinct seeds
        accs, epsilons = [], []
        for i in range(self.n_evals):
            seed_i = self.base_seed + i
            np.random.seed(seed_i)
            import torch as _torch
            _torch.manual_seed(seed_i)

            run_args = copy.deepcopy(args_copy)
            acc_i, eps_i = self._run_training(run_args, sigma_per_client, verbose)
            accs.append(acc_i)
            epsilons.append(eps_i)

            if verbose and self.n_evals > 1:
                print(f"    run {i+1}/{self.n_evals}: Acc={acc_i:.4f}  ε={eps_i:.4f}")

        acc     = float(np.mean(accs))
        epsilon = float(np.mean(epsilons))
        acc_std = float(np.std(accs))
        eps_std = float(np.std(epsilons))

        # Calculate fitness from means
        if self.fitness_type == 'linear':
            fitness = acc - self.lambda_weight * epsilon
        elif self.fitness_type == 'ratio':
            fitness = acc / (epsilon + 1e-6)
        else:
            raise ValueError(f"Unknown fitness type: {self.fitness_type}")

        result = (fitness, acc, epsilon)

        # Cache result
        if self.use_cache:
            self.cache[cache_key] = result

        if verbose:
            if self.n_evals > 1:
                print(f"    -> Acc={acc:.4f}±{acc_std:.4f}  ε={epsilon:.4f}±{eps_std:.4f}  Fitness={fitness:.4f}")
            else:
                print(f"    -> Acc={acc:.4f}  ε={epsilon:.4f}  Fitness={fitness:.4f}")

        return result

    def _run_training(
        self,
        args,
        sigma_per_client: List[float],
        verbose: bool
    ) -> Tuple[float, float]:
        """
        Run FL training with given sigma configuration.

        Returns:
            (best_accuracy, final_epsilon)
        """
        # Reinitialize model
        model_copy = copy.deepcopy(self.args.model)
        model_copy = model_copy.to(args.device)
        args.model = model_copy

        # Create server
        server = self.server_class(args, times=0)

        # Apply per-client sigma
        for client in server.clients:
            client.dp_sigma = sigma_per_client[client.id]

        # Suppress output during training
        import sys
        import io
        if not verbose:
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()

        try:
            # Run training
            server.train()

            # Get results
            best_acc = max(server.rs_test_acc) if server.rs_test_acc else 0.0
            final_epsilon = server.rs_epsilon[-1] if server.rs_epsilon else 0.0

        finally:
            if not verbose:
                sys.stdout = old_stdout

        # Cleanup
        del server
        torch.cuda.empty_cache()

        return best_acc, final_epsilon

    def evaluate_parallel(
        self,
        sigma_vectors: List[List[float]],
        client_groups: 'ClientGrouper',
        verbose: bool = False,
        on_result: Optional[Callable] = None,
    ) -> List[Tuple[float, float, float]]:
        """
        Evaluate multiple sigma configurations in parallel using multiple GPUs.

        Uses multiprocessing with the 'spawn' start method so that each worker
        gets an independent CUDA context.  ThreadPoolExecutor was replaced
        because threads share the same CUDA context and deadlock when two
        threads perform simultaneous CUDA operations.

        Args:
            sigma_vectors: List of sigma vectors to evaluate
            client_groups: ClientGrouper instance
            verbose: Print progress
            on_result: Optional callback called as soon as each individual
                       finishes: on_result(sigma_vector_idx, (fitness, acc, eps)).
                       Also called for cache hits so callers get full coverage.

        Returns:
            List of (fitness, accuracy, epsilon) tuples
        """
        import multiprocessing as mp

        results = [None] * len(sigma_vectors)

        # Check cache first and filter out cached results
        to_evaluate = []
        for idx, sigma_vector in enumerate(sigma_vectors):
            cache_key = self._sigma_to_key(sigma_vector)
            if self.use_cache and cache_key in self.cache:
                results[idx] = self.cache[cache_key]
                if on_result:
                    on_result(idx, results[idx])
            else:
                to_evaluate.append((idx, sigma_vector))

        if not to_evaluate:
            return results

        # Move model to CPU so it can be pickled by the spawn'd processes.
        args_cpu = copy.deepcopy(self.args)
        args_cpu.model = args_cpu.model.cpu()

        # Assign GPUs round-robin across tasks.
        tasks = [
            (idx, sigma_vector,
             self.device_ids[i % len(self.device_ids)],
             args_cpu, self.server_class, self.client_class,
             self.fitness_type, self.lambda_weight,
             client_groups, verbose)
            for i, (idx, sigma_vector) in enumerate(to_evaluate)
        ]

        ctx = mp.get_context('spawn')
        # imap_unordered yields each result as soon as the worker finishes,
        # enabling per-individual CSV checkpointing without waiting for the
        # full generation to complete.
        with ctx.Pool(processes=len(self.device_ids)) as pool:
            for idx, result in pool.imap_unordered(_mp_evaluate_worker, tasks):
                results[idx] = result
                if self.use_cache:
                    self.cache[self._sigma_to_key(sigma_vectors[idx])] = result
                if on_result:
                    on_result(idx, result)

        return results

    def clear_cache(self):
        """Clear the fitness cache."""
        self.cache.clear()

    def get_stats(self) -> Dict:
        """Get evaluation statistics."""
        return {
            'total_evaluations': self.evaluation_count,
            'cache_size': len(self.cache),
            'cache_hit_rate': (
                len(self.cache) / self.evaluation_count
                if self.evaluation_count > 0 else 0
            )
        }
