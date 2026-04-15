# Optimization module for sigma allocation in Federated Learning with DP
# Based on roteiro_optimazation.md and arq_AG.md

from .genetic_algorithm import GeneticAlgorithm
from .fitness import FitnessEvaluator
from .client_groups import ClientGrouper

__all__ = ['GeneticAlgorithm', 'FitnessEvaluator', 'ClientGrouper']
