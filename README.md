# GA-DPFL: Genetic Algorithm for Differentially Private Federated Learning

This repository contains the implementation of **Adaptive Noise Allocation for Differentially Private Federated Learning via Genetic Optimization**, accepted at LANC 2026.

## Overview

We propose a Genetic Algorithm (GA)-based framework for optimizing time-varying noise schedules in differentially private federated learning. The approach encodes noise schedule parameters as a chromosome with three genes:

- **σ_start**: Initial noise multiplier
- **σ_end**: Final noise multiplier
- **C**: Gradient clipping norm

The noise schedule follows a linear interpolation:

```
σ(t) = σ_start + (σ_end - σ_start) · t/T
```

A fitness function balances model accuracy against privacy cost:

```
F(z) = Acc_max(z) - λ · ε_final(z)
```

## Key Results

| Algorithm | Configuration | Accuracy | ε | Fitness | Gain |
|-----------|--------------|----------|---|---------|------|
| FedAvg | Fixed σ=4 | 48.10% | 0.588 | 0.434 | -- |
| FedAvg | GA [5.09→8.19, 1.81] | 49.48% | 0.346 | 0.460 | **+6.1%** |
| FedALA | Fixed σ=20 | 72.77% | 0.134 | 0.714 | -- |
| SCAFFOLD | Fixed σ=20 | 26.57% | 0.132 | 0.252 | -- |

## Installation

### Requirements

- Python 3.10+
- CUDA 12.1+
- PyTorch 2.0+

### Setup

```bash
# Clone the repository
git clone https://github.com/ufpa-laser/ga-dpfl.git
cd ga-dpfl

# Create conda environment (recommended)
conda env create -f env_cuda_latest.yaml
conda activate fl

# Or install via pip
pip install -r requirements.txt
```

## Usage

### 1. Generate Dataset

```bash
cd dataset

# For MNIST (non-IID, Dirichlet α=0.2, 20 clients)
python generate_mnist.py noniid - dir 0.2

# For CIFAR-10
python generate_cifar10.py noniid - dir 0.2
```

### 2. Run GA Optimization

```bash
cd system

# FedAvg with GA optimization
python run_ag_mnist_fedavg.py \
    --rounds 20 \
    --clients 20 \
    --population 10 \
    --generations 20 \
    --device_id 0
```

### 3. Run Fixed Baseline Experiments

```bash
cd system

python run_fixed_sigma_mnist.py \
    --algorithm FedAvg \
    --sigma 4.0 \
    --rounds 20 \
    --clients 20
```

### 4. Full Comparison (Fixed + GA)

```bash
cd system

python run_full_comparison.py \
    --rounds 20 \
    --clients 20 \
    --population 10 \
    --generations 20 \
    --device_id 0
```

## GA Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--population` | 10 | Population size |
| `--generations` | 20 | Number of generations |
| `--mutation_rate` | 0.3 | Mutation probability per gene |
| `--crossover_rate` | 0.8 | Crossover probability |
| `--elitism` | 2 | Number of elite individuals preserved |
| `--tournament_size` | 3 | Tournament selection size |
| `--patience` | 6 | Early stopping patience |
| `--min_diversity` | 0.01 | Minimum population diversity |

## Project Structure

```
ga-dpfl/
├── dataset/
│   ├── generate_mnist.py      # MNIST data generation
│   ├── generate_cifar10.py    # CIFAR-10 data generation
│   └── utils/                 # Dataset utilities
├── system/
│   ├── flcore/
│   │   ├── clients/           # FL client implementations
│   │   ├── servers/           # FL server implementations (FedAvg, FedALA, SCAFFOLD)
│   │   └── trainmodel/        # Model training logic
│   ├── models/                # Neural network architectures
│   ├── optimization/
│   │   ├── genetic_algorithm.py  # GA implementation
│   │   ├── fitness.py            # Fitness evaluation
│   │   └── client_groups.py      # Client grouping utilities
│   ├── utils/                 # Logging and helper functions
│   ├── run_ag_mnist_fedavg.py    # GA optimization script
│   ├── run_fixed_sigma_mnist.py  # Fixed baseline script
│   └── run_full_comparison.py    # Full comparison script
├── requirements.txt
├── env_cuda_latest.yaml
└── README.md
```

## Supported FL Algorithms

- **FedAvg**: Federated Averaging with DP-SGD
- **FedALA**: Adaptive Local Aggregation with DP-SGD
- **SCAFFOLD**: Stochastic Controlled Averaging with DP-SGD

## Differential Privacy

We use [Opacus](https://opacus.ai/) for DP-SGD implementation with:
- Rényi Differential Privacy (RDP) accounting
- Per-sample gradient clipping
- Gaussian noise injection
- Privacy amplification by subsampling

## Citation

If you use this code, please cite our paper:

```bibtex
@inproceedings{silva2026adaptive,
  title={Adaptive Noise Allocation for Differentially Private Federated Learning via Genetic Optimization},
  author={Silva, Marcelo and Martins, Hugo and Bastos, Lucas and Veiga, Rafael and Rosario, Denis and Costa, Allan and Cerqueira, Eduardo},
  booktitle={Proceedings of the 2026 Latin America Networking Conference (LANC)},
  year={2026},
  organization={ACM}
}
```

## Acknowledgments

This work extends [PFLlib](https://github.com/TsingZ0/PFLlib) with differential privacy support via Opacus.

## License

This project is licensed under the GPL v2 License - see the [LICENSE](LICENSE) file for details.
