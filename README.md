# DPFL: Differentially Private Federated Learning

Implementation of Differentially Private Federated Learning based on PFLlib with Opacus integration.

## Overview

A framework for training federated learning models with differential privacy guarantees using DP-SGD. Supports MNIST and CIFAR-10 datasets with non-IID data distribution (Dirichlet alpha=0.2).

## Experimental Results

### MNIST (50 rounds, 20 clients)

| Algorithm | sigma | Accuracy | epsilon |
|-----------|------:|--------:|--------:|
| FedAvg    |  0    | 89.12%  |   0.00  |
| FedAvg    |  1    | 55.40%  |   6.28  |
| FedAvg    |  4    | 53.39%  |   0.95  |
| FedAvg    | 12    | 48.50%  |   0.28  |
| FedAvg    | 20    | 48.17%  |   0.18  |
| SCAFFOLD  |  0    | 93.59%  |   0.00  |
| SCAFFOLD  |  1    | 56.03%  |   6.28  |
| SCAFFOLD  |  4    | 51.98%  |   0.95  |
| SCAFFOLD  | 12    | 46.29%  |   0.28  |
| SCAFFOLD  | 20    | 45.57%  |   0.18  |
| FedALA    |  0    | 98.17%  |   0.00  |
| FedALA    |  1    | 75.01%  |   6.28  |
| FedALA    |  4    | 74.76%  |   0.95  |
| FedALA    | 12    | 75.07%  |   0.28  |
| FedALA    | 20    | 76.26%  |   0.18  |

### CIFAR-10 (50 rounds, 20 clients)

| Algorithm | sigma | Accuracy | epsilon |
|-----------|------:|--------:|--------:|
| FedAvg    |  0    | 50.16%  |   0.00  |
| FedAvg    |  1    | 23.86%  |   3.87  |
| FedAvg    |  4    | 20.41%  |   0.60  |
| FedAvg    | 12    | 12.01%  |   0.19  |
| FedAvg    | 20    | 10.72%  |   0.14  |
| SCAFFOLD  |  0    | 47.83%  |   0.00  |
| SCAFFOLD  |  1    | 24.35%  |   3.87  |
| SCAFFOLD  |  4    | 22.60%  |   0.60  |
| SCAFFOLD  | 12    | 14.47%  |   0.19  |
| SCAFFOLD  | 20    | 13.31%  |   0.14  |
| FedALA    |  0    | 79.59%  |   0.00  |
| FedALA    |  1    | 61.68%  |   3.87  |
| FedALA    |  4    | 60.50%  |   0.60  |
| FedALA    | 12    | 38.84%  |   0.19  |
| FedALA    | 20    | 31.99%  |   0.14  |

### Key Findings

- **FedALA** consistently outperforms FedAvg and SCAFFOLD under DP noise
- Higher sigma values provide stronger privacy (lower epsilon) but degrade accuracy
- CIFAR-10 is more sensitive to DP noise than MNIST due to model complexity
- FedALA maintains ~75% accuracy on MNIST even with sigma=20

## Installation

### Requirements

- Python 3.10+
- CUDA 12.1+
- PyTorch 2.0+

### Setup

```bash
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

# For MNIST (non-IID, Dirichlet alpha=0.2, 20 clients)
python generate_mnist.py noniid - dir 0.2

# For CIFAR-10
python generate_cifar10.py noniid - dir 0.2
```

### 2. Run Training

```bash
cd system

# Run experiments with multiple sigma values
python run_fixed_sigma_mnist.py \
    --algorithms FedAvg SCAFFOLD FedALA \
    --sigmas 0 1 4 12 20 \
    --rounds 50 \
    --clients 20
```

## Project Structure

```
dpfl/
├── dataset/
│   ├── generate_mnist.py      # MNIST data generation
│   ├── generate_cifar10.py    # CIFAR-10 data generation
│   └── utils/                 # Dataset utilities
├── system/
│   ├── flcore/
│   │   ├── clients/           # FL client implementations
│   │   ├── servers/           # FL server implementations
│   │   ├── trainmodel/        # Model architectures
│   │   └── optimizers/        # FL optimizers
│   ├── utils/                 # Logging and helper functions
│   └── run_fixed_sigma_mnist.py  # Training script
├── requirements.txt
└── README.md
```

## Supported FL Algorithms

- **FedAvg**: Federated Averaging with DP-SGD
- **FedALA**: Adaptive Local Aggregation with DP-SGD
- **SCAFFOLD**: Stochastic Controlled Averaging with DP-SGD

## Differential Privacy

We use [Opacus](https://opacus.ai/) for DP-SGD implementation with:
- Renyi Differential Privacy (RDP) accounting
- Per-sample gradient clipping (C=1.0)
- Gaussian noise injection
- Privacy amplification by subsampling

## License

This project is licensed under the GPL v2 License.
