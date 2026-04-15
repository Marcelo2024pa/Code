# Client grouping for sigma allocation optimization
# Groups clients into low/mid/high based on dataset size

import numpy as np
from typing import List, Dict, Tuple


class ClientGrouper:
    """
    Groups clients into categories based on their dataset size.

    Groups:
    - low: clients with small datasets (bottom 33%)
    - mid: clients with medium datasets (middle 33%)
    - high: clients with large datasets (top 33%)
    """

    def __init__(self, num_groups: int = 3):
        self.num_groups = num_groups
        self.group_names = ['low', 'mid', 'high'][:num_groups]
        self.client_groups: Dict[int, str] = {}
        self.group_indices: Dict[str, List[int]] = {name: [] for name in self.group_names}
        self.thresholds: List[float] = []

    def fit(self, client_train_samples: List[int]) -> 'ClientGrouper':
        """
        Fit the grouper based on client dataset sizes.

        Args:
            client_train_samples: List of training samples per client

        Returns:
            self
        """
        samples = np.array(client_train_samples)
        num_clients = len(samples)

        # Calculate percentile thresholds
        percentiles = np.linspace(0, 100, self.num_groups + 1)[1:-1]
        self.thresholds = [np.percentile(samples, p) for p in percentiles]

        # Assign clients to groups
        self.client_groups = {}
        self.group_indices = {name: [] for name in self.group_names}

        for client_id, sample_count in enumerate(samples):
            group_idx = 0
            for threshold in self.thresholds:
                if sample_count > threshold:
                    group_idx += 1

            group_name = self.group_names[group_idx]
            self.client_groups[client_id] = group_name
            self.group_indices[group_name].append(client_id)

        return self

    def get_client_group(self, client_id: int) -> str:
        """Get the group name for a specific client."""
        return self.client_groups.get(client_id, 'mid')

    def get_group_clients(self, group_name: str) -> List[int]:
        """Get all client IDs belonging to a group."""
        return self.group_indices.get(group_name, [])

    def apply_sigma_vector(self, sigma_vector: List[float], num_clients: int) -> List[float]:
        """
        Apply a reduced sigma vector to all clients based on their groups.

        Args:
            sigma_vector: [sigma_low, sigma_mid, sigma_high] or similar
            num_clients: Total number of clients

        Returns:
            List of sigma values, one per client
        """
        if len(sigma_vector) != self.num_groups:
            raise ValueError(f"Expected {self.num_groups} sigma values, got {len(sigma_vector)}")

        sigma_per_client = []
        for client_id in range(num_clients):
            group_name = self.get_client_group(client_id)
            group_idx = self.group_names.index(group_name)
            sigma_per_client.append(sigma_vector[group_idx])

        return sigma_per_client

    def fit_by_entropy(self, client_label_lists: List[List[int]], num_classes: int) -> 'ClientGrouper':
        """
        Fit grouper based on class distribution entropy (non-IID heterogeneity).

        Theory: clients with LOW entropy have concentrated class distributions
        (highly specialized gradients) → should receive LESS noise (smaller σ).
        Clients with HIGH entropy are nearly uniform → can tolerate MORE noise.

        Groups map as:
            'low'  → low entropy  → smallest σ  (most informative gradients)
            'mid'  → mid entropy  → medium σ
            'high' → high entropy → largest σ   (least informative gradients)

        Args:
            client_label_lists: list of label arrays, one per client
            num_classes: total number of classes in the dataset
        """
        entropies = []
        for labels in client_label_lists:
            counts = np.bincount(np.array(labels, dtype=int), minlength=num_classes).astype(float)
            total = counts.sum()
            if total == 0:
                entropies.append(0.0)
                continue
            probs = counts / total
            probs = probs[probs > 0]
            h = float(-np.sum(probs * np.log(probs)))
            entropies.append(h)

        self.entropies = entropies
        return self.fit(entropies)

    def summary(self) -> str:
        """Return a summary of the grouping."""
        lines = ["Client Grouping Summary:"]
        lines.append(f"  Thresholds: {self.thresholds}")
        for name in self.group_names:
            clients = self.group_indices[name]
            lines.append(f"  {name}: {len(clients)} clients -> {clients}")
        return "\n".join(lines)
