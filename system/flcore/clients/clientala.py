"""
FedALA client — Federated Adaptive Local Aggregation.

Before each local training round, the client runs the ALA module:
it learns layer-wise mixing weights between the received global model
and its previous local model (or the global model itself on the first round).
This is a local operation — no data leaves the client — so no DP is needed
for the ALA step.

DP-SGD (via Opacus) is applied only during standard local training,
identical to FedAvg.
"""

import copy
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from flcore.clients.clientbase import Client
from utils.privacy import initialize_dp, get_dp_params
from utils.data_utils import read_client_data

# functional_call (PyTorch >= 2.0) enables differentiable forward pass
# with an explicit parameter dict — required for the ALA gradient step.
from torch.func import functional_call


class clientALA(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.current_epsilon = 0.0
        self.client_accountant = None

        # ALA state (persisted across rounds)
        self.ala_logits = None      # learned layer-wise mixing logits
        self.ala_eta    = 0.01      # ALA optimiser learning rate
        self.ala_steps  = 10        # gradient steps per round

        self._global_model_snapshot = None
        self._first_round = True    # flag: initialise from global on round 0

    # ------------------------------------------------------------------
    # Override set_parameters
    #
    # BUG original: super().set_parameters(model) sobrescrevia self.model
    # com o modelo global, então params_g == params_l → ALA nunca rodava.
    #
    # FIX: a partir da 2ª rodada, apenas salva o snapshot global.
    # self.model permanece como o modelo TREINADO da rodada anterior,
    # que é exatamente o "modelo local personalizado" que o ALA deve misturar
    # com o modelo global recebido do servidor.
    # ------------------------------------------------------------------
    def set_parameters(self, model):
        self._global_model_snapshot = copy.deepcopy(model)
        if self._first_round:
            # Rodada 0: inicializa o modelo local a partir do global
            super().set_parameters(model)
            self._first_round = False
        # Rodadas seguintes: self.model mantém o treinado da rodada anterior

    # ------------------------------------------------------------------
    # Main training method
    # ------------------------------------------------------------------
    def train(self):
        # 1) ALA step: adapt local model init (no DP needed, local only)
        if self._global_model_snapshot is not None:
            self._adaptive_aggregation(self._global_model_snapshot)

        # 2) Standard local training (with DP-SGD if privacy is enabled)
        trainloader = self.load_train_data()
        self.model.train()

        if self.privacy:
            model_origin = copy.deepcopy(self.model)
            self.model, self.optimizer, trainloader, privacy_engine = \
                initialize_dp(self.model, self.optimizer, trainloader,
                              self.dp_sigma, self.client_accountant)

        start_time = time.time()
        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max(2, max_local_epochs // 2))

        for _ in range(max_local_epochs):
            for x, y in trainloader:
                if isinstance(x, list):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                output = self.model(x)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

        if self.privacy:
            self.client_accountant = privacy_engine.accountant
            eps, DELTA = get_dp_params(privacy_engine)
            self.current_epsilon = eps
            print(f"Client {self.id}: epsilon = {eps:.2f}, delta = {DELTA}")

            for param, param_dp in zip(model_origin.parameters(), self.model.parameters()):
                param.data = param_dp.data.clone()
            self.model = model_origin
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

    # ------------------------------------------------------------------
    # ALA: Adaptive Local Aggregation
    # ------------------------------------------------------------------
    def _adaptive_aggregation(self, global_model):
        """
        Learn layer-wise mixing logits α_l so that
            x_l = sigmoid(α_l) * w_global_l + (1 - sigmoid(α_l)) * w_local_l
        minimises local loss.  Then update self.model in-place with x_l.
        """
        params_g = [p.data.clone().detach() for p in global_model.parameters()]
        params_l = [p.data.clone().detach() for p in self.model.parameters()]

        # Nothing to adapt if models are already identical (round 0 or no drift)
        if all(torch.equal(pg, pl) for pg, pl in zip(params_g, params_l)):
            return

        n_layers    = len(params_g)
        param_names = [name for name, _ in self.model.named_parameters()]

        # Initialise or reuse mixing logits
        if self.ala_logits is None or self.ala_logits.shape[0] != n_layers:
            self.ala_logits = torch.zeros(n_layers, device=self.device)

        logits    = self.ala_logits.clone().detach().requires_grad_(True)
        optimiser = torch.optim.Adam([logits], lr=self.ala_eta)

        # Load a small batch from local data for the ALA optimisation
        local_data = read_client_data(self.dataset, self.id, is_train=True)
        loader     = DataLoader(local_data, batch_size=self.batch_size,
                                shuffle=True, drop_last=False)
        data_iter  = iter(loader)

        for _ in range(self.ala_steps):
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                x, y = next(data_iter)

            x, y = x.to(self.device), y.to(self.device)

            # Build mixed parameter dict (differentiable w.r.t. logits)
            alphas     = torch.sigmoid(logits)
            mixed_dict = {
                name: alphas[i] * pg + (1.0 - alphas[i]) * pl
                for i, (name, pg, pl) in enumerate(
                    zip(param_names, params_g, params_l))
            }

            output = functional_call(self.model, mixed_dict, x)
            loss   = self.loss(output, y)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        # Apply the learned mixing to self.model (in-place, no grad needed)
        with torch.no_grad():
            alphas = torch.sigmoid(logits)
            for i, param in enumerate(self.model.parameters()):
                a = alphas[i].item()
                param.data = a * params_g[i] + (1.0 - a) * params_l[i]

        # Persist logits for the next round (warm start)
        self.ala_logits = logits.detach()
