# PFLlib: Personalized Federated Learning Algorithm Library
# Copyright (C) 2021  Jianqing Zhang

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

import torch
from opacus import PrivacyEngine
from opacus.optimizers.optimizer import DPOptimizer

# Monkey-patch para corrigir bug de device mismatch no opacus 1.4.0
# Problema: batch vazio (Poisson sampling) cria per_sample_clip_factor em CPU
# enquanto grad_samples estão em CUDA
_original_clip_and_accumulate = DPOptimizer.clip_and_accumulate

def _fixed_clip_and_accumulate(self):
    if len(self.grad_samples[0]) == 0:
        # Batch vazio: inicializa summed_grad como zeros no device correto
        from opacus.optimizers.optimizer import _check_processed_flag, _mark_as_processed
        for p in self.params:
            _check_processed_flag(p.grad_sample)
            if p.summed_grad is not None:
                p.summed_grad += torch.zeros_like(p)
            else:
                p.summed_grad = torch.zeros_like(p)
            _mark_as_processed(p.grad_sample)
        return
    return _original_clip_and_accumulate(self)

DPOptimizer.clip_and_accumulate = _fixed_clip_and_accumulate

MAX_GRAD_NORM = 1.0
DELTA = 1e-5

def initialize_dp(model, optimizer, data_loader, dp_sigma, accountant=None):
    # Alterado o accountant para 'rdp' para evitar o erro de falta de memória (ArrayMemoryError)
    privacy_engine = PrivacyEngine(accountant="rdp")
    
    # Se já existir um histórico de privacidade de rodadas anteriores, injetamos ele
    if accountant is not None:
        privacy_engine.accountant = accountant
    
    model, optimizer, data_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=data_loader,
        noise_multiplier = dp_sigma, 
        max_grad_norm = MAX_GRAD_NORM,
    )

    return model, optimizer, data_loader, privacy_engine


def get_dp_params(privacy_engine):
    return privacy_engine.get_epsilon(delta=DELTA), DELTA

