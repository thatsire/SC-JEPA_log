import torch
import torch.nn as nn
import random
import numpy as np

# ============================================================================
# Utilities per la Riproducibilità
# ============================================================================
def set_seed(seed: int):
    """
    Setta il seed casuale per garantire la riproducibilità degli esperimenti.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ============================================================================
# Utilities per il Pre-Training (SC-JEPA)
# ============================================================================
def instance_normalize(batch: torch.Tensor):
    """
    Instance-wise normalization.
    Forza il modello a concentrarsi sulla forma dell'onda (picchi, cali)
    ignorando la magnitudine assoluta.
    """
    mean = batch.mean(dim=(1, 2), keepdim=True)
    std = batch.std(dim=(1, 2), keepdim=True) + 1e-6
    normalized = (batch - mean) / std
    return normalized, mean, std


def reverse_instance_normalize(normed: torch.Tensor,
                               mean: torch.Tensor,
                               std: torch.Tensor):
    """
    Riporta i valori normalizzati alla loro scala reale.
    """
    mean = mean.permute(0, 1, 3, 2)
    std = std.permute(0, 1, 3, 2)
    return normed * std + mean


@torch.no_grad()
def update_ema(model_online: nn.Module,
               model_ema: nn.Module,
               decay: float):
    """
    Aggiornamento Exponential moving average (EMA).
    Il teacher (model_ema) si aggiorna lentamente seguendo lo student (model_online)
    per evitare il collasso della rete durante il pre-training.
    """
    for p_online, p_ema in zip(model_online.parameters(),
                               model_ema.parameters()):
        p_ema.data.mul_(decay).add_(p_online.data, alpha=1 - decay)


def coarse_scale_to_patch(x_input: torch.Tensor,
                          add_patch_dim: bool = False):
    """
    Prende un gruppo di patch a grana fine e ne fa la media per creare
    una singola patch a grana grossa (coarse patch).
    Forza il modello a comprendere i dati a scale temporali diverse.
    """
    B, P, L, C = x_input.shape

    # Appiattisce il tempo
    x_flat = x_input.contiguous().view(B, P * L, C)
    
    # Raggruppa e fa la media
    x_grouped = x_flat.view(B, L, P, C)
    new_patch = x_grouped.mean(dim=2)

    if add_patch_dim:
        new_patch = new_patch.unsqueeze(1)

    return new_patch