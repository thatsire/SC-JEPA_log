import random

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def update_ema(model_online: nn.Module, model_ema: nn.Module, decay: float):
    """
    Exponential moving average update of the target (teacher) network.
    """
    for p_online, p_ema in zip(model_online.parameters(), model_ema.parameters()):
        p_ema.data.mul_(decay).add_(p_online.data, alpha=1 - decay)


def coarse_scale_to_patch(x_input: torch.Tensor, add_patch_dim: bool = False):
    """
    Temporal downsampling of a group of P fine patches into ONE coarse patch.

    (B, P, L, C) -> (B, L, C): the P*L time steps are split into L consecutive groups of
    P steps and each group is averaged, so the coarse patch spans the whole P*L hours at a
    resolution of P hours per step. With P=4, L=6: 24 hours -> 6 steps of 4 hours.
    """
    B, P, L, C = x_input.shape
    x_flat = x_input.contiguous().view(B, P * L, C)
    new_patch = x_flat.view(B, L, P, C).mean(dim=2)
    if add_patch_dim:
        new_patch = new_patch.unsqueeze(1)  # (B, 1, L, C)
    return new_patch
