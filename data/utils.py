import torch
import torch.nn as nn
import random
import numpy as np


def set_seed(seed: int):   
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def instance_normalize(batch: torch.Tensor):
    """
    Instance-wise normalization.
    It forces the model to focus on the waveform (peaks, dips) while ignoring
    the absolute magnitude.
    """
    mean = batch.mean(dim=(1, 2), keepdim=True)
    std = batch.std(dim=(1, 2), keepdim=True) + 1e-6
    normalized = (batch - mean) / std
    return normalized, mean, std


def reverse_instance_normalize(normed: torch.Tensor,
                               mean: torch.Tensor,
                               std: torch.Tensor):
    mean = mean.permute(0, 1, 3, 2)
    std = std.permute(0, 1, 3, 2)
    return normed * std + mean


@torch.no_grad()
def update_ema(model_online: nn.Module,
               model_ema: nn.Module,
               decay: float):
    """
    Exponential moving average (EMA) update.
    The teacher (model_ema) is slowly updated following the student (model_online)
    to avoid the collapse of the network during pre-training.
    """
    for p_online, p_ema in zip(model_online.parameters(),
                               model_ema.parameters()):
        p_ema.data.mul_(decay).add_(p_online.data, alpha=1 - decay)


def coarse_scale_to_patch(x_input: torch.Tensor,
                          add_patch_dim: bool = False):
    """
    It takes a group of fine-grained patches and averages them to create a single 
    coarse-grained patch (coarse patch).
    It forces the model to understand data at different time scales.
    """
    B, P, L, C = x_input.shape

    x_flat = x_input.contiguous().view(B, P * L, C)
    
    x_grouped = x_flat.view(B, L, P, C)
    new_patch = x_grouped.mean(dim=2)

    if add_patch_dim:
        new_patch = new_patch.unsqueeze(1)

    return new_patch