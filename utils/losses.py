import torch
import torch.nn.functional as F


def kl_loss_fine(logits_pred, p_future, temperature):
    """Patch-level KL divergence between predicted and target code distributions."""
    logp_pred = F.log_softmax(logits_pred / temperature, dim=-1)
    return F.kl_div(logp_pred, p_future, reduction="batchmean")


def kl_loss_coarse(logits_pred_coarse, p_future_coarse, temperature):
    """Coarse-scale KL divergence (single patch)."""
    logp_pred = F.log_softmax(logits_pred_coarse / temperature, dim=-1)
    return F.kl_div(logp_pred, p_future_coarse, reduction="batchmean")


def mse_alignment_loss(z_pred, z_q_future):
    """Latent-space alignment between predicted and target quantized latents."""
    return F.mse_loss(z_pred, z_q_future.detach())


def vq_losses(h_past, z_q_past):
    """
    Codebook and commitment losses. z_q is a combination of L2-normalised codebook
    vectors, so the encoder output is compared in the same normalised space
    (otherwise the commitment term only shrinks the norm of h).
    """
    h_norm = F.normalize(h_past, dim=-1)
    loss_q = F.mse_loss(z_q_past, h_norm.detach())
    loss_commit = F.mse_loss(h_norm, z_q_past.detach())
    return loss_q, loss_commit


def entropy_losses(p_past, eps=1e-6):
    """
    - sample-wise entropy (minimised: each patch should commit to few codes)
    - batch-wise entropy (maximised via the negative sign: all codes should be used)
    """
    probs_safe = torch.clamp(p_past, min=eps, max=1 - eps)
    entropy_sample = -(probs_safe * torch.log(probs_safe)).sum(dim=-1).mean()
    avg_probs = probs_safe.mean(dim=(0, 1))
    batch_entropy = -(avg_probs * torch.log(avg_probs)).sum()
    return entropy_sample, -batch_entropy


@torch.no_grad()
def codebook_perplexity(probs):
    """exp(entropy of the average assignment): K means all codes used equally, 1 means collapse."""
    avg = probs.reshape(-1, probs.shape[-1]).mean(0)
    return torch.exp(-(avg * torch.log(avg.clamp_min(1e-9))).sum())
