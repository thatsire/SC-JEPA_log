# ============================================================================
# Quantizer Module
# ============================================================================
# Soft codebook quantizer using cosine similarity
#
# Input : (B, num_patches, latent_dim)
# Output:
#   - probs: (B, num_patches, num_codes)   soft assignment
#   - z_q  : (B, num_patches, latent_dim)  quantized latent (convex combination
#            of the L2-normalised codebook vectors)
#
# NOTE on the temperature: the cosine distance lives in [0, 2], so the softmax
# logits span at most 2 units. With temperature=1.0 the assignment over K=64
# codes is almost uniform (max prob <= 0.105 in theory, ~0.02 in practice), the
# quantized latent is the same vector for every input and every JEPA target
# becomes trivial. A temperature around 0.05-0.1 is required for the codes to
# carry information.
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class Quantizer(nn.Module):
    def __init__(self, num_codes: int, embedding_dim: int, temperature: float = 0.1):
        super().__init__()
        self.num_codes = num_codes
        self.embedding_dim = embedding_dim
        self.temperature = temperature

        self.embedding = nn.Embedding(num_codes, embedding_dim)
        nn.init.normal_(self.embedding.weight)  # random directions on the sphere after L2-norm

    def forward(self, z: torch.Tensor):
        z_norm = F.normalize(z, dim=-1)                          # (B, N, D)
        codebook = F.normalize(self.embedding.weight, dim=-1)    # (K, D)
        sim = torch.einsum("bnd,kd->bnk", z_norm, codebook)      # cosine similarity (B, N, K)
        probs = F.softmax(sim / self.temperature, dim=-1)        # (B, N, K)
        z_q = torch.einsum("bnk,kd->bnd", probs, codebook)       # (B, N, D)
        return probs, z_q
