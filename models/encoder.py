import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
# Linear Feature Extractor
# ============================================================================
# Lightweight linear extractor that replaces the old CNN/Residual approach.

class CnnFeatureExtractor(nn.Module):
    """Lightweight linear feature extractor"""
    def __init__(self, in_channels, h_dim, latent_dim, patch_len=20):
        super().__init__()

        # Simple linear projection from patch_len * in_channels to h_dim
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels * patch_len, h_dim),  
            nn.ReLU(inplace=False),
            nn.Dropout(0.1)
        )

        # Lightweight feature processing
        self.feature_net = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(inplace=False),
            nn.Dropout(0.1),
            nn.Linear(h_dim, latent_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape = (B, in_channels, patch_len)
        B = x.shape[0]
        # Flatten: (B, in_channels, patch_len) -> (B, in_channels * patch_len)
        x = x.reshape(B, -1)
        x = self.input_proj(x)
        return self.feature_net(x)


# ============================================================================
# Encoder Architecture
# ============================================================================
# Combines feature extractor with Transformer for patch-level representations
# Key: Returns ALL patch representations, not just CLS token

class Encoder(nn.Module):
    """Encoder = Feature extractor + TransformerEncoder."""
    def __init__(self, num_patches, patch_len, latent_dim,
                 cnn_h_dim, trans_nhead, trans_num_layers, in_channels=1):
        super().__init__()
        
        # Inizializziamo l'estrattore pulito (senza parametri fantasma)
        self.cnn = CnnFeatureExtractor(
            in_channels, cnn_h_dim, latent_dim, patch_len
        )
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, latent_dim))
        self.pos_emb = nn.Parameter(torch.randn(1, num_patches + 1, latent_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=trans_nhead,
            dim_feedforward=latent_dim * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=trans_num_layers
        )

        self.final_proj = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        B, N, L, C = x.shape
        x = x.permute(0,1,3,2).contiguous().view(B*N, C, L)
        feat = self.cnn(x)                          # (B*N, latent_dim)
        feat = feat.view(B, N, -1)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, latent_dim)
        feat = torch.cat([cls_tokens, feat], dim=1)    # (B, N+1, latent_dim)
        feat = feat + self.pos_emb[:, :feat.shape[1], :]

        out = self.transformer(feat)
        all_patch_reprs = out[:, 1:, :]  # (B, N, latent_dim) - exclude CLS token
        return self.final_proj(all_patch_reprs)  # (B, N, latent_dim)