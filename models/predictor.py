import torch
import torch.nn as nn
import torch.nn.functional as F

LATENT_DIM = 256

class TransformerPredictor(nn.Module):
    """
    Transformer-based predictor that maps past patch-level code distributions
    to future patch-level code distribution logits.
    """
    def __init__(self, num_codes=128, nhead=4, num_layers=2,
                 hidden_dim=128, dropout=0.1, num_patches=5, latent_dim=256):
        super().__init__()
        self.num_codes = num_codes
        self.hidden_dim = hidden_dim

        # Project code distributions into continuous hidden space
        self.input_proj = nn.Sequential(
            nn.Linear(num_codes, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=False),
            nn.Dropout(dropout)
        )

        # Learnable positional embeddings for patch ordering
        self.pos_emb = nn.Parameter(torch.randn(1, num_patches, hidden_dim))

        # Transformer encoder for modeling temporal dependencies between patches
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Output head for predicting code distribution logits
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=False),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_codes)
        )

        # Output head for predicting latent representations
        self.latent_proj = nn.Linear(hidden_dim, latent_dim)

        self._init_weights()

    def _init_weights(self):
        """
        Initialize linear and normalization layers for stable training.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, p: torch.Tensor):
        B, N, _ = p.shape  # K is ignored

        # Project input distributions to hidden space
        x = self.input_proj(p)  # (B, N, H)

        # Add positional embeddings
        x = x + self.pos_emb[:, :N, :]  # (B, N, H)

        # Transformer encoding
        x = self.transformer(x)  # (B, N, H)

        # Predict future code distributions and latent representations
        logits = self.output_proj(x)       # (B, N, K)
        latent_pred = self.latent_proj(x)  # (B, N, D)

        return logits, latent_pred


class CoarsePredictor(nn.Module):
    """
    Coarse-scale predictor that maps multiple fine-scale patch code distributions
    to a single future coarse-scale patch distribution.
    """
    def __init__(self, num_codes=128, nhead=4, num_layers=2,
                 hidden_dim=128, dropout=0.1, num_patches=5, latent_dim=256):
        super().__init__()
        self.num_codes = num_codes
        self.hidden_dim = hidden_dim
        self.num_patches = num_patches

        # Project input code distributions
        self.input_proj = nn.Sequential(
            nn.Linear(num_codes, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=False),
            nn.Dropout(dropout)
        )

        # Positional embeddings for input patches
        self.pos_emb = nn.Parameter(torch.randn(1, num_patches, hidden_dim))

        # Learnable query token used to aggregate information
        self.query_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

        # Transformer encoder for processing fine-scale patches
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Cross-attention: query token attends to encoded patches
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.cross_norm = nn.LayerNorm(hidden_dim)

        # Output head for coarse-scale code distribution
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=False),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_codes)
        )

        # Output head for coarse-scale latent representation
        self.latent_proj = nn.Linear(hidden_dim, latent_dim)

        self._init_weights()

    def _init_weights(self):
        """
        Initialize linear and normalization layers.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, p: torch.Tensor):
        B, N, _ = p.shape  # K is ignored

        # Encode fine-scale patches
        x = self.input_proj(p)                 # (B, 5, H)
        x = x + self.pos_emb[:, :N, :]         # (B, 5, H)
        x = self.transformer(x)                # (B, 5, H)

        # Query token aggregates information from all patches
        query = self.query_token.expand(B, -1, -1)  # (B, 1, H)
        x_coarse, _ = self.cross_attention(query, x, x)
        x_coarse = self.cross_norm(x_coarse)        # (B, 1, H)

        # Predict coarse-scale outputs
        logits = self.output_proj(x_coarse)         # (B, 1, K)
        latent_pred = self.latent_proj(x_coarse)    # (B, 1, D)

        return logits, latent_pred