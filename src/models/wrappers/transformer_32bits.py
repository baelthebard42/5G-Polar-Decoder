import torch
import torch.nn as nn
import math


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, seq_len, d_model):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.normal_(self.pos_embedding, std=0.02)

    def forward(self, x):
        return x + self.pos_embedding

class TransformerPolarDecoder(nn.Module):
    """
    Transformer-based Polar Decoder (N=32)

    Input:
        channel_ob_vector: (B, 32)
        frozen_prior:      (B, 32)  values in {0,1}
        SNR_db:            (B,) or unused (kept for compatibility)

    Output:
        logits:            (B, 32)
    """

    def __init__(
        self,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 4,
        seq_len: int = 32,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers

        # Embeddings
        self.channel_embedding = nn.Linear(1, d_model)
        self.frozen_embedding = nn.Embedding(2, d_model)

        self.input_projection = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Positional Encoding
        self.positional_encoding = LearnablePositionalEncoding(seq_len, d_model)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,   # pre-norm (important for stability)
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(d_model // 2, 1)
        )

        self.layer_norm = nn.LayerNorm(d_model)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.channel_embedding.weight)
        nn.init.zeros_(self.channel_embedding.bias)
        nn.init.normal_(self.frozen_embedding.weight, std=0.02)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, channel_ob_vector, frozen_prior, SNR_db=None):

        if channel_ob_vector.shape[1] != self.seq_len:
            raise ValueError(f"Expected sequence length {self.seq_len}")

        # (B, 32, d_model)
        ch_emb = self.channel_embedding(channel_ob_vector.unsqueeze(-1))
        fr_emb = self.frozen_embedding(frozen_prior)

        # Concatenate channel + frozen prior
        x = torch.cat([ch_emb, fr_emb], dim=-1)
        x = self.input_projection(x)

        # Add positional information
        x = self.positional_encoding(x)

        # Transformer Encoder
        x = self.encoder(x)

        # Normalize
        x = self.layer_norm(x)

        # Bit-wise logits
        logits = self.output_head(x).squeeze(-1)

        return logits
