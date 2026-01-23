import torch
import torch.nn as nn
import math



class CodeAwareMask:
    """
    Paper-faithful code-aware causal mask for polar decoding (SC-style)

    Mask shape: (N, N)
    True  = blocked (attention not allowed)
    False = allowed (attention permitted)
    """

    def __init__(self, frozen_prior):
        """
        Args:
            frozen_prior: (B, N) tensor where 0=frozen, 1=information bit
        """
        self.N = frozen_prior.shape[1]
        self.frozen_prior = frozen_prior
        # Frozen bits are where frozen_prior == 0
        self.frozen = frozen_prior[0] == 0  # Use first sample (assumes same pattern across batch)

    def create_mask(self, device='cuda'):
        """
        Create code-aware attention mask
        
        Returns:
            mask: (N, N) boolean tensor
        """
        N = self.N
        mask = torch.zeros(N, N, dtype=torch.bool, device=device)

        for i in range(N):
            if self.frozen[i]:
                # Frozen bits attend ONLY to themselves
                # (their value is predetermined, no need for context)
                mask[i, :] = True
                mask[i, i] = False
            else:
                # Information bits: causal attention only
                # Can attend to all previous positions (0 to i-1) and itself
                mask[i, i+1:] = True  # Block future positions

        return mask


class CodeAwareAttention(nn.Module):
    """
    Multi-head attention with code-aware masking for polar codes
    """

    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.nhead = nhead
        self.attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=True
        )

    def forward(self, query, key, value, attn_mask=None):
        """
        Apply multi-head attention with optional masking
        
        Args:
            query: (B, N, d_model)
            key: (B, N, d_model)
            value: (B, N, d_model)
            attn_mask: (N, N) boolean mask
            
        Returns:
            out: (B, N, d_model)
        """
        B, N, _ = query.shape

        if attn_mask is not None:
            # Expand mask for all heads and batches
            # PyTorch MultiheadAttention expects (B*nhead, N, N) or broadcastable shape
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, N, N)
            attn_mask = attn_mask.expand(B, self.nhead, N, N)  # (B, nhead, N, N)
            attn_mask = attn_mask.reshape(B * self.nhead, N, N)  # (B*nhead, N, N)

        out, _ = self.attn(
            query, key, value,
            attn_mask=attn_mask,
            need_weights=False
        )
        return out


class LearnablePositionalEncoding(nn.Module):
    """
    Learnable positional encoding for transformer
    """
    
    def __init__(self, max_len, d_model):
        super().__init__()
        self.encoding = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
    
    def forward(self, x):
        """
        Args:
            x: (B, seq_len, d_model)
        Returns:
            x + positional encoding: (B, seq_len, d_model)
        """
        return x + self.encoding[:, :x.size(1), :]



class TransformerPolarDecoder_v3(nn.Module):
    """
    Paper-faithful Transformer-based Polar Decoder (N=32)

    - One token per bit
    - Q = K = V from same tensor
    - All structure via code-aware mask
    """

    def __init__(
        self,
        d_model=64,
        nhead=4,
        num_layers=4,
        seq_len=32,
        dim_feedforward=256,
        use_mask=False,
        dropout=0.1,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.use_mask = use_mask
        

        # Embeddings (paper-style)
        self.channel_embedding = nn.Linear(1, d_model)
        self.frozen_embedding = nn.Embedding(2, d_model)

        self.positional_encoding = LearnablePositionalEncoding(seq_len, d_model)

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attn': CodeAwareAttention(d_model, nhead, dropout),
                'norm1': nn.LayerNorm(d_model),
                'ffn': nn.Sequential(
                    nn.Linear(d_model, dim_feedforward),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(dim_feedforward, d_model),
                    nn.Dropout(dropout)
                ),
                'norm2': nn.LayerNorm(d_model)
            })
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)

        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(d_model // 2, 1)
        )

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

        B, N = channel_ob_vector.shape
        assert N == self.seq_len

        # (B, N, d)
        ch_emb = self.channel_embedding(channel_ob_vector.unsqueeze(-1))
        fr_emb = self.frozen_embedding(frozen_prior)

        # Single bit-token representation (paper eq.)
        x = ch_emb + fr_emb
        x = self.positional_encoding(x)

        # Code-aware mask
        mask = None
        if self.use_mask:
         mask = CodeAwareMask(frozen_prior).create_mask(device=x.device)

        for layer in self.layers:
            attn_out = layer['attn'](x, attn_mask=mask)
            x = layer['norm1'](x + attn_out)

            ffn_out = layer['ffn'](x)
            x = layer['norm2'](x + ffn_out)

        x = self.final_norm(x)

        logits = self.output_head(x).squeeze(-1)
        return logits

     




class TransformerPolarDecoder_v2(nn.Module):
    """
    Transformer-based Polar Decoder (N=32) with Code-Aware Masking

    Input:
        channel_ob_vector: (B, 32) - Channel observations (LLRs or soft values)
        frozen_prior:      (B, 32) - Frozen bit indicator {0=frozen, 1=information}
        SNR_db:            (B,)    - Optional SNR values (kept for compatibility)

    Output:
        logits:            (B, 32) - Decoded bit logits
    """

    def __init__(
        self,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 4,
        seq_len: int = 32,
        dim_feedforward: int = 256,
            use_mask=False,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.use_mask = use_mask

        # Input embeddings
        self.channel_embedding = nn.Linear(1, d_model)
        self.frozen_embedding = nn.Embedding(2, d_model)

        # Combined embedding projection
        self.input_projection = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Positional encoding (shared for all positions)
        self.positional_encoding = LearnablePositionalEncoding(seq_len, d_model)
        
        # Transformer encoder layers with code-aware attention
        self.encoder_layers = nn.ModuleList([
            nn.ModuleDict({
                'attention': CodeAwareAttention(d_model, nhead, dropout),
                'norm1': nn.LayerNorm(d_model),
                'ffn': nn.Sequential(
                    nn.Linear(d_model, dim_feedforward),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(dim_feedforward, d_model),
                    nn.Dropout(dropout)
                ),
                'norm2': nn.LayerNorm(d_model)
            })
            for _ in range(num_layers)
        ])

        # Final layer norm
        self.layer_norm = nn.LayerNorm(d_model)

        # Output head for bit predictions
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(d_model // 2, 1)
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier/normal initialization"""
        nn.init.xavier_uniform_(self.channel_embedding.weight)
        nn.init.zeros_(self.channel_embedding.bias)
        nn.init.normal_(self.frozen_embedding.weight, std=0.02)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, channel_ob_vector, frozen_prior, SNR_db=None):
        """
        Forward pass through the polar decoder
        
        Args:
            channel_ob_vector: (B, 32) - Channel observations
            frozen_prior: (B, 32) - Frozen bit mask (0=frozen, 1=info)
            SNR_db: (B,) - Optional SNR (unused)
            
        Returns:
            logits: (B, 32) - Bit predictions
        """
        if channel_ob_vector.shape[1] != self.seq_len:
            raise ValueError(f"Expected sequence length {self.seq_len}, got {channel_ob_vector.shape[1]}")

        B = channel_ob_vector.shape[0]

        # Embed channel observations and frozen indicators
        ch_emb = self.channel_embedding(channel_ob_vector.unsqueeze(-1))  # (B, 32, d_model)
        fr_emb = self.frozen_embedding(frozen_prior)                       # (B, 32, d_model)

        # Combine embeddings and project to d_model
        combined = torch.cat([ch_emb, fr_emb], dim=-1)  # (B, 32, 2*d_model)
        x = self.input_projection(combined)              # (B, 32, d_model)
        
        # Add positional encoding
        x = self.positional_encoding(x)                  # (B, 32, d_model)

        # Generate code-aware mask (same for all layers)
        code_mask = None
        if self.use_mask:
         mask_generator = CodeAwareMask(frozen_prior)
         code_mask = mask_generator.create_mask(device=x.device)  # (N, N)

        # Pass through transformer layers with self-attention
        for layer in self.encoder_layers:
            # Code-aware self-attention
            attn_out = layer['attention'](
                query=x,
                key=x,
                value=x,
                attn_mask=code_mask
            )
            x = layer['norm1'](x + attn_out)
            
            # Feed-forward network
            ffn_out = layer['ffn'](x)
            x = layer['norm2'](x + ffn_out)

        # Final layer norm
        x = self.layer_norm(x)  # (B, 32, d_model)
        
        # Generate bit logits
        logits = self.output_head(x).squeeze(-1)  # (B, 32)
        
        return logits
    

class TransformerPolarDecoder_v1(nn.Module):
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
    




def plot_code_aware_mask(frozen_prior):
    """
    Visualize paper-faithful code-aware attention mask (N x N)

    Args:
        frozen_prior: (1, N) tensor with values {0,1}
                      0 = frozen bit
                      1 = message bit
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    mask_gen = CodeAwareMask(frozen_prior)
    mask = mask_gen.create_mask(device='cpu')  # (N, N)

    N = frozen_prior.shape[1]
    frozen_indices = mask_gen.frozen.nonzero(as_tuple=True)[0].tolist()

    # Convert mask for visualization
    # True (blocked) -> 0 (red)
    # False (allowed) -> 1 (green)
    display_mask = (~mask).float().numpy()

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(
        display_mask,
        cmap='RdYlGn',
        vmin=0,
        vmax=1,
        aspect='equal'
    )

    # Axis ticks
    ax.set_xticks(np.arange(N))
    ax.set_yticks(np.arange(N))

    # Labels
    y_labels = []
    for i in range(N):
        if frozen_prior[0, i] == 0:
            y_labels.append(f'F{i}')
        else:
            y_labels.append(f'M{i}')

    ax.set_yticklabels(y_labels)
    ax.set_xticklabels([f'Bit{j}' for j in range(N)], rotation=90)

    # Grid
    ax.set_xticks(np.arange(-0.5, N, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, N, 1), minor=True)
    ax.grid(which='minor', color='black', linewidth=0.5)
    ax.tick_params(which='minor', bottom=False, left=False)

    ax.set_xlabel("Key / Value Bit Index")
    ax.set_ylabel("Query Bit Index")
    ax.set_title(
        f"Paper-Faithful Code-Aware Attention Mask (N={N}, "
        f"k={int(frozen_prior.sum())})\n"
        "Green = Allowed, Red = Blocked"
    )

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Allowed Attention")
    plt.tight_layout()
    plt.savefig('code_aware_mask.png', dpi=150)
    print("Saved visualization to 'code_aware_mask.png'")


# frozen_prior = torch.tensor([
#     1, 0, 1, 1, 0, 1, 1, 0,
#     0, 1, 1, 0, 1, 0, 0, 1,
#     1, 1, 0, 0, 1, 0, 1, 0,
#     0, 0, 1, 1, 1, 1, 0, 1
# ]).unsqueeze(0)

# plot_code_aware_mask(frozen_prior)

