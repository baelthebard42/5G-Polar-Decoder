import torch
import torch.nn as nn
import math

class CodeAwareMask:
     """
    Code-aware mask for polar codes with scattered frozen/message bits
    
    Works with ANY frozen bit pattern determined by reliability ranking
    """
     
     def __init__(self, frozen_prior):
        """
        Args:
            frozen_prior: (B, N) tensor with values {0, 1}
                         0 = frozen bit (unreliable position)
                         1 = message bit (reliable position)
                         
        Note: frozen_prior can have 0s and 1s at ANY positions
              based on channel reliability ranking
        """
        self.N = frozen_prior.shape[1]
        
        
        self.frozen_indices = (frozen_prior[0] == 0).nonzero(as_tuple=True)[0]
        self.message_indices = (frozen_prior[0] == 1).nonzero(as_tuple=True)[0]
        
        self.num_frozen = len(self.frozen_indices)
        self.num_message = len(self.message_indices)

     
     def create_mask(self, device='cuda'):
        """
        Creates attention mask for (N, 2N) attention matrix
        
        Attention structure:
        - Cols 0 to N-1: Channel observation embeddings
        - Cols N to 2N-1: Frozen bit prior embeddings
        
        Returns:
            mask: (N, 2N) boolean tensor
                  True = masked (blocked)
                  False = allowed
        """
        N = self.N
        mask = torch.zeros(N, 2*N, dtype=torch.bool, device=device)
        
        
        # ROW-WISE MASKING: Frozen Bit Rows
        
        # Each frozen bit only attends to itself
        # (it's a known constant, doesn't need context)
        
        for frozen_idx in self.frozen_indices:
            # Block ALL attention from this row
            mask[frozen_idx, :] = True
            
            # Allow ONLY self-attention in frozen embedding space
            # Column index = N + position_index
            mask[frozen_idx, N + frozen_idx] = False
        
      
        # COLUMN-WISE MASKING: Message Bit Rows
     
        # Message bits attend to:
        # - ALL channel observations (necessary for decoding)
        # - ONLY frozen bit positions (known priors)
        # - NOT other message bit positions (avoid circular dependency)
        
        for msg_idx in self.message_indices:
            # Channel observations (cols 0:N) - already allowed (False)
            # No action needed - message bits can see all channel obs
            
            # In frozen embedding space (cols N:2N):
            # Block attention to OTHER MESSAGE positions
            for other_msg_idx in self.message_indices:
                mask[msg_idx, N + other_msg_idx] = True
            
            # Frozen positions remain allowed (already False)
            # Message bits CAN attend to frozen bit embeddings
        
        return mask
     

class CodeAwareAttention(nn.Module):
    """
    Multi-head attention with code-aware masking
    """
    
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        
        self.attention = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=True
        )
        
    def forward(self, query, key, value, attn_mask=None):
        """
        Args:
            query: (B, N, d_model)
            key: (B, 2N, d_model)  # concatenated [channel, frozen]
            value: (B, 2N, d_model)
            attn_mask: (N, 2N) boolean mask
        
        Returns:
            output: (B, N, d_model)
        """
        if attn_mask is not None:
            # Expand mask for multi-head attention
            # PyTorch expects (B*nhead, N, 2N) or broadcastable shape
            B = query.shape[0]
            
            # Broadcast to (B, nhead, N, 2N) then reshape
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, N, 2N)
            attn_mask = attn_mask.expand(B, self.nhead, -1, -1)  # (B, nhead, N, 2N)
            attn_mask = attn_mask.reshape(B * self.nhead, query.shape[1], key.shape[1])
        
        output, _ = self.attention(
            query, key, value,
            attn_mask=attn_mask,
            need_weights=False
        )
        
        return output
     
    
     




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

        self.query_projection = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Separate projection for key/value (concatenated space)
        self.kv_projection = nn.Linear(d_model, d_model)

        # Positional Encoding
        self.positional_encoding = LearnablePositionalEncoding(seq_len, d_model)
        
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

        query = torch.cat([ch_emb, fr_emb], dim=-1)# (B, 32, 2*d_model)
        query = self.query_projection(query)  # (B, 32, d_model)
        query = self.positional_encoding(query)

        kv = torch.cat([ch_emb, fr_emb], dim=1)  # (B, 64, d_model)
        kv = self.kv_projection(kv)

        mask_generator = CodeAwareMask(frozen_prior)
        code_mask = mask_generator.create_mask(device=query.device)

        x = query
        
        for layer in self.encoder_layers:
         
            attn_out = layer['attention'](
                query=x,
                key=kv,
                value=kv,
                attn_mask=code_mask
            )
            x = layer['norm1'](x + attn_out)
            
        
            ffn_out = layer['ffn'](x)
            x = layer['norm2'](x + ffn_out)
        
        
        x = self.layer_norm(x)
        logits = self.output_head(x).squeeze(-1)  # (B, 32)
        
        return logits



       
def plot_mask_pattern(frozen_prior):
    """
    Visual representation of the mask
    """
    mask_gen = CodeAwareMask(frozen_prior)
    mask = mask_gen.create_mask(device='cpu').numpy()
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    N = frozen_prior.shape[1]
    
    # Create colored visualization
    # 0 (False) = green (allowed), 1 (True) = red (blocked)
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Invert for visualization (0=blocked in display)
    display_mask = (~torch.from_numpy(mask)).numpy().astype(float)
    
    im = ax.imshow(display_mask, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    # Add grid
    ax.set_xticks(np.arange(2*N))
    ax.set_yticks(np.arange(N))
    
    # Labels
    frozen_indices = mask_gen.frozen_indices.tolist()
    message_indices = mask_gen.message_indices.tolist()
    
    y_labels = []
    for i in range(N):
        if i in frozen_indices:
            y_labels.append(f'F{i}')
        else:
            y_labels.append(f'M{i}')
    
    ax.set_yticklabels(y_labels)
    
    # X-axis labels
    x_labels = [f'Ch{i}' if i < N else f'Fr{i-N}' for i in range(2*N)]
    ax.set_xticklabels(x_labels, rotation=90, fontsize=8)
    
    # Add vertical line to separate channel obs from frozen priors
    ax.axvline(x=N-0.5, color='blue', linewidth=2, linestyle='--')
    
    ax.set_xlabel('Attention Target (Channel Obs | Frozen Priors)')
    ax.set_ylabel('Attention Source (F=Frozen, M=Message)')
    ax.set_title(f'Code-Aware Attention Mask (N={N}, k={len(message_indices)})\nGreen=Allowed, Red=Blocked')
    
    plt.colorbar(im, ax=ax, label='Allowed')
    plt.tight_layout()
    plt.savefig('code_aware_mask.png', dpi=150)
    print("Saved visualization to 'code_aware_mask.png'")


# plot_mask_pattern(torch.tensor([
#     1, 0, 1, 1, 0, 1, 1, 0,
#     0, 1, 1, 0, 1, 0, 0, 1,
#     1, 1, 0, 0, 1, 0, 1, 0,
#     0, 0, 1, 1, 1, 1, 0, 1
# ]).unsqueeze(0))