# Faithful PyTorch implementation of ECCM (Hybrid Mamba–Transformer Decoder)
# Specialized for Polar codes of length N=32, closely following the paper
# NOTE: This implementation is architecturally faithful. Practical optimizations are intentionally avoided.

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# -----------------------------
# Utilities: Masks from parity-check matrix H
# -----------------------------

def graph_mask_from_H(H):
    """Implements g(H) from Eq.(2)(3)
    H: (n-k, n) binary
    Returns g(H) in {0,1}^{(2n-k)x(2n-k)}
    """
    device = H.device
    n_k, n = H.shape
    L = 2 * n - n_k

    Graph = torch.zeros((n, n), device=device)
    for m in range(n_k):
        idx = torch.where(H[m] == 1)[0]
        for i in idx:
            for j in idx:
                Graph[i, j] = 1

    top = torch.cat([Graph, H.t()], dim=1)
    bottom = torch.cat([H, torch.eye(n_k, device=device)], dim=1)
    gH = torch.cat([top, bottom], dim=0)
    return gH


def f_mask_from_H(H):
    """Implements f(H) = [H; I_{n-k}] Eq.(4)
    Returns (n-k, 2n-k)
    """
    n_k, n = H.shape
    test = torch.cat([H, torch.eye(n_k, device=H.device)], dim=1)
    print(f"shape of f mask: {test.shape}, n_k={n_k}, 2n-k={n+n_k}")
    assert test.shape[0] == n_k and test.shape[1] == n+n_k 
    return test

class ECCM_MambaBlock(nn.Module):
    def __init__(self, D, S, H):
        """
        D: embedding dimension
        S: state dimension (must be >= n-k)
        H: parity check matrix, shape (n-k, n)
        """
        super().__init__()

        n_k, n = H.shape
        L = 2 * n - n_k

        assert S >= n_k, "State dimension S must be >= (n-k)"

        self.D = D
        self.S = S
        self.L = L
        self.n_k = n_k

        # ---- register f(H) ----
        fH = f_mask_from_H(H).T  # (2n-k, n-k)
        self.register_buffer("fH", fH)

        # ---- projections ----
        self.Wu = nn.Linear(D, D, bias=False)
        self.Wz = nn.Linear(D, D, bias=False)

        self.conv = nn.Conv1d(
            D, D, kernel_size=3, padding=1, groups=D
        )

        self.Wb = nn.Linear(D, S, bias=False)
        self.Wc = nn.Linear(D, S, bias=False)
        self.Wd = nn.Linear(D, D, bias=False)

        # A ∈ R^{D×S}
        self.A = nn.Parameter(0.1 * torch.randn(D, S))

        # Skip connection
        self.R = nn.Parameter(torch.zeros(D))

    def _ssm_direction(self, u_conv: torch.Tensor, fH: torch.Tensor):
        """
        u_conv: [B, L, D]
        fH: [L, n-k]
        """
        B, L, D = u_conv.shape
        device = u_conv.device

        # state h ∈ [B, D, S]
        h = torch.zeros(B, D, self.S, device=device)
        y = torch.zeros_like(u_conv)

        for l in range(L):
            ul = u_conv[:, l]  # [B, D]

            # projections
            B_l = self.Wb(ul)  # [B, S]
            C_l = self.Wc(ul)  # [B, S]
            delta = torch.clamp(self.Wd(ul), -2.0, 2.0)  # [B, D]

            # discretization
            Abar = torch.exp(
                torch.clamp(
                    self.A.unsqueeze(0) * delta.unsqueeze(-1),
                    min=-20.0,
                    max=20.0,
                )
            )  # [B, D, S]

            Bbar = B_l.unsqueeze(1) * delta.unsqueeze(-1)  # [B, D, S]

            # ---- masking over STATE dimension ----
            state_mask = torch.zeros(self.S, device=device)
            state_mask[: self.n_k] = fH[l]  # [n-k]

            Bbar = Bbar * state_mask.view(1, 1, self.S)
            C_l = C_l * state_mask.view(1, self.S)

            # state update
            h = Abar * h + Bbar * ul.unsqueeze(-1)

            # output
            y[:, l] = (h * C_l.unsqueeze(1)).sum(dim=-1) + self.R * ul

        return y

    def forward(self, x):
        """
        x: [B, L, D]
        """
        # ---- forward direction ----
        u = self.Wu(x)
        z = F.silu(self.Wz(x))

        u_conv = self.conv(u.transpose(1, 2)).transpose(1, 2)
        y_fwd = self._ssm_direction(u_conv, self.fH)
        out_fwd = z * y_fwd

        # ---- backward direction ----
        x_rev = torch.flip(x, dims=[1])
        u_rev = self.Wu(x_rev)
        z_rev = F.silu(self.Wz(x_rev))
        u_conv_rev = self.conv(u_rev.transpose(1, 2)).transpose(1, 2)

        fH_rev = torch.flip(self.fH, dims=[0])
        y_bwd_rev = self._ssm_direction(u_conv_rev, fH_rev)
        out_bwd = torch.flip(z_rev * y_bwd_rev, dims=[1])

        return out_fwd + out_bwd

# -----------------------------
# Transformer Block (HPSA masked attention)
# -----------------------------

class ECCM_AttnBlock(nn.Module):
    def __init__(self, D, H, heads=8):
        super().__init__()
        self.D = D
        self.h = heads
        self.dk = D // heads
        
        self.Q = nn.Linear(D, D)
        self.K = nn.Linear(D, D)
        self.V = nn.Linear(D, D)
        self.O = nn.Linear(D, D)

        self.ff1 = nn.Linear(D, 4 * D)
        self.ff2 = nn.Linear(4 * D, D)
        self.ln = nn.LayerNorm(D)
        gH = graph_mask_from_H(H)
        self.register_buffer('gH', gH)
        
    
        

    def forward(self, x):
        """
        x: [B, L, D] where L = 2n-k
        """
        B, L, D = x.shape
        
        # Equations 21-22: Compute Q, K, V and reshape to multi-head
        q = self.Q(x).view(B, L, self.h, self.dk).transpose(1, 2)  # [B, h, L, dk]
        k = self.K(x).view(B, L, self.h, self.dk).transpose(1, 2)  # [B, h, L, dk]
        v = self.V(x).view(B, L, self.h, self.dk).transpose(1, 2)  # [B, h, L, dk]

        # Equation 23: HPSA (Hierarchical Parity-check Structured Attention)
        attn = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.dk)  # [B, h, L, L]
        
        # Apply mask: broadcast gH from [L, L] to [B, h, L, L]
        mask = self.gH[:L, :L].unsqueeze(0).unsqueeze(0)  # [1, 1, L, L]
        attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)

        # Compute attention output
        o = torch.matmul(attn, v)  # [B, h, L, dk]
        
        # Equation 24: Concatenate heads
        o = o.transpose(1, 2).contiguous().view(B, L, D)  # [B, L, D]
        
        # Equation 25: Output projection
        y_a = self.O(o)  # [B, L, D]

        y_a_norm = self.ln(y_a + x) 
     
        
        # Equation 27: Feed-forward network
        y = y_a_norm + self.ff2(F.relu(self.ff1(y_a_norm)))  # [B, L, D]
        
        return y
# -----------------------------
# Full ECCM Model
# -----------------------------

class ECCM(nn.Module):
    def __init__(self, H, device, N=32, k=11, D=128, S=128, blocks=8):
        super().__init__()
        self.H = H.to(device)
        self.N = N
        n_k = self.H.shape[0]
        self.L = N+n_k # 2n-k

        self.embed = nn.Parameter(torch.randn(self.L, D)/math.sqrt(D))

        layers = []
        for i in range(blocks):
            if i % 2 == 0:
                layers.append(ECCM_MambaBlock(D, S, self.H))
            else:
                layers.append(ECCM_AttnBlock(D, self.H))
        self.layers = nn.ModuleList(layers)

        self.wr = nn.Linear(D, 1)
        self.Ws = nn.Linear(self.L, N)

    def forward(self, yin, syndrome):
        # yin: (B, L)
        x = yin.unsqueeze(-1) * self.embed.unsqueeze(0)
        outputs = []

        for layer in self.layers:
            x = layer(x)
            o = torch.sigmoid(self.Ws(self.wr(x).squeeze(-1)))
            outputs.append(o)

            # Early stopping (Eq.29)
            s_hat = (o > 0.5).float() @ self.H.t() % 2
            if torch.all(s_hat == syndrome):
                break

        return outputs


# -----------------------------
# Loss (Eq.31-32)
# -----------------------------

def eccm_loss(outputs, xin, c):
    loss = 0.0
    z = (1 - torch.sign((1 - 2 * c) * xin)) / 2
    for o in outputs:
        loss += F.binary_cross_entropy(o, z)
    return loss


def get_estimated_codeword(output, y_in):

    return (1.0 - torch.sign((1.0 - 2.0 * output) * y_in)) / 2.0


