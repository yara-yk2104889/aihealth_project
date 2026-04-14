"""
models/mamba_blocks.py — Pure PyTorch Mamba-inspired blocks.
No external dependencies (no mamba-ssm required).

Uses a parallel cumsum-based scan instead of a sequential loop —
runs all tokens simultaneously on GPU, same speed regardless of sequence length.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MambaBottleneck(nn.Module):
    """
    Hybrid CNN + Mamba bottleneck in pure PyTorch.
    Parallelised scan — no sequential loop, fast on GPU at any sequence length.
    Drop-in replacement for Bottleneck — same __init__ signature.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 d_state: int = 16, d_conv: int = 4):
        super().__init__()
        self.d_state = d_state

        self.conv_in = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.norm    = nn.LayerNorm(out_channels)
        self.in_proj = nn.Linear(out_channels, out_channels * 2, bias=False)
        self.conv1d  = nn.Conv1d(out_channels, out_channels, d_conv,
                                  padding=d_conv - 1, groups=out_channels, bias=True)
        self.x_proj  = nn.Linear(out_channels, d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, out_channels, bias=True)

        A = torch.arange(1, d_state + 1).float().unsqueeze(0).expand(out_channels, -1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D     = nn.Parameter(torch.ones(out_channels))

        self.out_proj = nn.Linear(out_channels, out_channels, bias=False)
        self.act      = nn.SiLU()
        self.conv_out = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def _ssm(self, x_: torch.Tensor) -> torch.Tensor:
        """
        Parallelised SSM scan using cumsum — processes all tokens at once.
        Approximates the recurrence h_t = dA*h_{t-1} + dB*x_t in parallel.
        """
        A = -torch.exp(self.A_log.float())                     # [D, d_state]

        bcd = self.x_proj(x_)
        delta, B_m, C_m = bcd.split([1, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(delta))                # [B, L, D]

        # Discretise
        dA = torch.exp(delta.unsqueeze(-1) * A)                # [B, L, D, d_state]
        dB = delta.unsqueeze(-1) * B_m.unsqueeze(2)            # [B, L, D, d_state]

        # Input contribution at each step: u_t = dB_t * x_t
        u = dB * x_.unsqueeze(-1)                              # [B, L, D, d_state]

        # Parallel scan approximation via log-space cumsum
        # log(dA) cumsum gives cumulative decay, then scale inputs
        log_dA = torch.log(dA.clamp(min=1e-6))                # [B, L, D, d_state]
        cum_log_dA = torch.cumsum(log_dA, dim=1)              # [B, L, D, d_state]

        # Each output y_t = sum_{s<=t} u_s * exp(sum_{s<k<=t} log_dA_k)
        # Efficient: y = cumsum(u * exp(cum_log_dA - log_dA)) * exp(-cum_log_dA) -- approx
        # Simpler parallel approximation that preserves the gating structure:
        decay = torch.exp(cum_log_dA)                          # [B, L, D, d_state]
        h = torch.cumsum(u / decay.clamp(min=1e-6), dim=1) * decay  # [B, L, D, d_state]

        # Output: y_t = C_t * h_t  (summed over d_state)
        y = (h * C_m.unsqueeze(2)).sum(-1)                    # [B, L, D]
        return y + x_ * self.D

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)
        B, C, H, W = x.shape
        res = x

        seq   = self.norm(x.permute(0, 2, 3, 1).reshape(B, H * W, C))
        x_, z = self.in_proj(seq).chunk(2, dim=-1)
        x_    = self.conv1d(x_.transpose(1, 2))[:, :, :H*W].transpose(1, 2)
        x_    = self.act(x_)
        y     = self._ssm(x_) * self.act(z)
        out   = self.out_proj(y).reshape(B, H, W, C).permute(0, 3, 1, 2)

        return self.conv_out(out + res)
