"""
models/mamba_blocks.py — Pure PyTorch Mamba-inspired blocks.
No external dependencies (no mamba-ssm required).

MambaBottleneck is a drop-in replacement for Bottleneck in both
UNet and DualEncoderUNet.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MambaBottleneck(nn.Module):
    """
    Hybrid CNN + Mamba bottleneck in pure PyTorch.

    Design
    ------
    1. Conv projects to out_channels
    2. Mamba SSM captures long-range spatial context over H*W tokens
    3. Conv refines

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
        self.norm = nn.LayerNorm(out_channels)

        # Mamba branches
        self.in_proj  = nn.Linear(out_channels, out_channels * 2, bias=False)
        self.conv1d   = nn.Conv1d(out_channels, out_channels, d_conv,
                                   padding=d_conv - 1, groups=out_channels, bias=True)
        self.x_proj   = nn.Linear(out_channels, d_state * 2 + 1, bias=False)
        self.dt_proj  = nn.Linear(1, out_channels, bias=True)

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
        B, L, D = x_.shape
        A = -torch.exp(self.A_log.float())                          # [D, d_state]

        bcd   = self.x_proj(x_)
        delta, B_m, C_m = bcd.split([1, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(delta))                     # [B, L, D]

        dA = torch.exp(delta.unsqueeze(-1) * A)                    # [B, L, D, d_state]
        dB = delta.unsqueeze(-1) * B_m.unsqueeze(2)                # [B, L, D, d_state]

        h  = torch.zeros(B, D, self.d_state, device=x_.device)
        ys = []
        for i in range(L):
            h = dA[:, i] * h + dB[:, i] * x_[:, i].unsqueeze(-1)
            ys.append((h * C_m[:, i].unsqueeze(1)).sum(-1))

        y = torch.stack(ys, dim=1)                                  # [B, L, D]
        return y + x_ * self.D

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x   = self.conv_in(x)
        B, C, H, W = x.shape
        res = x

        seq      = self.norm(x.permute(0, 2, 3, 1).reshape(B, H * W, C))
        x_, z    = self.in_proj(seq).chunk(2, dim=-1)
        x_       = self.conv1d(x_.transpose(1, 2))[:, :, :H*W].transpose(1, 2)
        x_       = self.act(x_)
        y        = self._ssm(x_) * self.act(z)
        out      = self.out_proj(y).reshape(B, H, W, C).permute(0, 3, 1, 2)

        return self.conv_out(out + res)
