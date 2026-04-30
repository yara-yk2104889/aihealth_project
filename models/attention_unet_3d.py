"""
models/attention_unet_3d.py — 3-D Attention U-Net for volumetric stroke lesion segmentation.

Architecture (default features=(16,32,64,128))
----------------------------------------------
    Input  [B, 2,  D,    H,    W   ]   (DWI + ADC, full 3-D volume)
    Enc0   [B, 16, D,    H,    W   ]   → skip0
    Enc1   [B, 32, D/2,  H/2,  W/2 ]   → skip1
    Enc2   [B, 64, D/4,  H/4,  W/4 ]   → skip2
    Enc3   [B,128, D/8,  H/8,  W/8 ]   → skip3
    BN     [B,256, D/16, H/16, W/16]
    Dec3   [B,128, D/8,  H/8,  W/8 ]   (trilinear up + attention on skip3 + DoubleConv)
    Dec2   [B, 64, D/4,  H/4,  W/4 ]
    Dec1   [B, 32, D/2,  H/2,  W/2 ]
    Dec0   [B, 16, D,    H,    W   ]
    Out    [B,  1, D,    H,    W   ]   (raw logit)

Attention gate (Oktay et al. 2018):
    alpha = sigmoid( psi( ReLU( W_g(g) + W_x(x) ) ) )
    attended_skip = x * alpha

Reference: Oktay et al., "Attention U-Net: Learning Where to Look for the Pancreas", MIDL 2018.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Building blocks ───────────────────────────────────────────────────────────

class DoubleConv3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class EncoderBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = DoubleConv3D(in_channels, out_channels)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor):
        skip   = self.conv(x)
        pooled = self.pool(skip)
        return skip, pooled


class AttentionGate3D(nn.Module):
    """
    Soft attention gate.
    g     — gating signal from the decoder path (same spatial res as skip)
    x     — skip connection from the encoder
    Returns x weighted by learned spatial attention coefficients alpha in [0, 1].
    """

    def __init__(self, gate_channels: int, skip_channels: int, inter_channels: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(gate_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(inter_channels),
        )
        self.W_x = nn.Sequential(
            nn.Conv3d(skip_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(inter_channels),
        )
        self.psi = nn.Sequential(
            nn.Conv3d(inter_channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm3d(1),
            nn.Sigmoid(),
        )

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g_proj = self.W_g(g)
        x_proj = self.W_x(x)
        # Align spatial dims in case of boundary rounding after upsample
        if g_proj.shape[2:] != x_proj.shape[2:]:
            g_proj = F.interpolate(
                g_proj, size=x_proj.shape[2:], mode="trilinear", align_corners=False
            )
        alpha = self.psi(F.relu(g_proj + x_proj, inplace=True))  # [B,1,D,H,W]
        return x * alpha


class AttentionDecoderBlock3D(nn.Module):
    """
    Trilinear upsample → attention-weighted skip → concat → DoubleConv.
    in_channels  : channels coming from the deeper decoder level
    skip_channels: encoder skip connection channels (= gating signal channels after upsample)
    out_channels : output channels
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
            nn.Conv3d(in_channels, skip_channels, kernel_size=1, bias=False),
        )
        self.attention = AttentionGate3D(
            gate_channels  = skip_channels,
            skip_channels  = skip_channels,
            inter_channels = max(skip_channels // 2, 1),
        )
        # concat: skip_channels (upsampled) + skip_channels (attended skip)
        self.conv = DoubleConv3D(skip_channels * 2, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        g        = self.upsample(x)               # gating signal, aligned to skip
        attended = self.attention(g, skip)        # attention-weighted skip
        x_cat    = torch.cat([g, attended], dim=1)
        return self.conv(x_cat)


class Bottleneck3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = DoubleConv3D(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# ── Full model ────────────────────────────────────────────────────────────────

class AttentionUNet3D(nn.Module):
    """
    3-D Attention U-Net.

    Parameters
    ----------
    in_channels  : number of input modalities (2 = DWI + ADC)
    out_channels : 1 for binary segmentation
    features     : encoder channel widths at each level
                   default (16,32,64,128) balances capacity and memory for
                   batch_size=8 on volumes of size (32,128,128)
    """

    def __init__(
        self,
        in_channels:  int           = 2,
        out_channels: int           = 1,
        features:     Tuple[int, ...]= (16, 32, 64, 128),
    ):
        super().__init__()
        self.features = features

        # ── Encoder ──────────────────────────────────────────────────────
        enc_ch = [in_channels] + list(features)
        self.encoders = nn.ModuleList([
            EncoderBlock3D(enc_ch[i], enc_ch[i + 1])
            for i in range(len(features))
        ])

        # ── Bottleneck ───────────────────────────────────────────────────
        bn_in  = features[-1]
        bn_out = features[-1] * 2
        self.bottleneck = Bottleneck3D(bn_in, bn_out)

        # ── Decoder ──────────────────────────────────────────────────────
        # dec_in[i]   = output from previous level (bn_out, then each decoder output)
        # dec_skip[i] = matching encoder skip channels
        # dec_out[i]  = this decoder block's output channels
        dec_in   = [bn_out] + list(reversed(features[1:]))
        dec_skip = list(reversed(features))
        dec_out  = list(reversed(features))

        self.decoders = nn.ModuleList([
            AttentionDecoderBlock3D(
                in_channels   = dec_in[i],
                skip_channels = dec_skip[i],
                out_channels  = dec_out[i],
            )
            for i in range(len(features))
        ])

        # ── Output ───────────────────────────────────────────────────────
        self.out_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : FloatTensor [B, in_channels, D, H, W]

        Returns
        -------
        logits : FloatTensor [B, out_channels, D, H, W]   (raw, pre-sigmoid)
        """
        skips = []
        for enc in self.encoders:
            skip, x = enc(x)
            skips.append(skip)

        x = self.bottleneck(x)

        for dec, skip in zip(self.decoders, reversed(skips)):
            x = dec(x, skip)

        return self.out_conv(x)

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Returns binary mask (0/1) without grad."""
        with torch.no_grad():
            return (torch.sigmoid(self.forward(x)) > threshold).float()

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
