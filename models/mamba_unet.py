"""
models/mamba_unet.py — U-Net with Mamba in deep encoder + bottleneck.

Mamba is applied only to the deeper encoder stages where spatial resolution
is small enough that the sequential SSM scan is fast and memory-efficient:

    Enc0  [32ch,  128×128 = 16384 tokens] → standard CNN
    Enc1  [64ch,   64×64  =  4096 tokens] → standard CNN
    Enc2  [128ch,  32×32  =  1024 tokens] → Mamba  ✓
    Enc3  [256ch,  16×16  =   256 tokens] → Mamba  ✓
    BN    [512ch,   8×8   =    64 tokens] → Mamba  ✓
    Dec3 … Dec0                           → standard CNN

The shallow encoder (Enc0, Enc1) stays as CNN because the long token
sequences (4096–16384) would be slow and memory-intensive for the SSM.
"""

from typing import Tuple
import torch
import torch.nn as nn

from .blocks import EncoderBlock, DecoderBlock, Bottleneck
from .mamba_blocks import MambaBottleneck


class MambaEncoderBlock(nn.Module):
    """
    Encoder block where the DoubleConv is replaced by MambaBottleneck.
    Same interface as EncoderBlock: returns (skip, pooled).
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = MambaBottleneck(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor):
        skip   = self.conv(x)
        pooled = self.pool(skip)
        return skip, pooled


class MambaEncoderUNet(nn.Module):
    """
    U-Net with Mamba applied to deep encoder stages + bottleneck.

    mamba_from controls which encoder stages use Mamba (0-indexed):
        mamba_from=2  →  Enc2, Enc3 + Bottleneck use Mamba  (default)
        mamba_from=3  →  Enc3 + Bottleneck only
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 1,
        features: Tuple[int, ...] = (32, 64, 128, 256),
        mamba_from: int = 2,
        bilinear: bool = True,
    ):
        super().__init__()
        self.features = features

        # ── Encoder ──────────────────────────────────────────────────────
        encoder_channels = [in_channels] + list(features)
        self.encoders = nn.ModuleList([
            MambaEncoderBlock(encoder_channels[i], encoder_channels[i + 1])
            if i >= mamba_from
            else EncoderBlock(encoder_channels[i], encoder_channels[i + 1])
            for i in range(len(features))
        ])
        print(f"[MambaEncoderUNet] Mamba encoder blocks: "
              f"{[i for i in range(len(features)) if i >= mamba_from]}")

        # ── Bottleneck (always Mamba) ─────────────────────────────────────
        bn_in  = features[-1]
        bn_out = features[-1] * 2
        self.bottleneck = MambaBottleneck(bn_in, bn_out)

        # ── Decoder (standard CNN) ────────────────────────────────────────
        decoder_in_channels = [bn_out] + [f for f in reversed(features[1:])]
        decoder_skip_ch     = list(reversed(features))
        decoder_out_ch      = list(reversed(features))

        self.decoders = nn.ModuleList([
            DecoderBlock(
                in_channels   = decoder_in_channels[i],
                skip_channels = decoder_skip_ch[i],
                out_channels  = decoder_out_ch[i],
                bilinear      = bilinear,
            )
            for i in range(len(features))
        ])

        self.out_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for enc in self.encoders:
            skip, x = enc(x)
            skips.append(skip)

        x = self.bottleneck(x)

        for dec, skip in zip(self.decoders, reversed(skips)):
            x = dec(x, skip)

        return self.out_conv(x)

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        with torch.no_grad():
            return (torch.sigmoid(self.forward(x)) > threshold).float()

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
