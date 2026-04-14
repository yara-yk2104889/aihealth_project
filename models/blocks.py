"""
models/blocks.py — Reusable encoder / decoder building blocks.

Design for extensibility
-------------------------
Every block is a plain nn.Module with a documented interface.
To integrate Mamba later:

    class MambaBlock(nn.Module):
        \"\"\"Drop-in replacement for DoubleConv.\"\"\"
        def __init__(self, in_ch, out_ch, ...): ...
        def forward(self, x): ...   # same signature

    class MambaEncoderBlock(EncoderBlock):
        def __init__(self, in_ch, out_ch):
            nn.Module.__init__(self)            # skip EncoderBlock.__init__
            self.conv = MambaBlock(in_ch, out_ch)
            self.pool = nn.MaxPool2d(2)

Then pass MambaEncoderBlock to UNet(encoder_block_cls=MambaEncoderBlock).
"""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """
    Two consecutive Conv2d → BatchNorm → ReLU layers.
    The basic spatial feature extractor used everywhere in U-Net.
    """

    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = None):
        super().__init__()
        mid_channels = mid_channels or out_channels
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class EncoderBlock(nn.Module):
    """
    One encoder stage: DoubleConv followed by MaxPool2d(2).

    forward() returns (skip, pooled):
        skip   — features before pooling   → concatenated with decoder skip
        pooled — downsampled features       → passed to the next encoder stage
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor):
        skip   = self.conv(x)
        pooled = self.pool(skip)
        return skip, pooled


class DecoderBlock(nn.Module):
    """
    One decoder stage.

    Steps:
        1. Upsample  (bilinear ×2 + 1×1 conv to halve channels)
        2. Concatenate with skip connection from the matching encoder stage
        3. DoubleConv

    Parameters
    ----------
    in_channels   : channels arriving from the layer below (before upsample)
    skip_channels : channels of the matching encoder skip
    out_channels  : desired output channel count
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int,
                 bilinear: bool = True):
        super().__init__()
        if bilinear:
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            )
        else:
            self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

        self.conv = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)

        # Pad if spatial dims don't match exactly (edge case with odd input sizes)
        if x.shape != skip.shape:
            x = nn.functional.pad(x, [
                0, skip.shape[-1] - x.shape[-1],
                0, skip.shape[-2] - x.shape[-2],
            ])

        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class Bottleneck(nn.Module):
    """
    The deepest U-Net block (no skip connection).
    Replace with a MambaBottleneck to add global context without attention.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)
