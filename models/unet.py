"""
models/unet.py — Baseline U-Net for stroke lesion segmentation.

Architecture (default features=(32,64,128,256))
-----------------------------------------------
    Input  [B, 2, H, W]         (DWI + ADC)
    Enc0   [B, 32,  H/1, W/1]   → skip0
    Enc1   [B, 64,  H/2, W/2]   → skip1
    Enc2   [B, 128, H/4, W/4]   → skip2
    Enc3   [B, 256, H/8, W/8]   → skip3
    BN     [B, 512, H/16,W/16]
    Dec3   [B, 256, H/8, W/8]   (upsample + cat skip3)
    Dec2   [B, 128, H/4, W/4]   (upsample + cat skip2)
    Dec1   [B, 64,  H/2, W/2]   (upsample + cat skip1)
    Dec0   [B, 32,  H/1, W/1]   (upsample + cat skip0)
    Out    [B, 1,   H,   W]     (raw logit → sigmoid for inference)

Extension points
----------------
Pass custom block classes to swap encoder / decoder / bottleneck:

    from models.blocks import EncoderBlock, DecoderBlock, Bottleneck
    # (future)
    from models.mamba_blocks import MambaEncoderBlock, MambaBottleneck

    model = UNet(
        encoder_block_cls = MambaEncoderBlock,
        bottleneck_cls    = MambaBottleneck,
    )

SAM refinement: apply a SAMRefiner on top of the logit output in train.py.
"""

from typing import Tuple, Type

import torch
import torch.nn as nn

from .blocks import EncoderBlock, DecoderBlock, Bottleneck


class UNet(nn.Module):

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 1,
        features: Tuple[int, ...] = (32, 64, 128, 256),
        encoder_block_cls: Type[EncoderBlock] = EncoderBlock,
        decoder_block_cls: Type[DecoderBlock] = DecoderBlock,
        bottleneck_cls: Type[Bottleneck] = Bottleneck,
        bilinear: bool = True,
    ):
        super().__init__()
        self.features = features

        # ── Encoder ──────────────────────────────────────────────────────
        encoder_channels = [in_channels] + list(features)
        self.encoders = nn.ModuleList([
            encoder_block_cls(encoder_channels[i], encoder_channels[i + 1])
            for i in range(len(features))
        ])

        # ── Bottleneck ───────────────────────────────────────────────────
        bn_in  = features[-1]
        bn_out = features[-1] * 2
        self.bottleneck = bottleneck_cls(bn_in, bn_out)

        # ── Decoder ──────────────────────────────────────────────────────
        # Reversed: deepest → shallowest
        # At each level:  in_ch from below, skip_ch from matching encoder, out_ch = features[i]
        decoder_in_channels = [bn_out] + [f for f in reversed(features[1:])]
        decoder_skip_ch     = list(reversed(features))
        decoder_out_ch      = list(reversed(features))

        self.decoders = nn.ModuleList([
            decoder_block_cls(
                in_channels   = decoder_in_channels[i],
                skip_channels = decoder_skip_ch[i],
                out_channels  = decoder_out_ch[i],
                bilinear      = bilinear,
            )
            for i in range(len(features))
        ])

        # ── Output ───────────────────────────────────────────────────────
        self.out_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : FloatTensor [B, in_channels, H, W]

        Returns
        -------
        logits : FloatTensor [B, out_channels, H, W]
            Raw (pre-sigmoid) logits.  Apply torch.sigmoid() for inference.
        """
        skips = []

        # Encoder path
        for enc in self.encoders:
            skip, x = enc(x)
            skips.append(skip)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder path (reverse skip order)
        for dec, skip in zip(self.decoders, reversed(skips)):
            x = dec(x, skip)

        return self.out_conv(x)

    # ── Utility ──────────────────────────────────────────────────────────────

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Convenience: returns binary mask (0/1) without grad."""
        with torch.no_grad():
            logits = self.forward(x)
            return (torch.sigmoid(logits) > threshold).float()

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
