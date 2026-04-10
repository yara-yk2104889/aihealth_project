"""
models/dual_encoder_unet.py — Dual-encoder U-Net with late fusion.

Architecture
------------
Each modality (DWI, ADC) has its own independent encoder.
Skip connections from both encoders are concatenated at each decoder stage.
The bottleneck fuses both modality streams by concatenation + conv projection.

                DWI [B,1,H,W]      ADC [B,1,H,W]
                     │                   │
              Enc_DWI_0            Enc_ADC_0    → skip_dwi_0, skip_adc_0
                     │                   │
              Enc_DWI_1            Enc_ADC_1    → skip_dwi_1, skip_adc_1
                     │                   │
              Enc_DWI_2            Enc_ADC_2    → skip_dwi_2, skip_adc_2
                     │                   │
              Enc_DWI_3            Enc_ADC_3    → skip_dwi_3, skip_adc_3
                     │                   │
                     └────── cat ─────────┘
                               │
                          Bottleneck (fuse)
                               │
                    Dec3 ← cat(skip_dwi_3, skip_adc_3)
                               │
                    Dec2 ← cat(skip_dwi_2, skip_adc_2)
                               │
                    Dec1 ← cat(skip_dwi_1, skip_adc_1)
                               │
                    Dec0 ← cat(skip_dwi_0, skip_adc_0)
                               │
                        Output [B,1,H,W]
"""

from typing import Tuple
import torch
import torch.nn as nn

from .blocks import DoubleConv, EncoderBlock, DecoderBlock, Bottleneck


class DualEncoderUNet(nn.Module):

    def __init__(
        self,
        out_channels: int = 1,
        features: Tuple[int, ...] = (32, 64, 128, 256),
        bilinear: bool = True,
    ):
        super().__init__()
        self.features = features

        # ── Two independent encoders (one per modality) ───────────────────
        # Each takes a single-channel input
        encoder_channels = [1] + list(features)
        self.dwi_encoders = nn.ModuleList([
            EncoderBlock(encoder_channels[i], encoder_channels[i + 1])
            for i in range(len(features))
        ])
        self.adc_encoders = nn.ModuleList([
            EncoderBlock(encoder_channels[i], encoder_channels[i + 1])
            for i in range(len(features))
        ])

        # ── Bottleneck — fuses both streams ───────────────────────────────
        # Input: cat(dwi_bottleneck, adc_bottleneck) = features[-1] * 2 channels
        # Output: features[-1] * 2 channels
        bn_in  = features[-1] * 2   # from concatenating both encoder outputs
        bn_out = features[-1] * 2
        self.bottleneck = Bottleneck(bn_in, bn_out)

        # ── Decoder — skip connections are cat(dwi_skip, adc_skip) ────────
        # Each skip has features[i] channels from each encoder → features[i]*2 total
        decoder_out_ch      = list(reversed(features))
        decoder_in_channels = [bn_out] + decoder_out_ch[:-1]   # output of prev decoder stage
        decoder_skip_ch     = [f * 2 for f in reversed(features)]   # both skips cat'd

        self.decoders = nn.ModuleList([
            DecoderBlock(
                in_channels   = decoder_in_channels[i],
                skip_channels = decoder_skip_ch[i],
                out_channels  = decoder_out_ch[i],
                bilinear      = bilinear,
            )
            for i in range(len(features))
        ])

        # ── Output ────────────────────────────────────────────────────────
        self.out_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : FloatTensor [B, 2, H, W]
            Channel 0 = DWI, Channel 1 = ADC (same format as baseline UNet)

        Returns
        -------
        logits : FloatTensor [B, 1, H, W]
        """
        dwi = x[:, 0:1, :, :]   # [B, 1, H, W]
        adc = x[:, 1:2, :, :]   # [B, 1, H, W]

        # Encode each modality independently
        dwi_skips, adc_skips = [], []

        for dwi_enc, adc_enc in zip(self.dwi_encoders, self.adc_encoders):
            dwi_skip, dwi = dwi_enc(dwi)
            adc_skip, adc = adc_enc(adc)
            dwi_skips.append(dwi_skip)
            adc_skips.append(adc_skip)

        # Fuse at bottleneck by concatenation
        x = self.bottleneck(torch.cat([dwi, adc], dim=1))

        # Decode with fused skip connections
        for dec, dwi_skip, adc_skip in zip(
            self.decoders, reversed(dwi_skips), reversed(adc_skips)
        ):
            fused_skip = torch.cat([dwi_skip, adc_skip], dim=1)
            x = dec(x, fused_skip)

        return self.out_conv(x)

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        with torch.no_grad():
            return (torch.sigmoid(self.forward(x)) > threshold).float()

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
