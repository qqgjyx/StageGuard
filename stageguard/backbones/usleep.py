"""U-Sleep backbone: U-Net encoder-decoder for sleep staging."""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import BackboneBase


class _EncoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=7, padding=3),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            nn.Conv1d(out_ch, out_ch, kernel_size=7, padding=3),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
        )
        self.pool = nn.MaxPool1d(2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.conv(x)
        return self.pool(feat), feat


class _DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose1d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv1d(out_ch * 2, out_ch, kernel_size=7, padding=3),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Handle size mismatch from odd-length inputs
        if x.shape[-1] != skip.shape[-1]:
            x = nn.functional.pad(x, (0, skip.shape[-1] - x.shape[-1]))
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class USleepBackbone(BackboneBase):
    """Simplified U-Sleep-style U-Net encoder-decoder.

    Expects input of shape (B, T, in_channels, epoch_samples).
    Outputs (B, T, num_classes) logits.
    """

    def __init__(
        self,
        num_classes: int = 3,
        in_channels: int = 1,
        base_filters: int = 16,
        depth: int = 3,
    ) -> None:
        super().__init__(num_classes)
        self.depth = depth

        # Encoder
        self.encoders = nn.ModuleList()
        ch = in_channels
        for i in range(depth):
            out_ch = base_filters * (2 ** i)
            self.encoders.append(_EncoderBlock(ch, out_ch))
            ch = out_ch

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv1d(ch, ch * 2, kernel_size=7, padding=3),
            nn.BatchNorm1d(ch * 2),
            nn.ReLU(),
        )
        ch = ch * 2

        # Decoder
        self.decoders = nn.ModuleList()
        for i in range(depth - 1, -1, -1):
            out_ch = base_filters * (2 ** i)
            self.decoders.append(_DecoderBlock(ch, out_ch))
            ch = out_ch

        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(ch, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, T, in_channels, epoch_samples) raw signal epochs.

        Returns:
            logits: (B, T, num_classes).
        """
        B, T = x.shape[:2]
        x = x.reshape(B * T, *x.shape[2:])  # (B*T, C_in, samples)

        # Encoder
        skips = []
        for enc in self.encoders:
            x, skip = enc(x)
            skips.append(skip)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        for dec, skip in zip(self.decoders, reversed(skips)):
            x = dec(x, skip)

        # Classify
        logits = self.classifier(x)  # (B*T, num_classes)
        return logits.reshape(B, T, self.num_classes)
