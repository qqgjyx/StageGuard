"""Abstract base class for sleep-staging backbones."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BackboneBase(ABC, nn.Module):
    """Base class for all sleep-staging backbones.

    Subclasses must implement ``forward`` returning logits of shape (B, T, C).
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor. Shape depends on the backbone.

        Returns:
            logits: (B, T, C) class logits per epoch.
        """
        ...
