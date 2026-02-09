"""StageGuardWrapper: ties backbone, loss, and decoder together."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModalityConfig
from .decoder import SemiMarkovDecoder
from .losses import stageguard_loss


class StageGuardWrapper(nn.Module):
    """Wraps any backbone with StageGuard loss and constrained decoding.

    Usage::

        backbone = AccuSleepBackbone(num_classes=3)
        config = ModalityConfig.from_yaml("configs/mouse_eeg.yaml")
        model = StageGuardWrapper(backbone, config)

        # Training
        loss, details = model.training_step(x, targets)
        loss.backward()

        # Inference
        predictions = model.predict(x)

    Args:
        backbone: Any nn.Module whose forward returns (B, T, C) logits.
        config: Modality configuration with constraint parameters.
    """

    def __init__(self, backbone: nn.Module, config: ModalityConfig) -> None:
        super().__init__()
        self.backbone = backbone
        self.config = config
        self.decoder = SemiMarkovDecoder(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the backbone.

        Args:
            x: Input tensor (shape depends on backbone).

        Returns:
            logits: (B, T, C) class logits.
        """
        return self.backbone(x)

    def training_step(
        self, x: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute StageGuard loss for a training batch.

        Args:
            x: Input tensor.
            targets: (B, T) integer class labels.

        Returns:
            loss: Scalar combined loss.
            details: Dict with 'ce_loss' and 'trans_loss'.
        """
        logits = self.forward(x)
        return stageguard_loss(
            logits,
            targets,
            rare_transitions=self.config.rare_transitions,
            lambda_trans=self.config.lambda_trans,
        )

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        sqi_scores: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Run inference with constrained decoding.

        Args:
            x: Input tensor.
            sqi_scores: (B, T) optional signal quality scores.

        Returns:
            stages: (B, T) integer stage predictions.
        """
        self.eval()
        logits = self.forward(x)
        log_probs = F.log_softmax(logits, dim=-1).cpu().numpy()
        return self.decoder.decode_batch(log_probs, sqi_scores)
