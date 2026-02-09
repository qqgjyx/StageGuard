"""End-to-end demo with synthetic data.

Shows: forward pass → loss computation → constrained decoding → metrics.
"""

import numpy as np
import torch

from stageguard.backbones import get_backbone
from stageguard.config import ModalityConfig
from stageguard.metrics import (
    classification_metrics,
    fragmentation_index,
    sleep_architecture,
    transition_violation_rate,
)
from stageguard.wrapper import StageGuardWrapper


def main():
    # --- Configuration ---
    config = ModalityConfig.from_yaml("configs/mouse_eeg.yaml")
    print(f"Config: {config.dataset_name}, {config.num_classes} classes")
    print(f"Stages: {config.stage_names}")
    print(f"Rare transitions: {config.rare_transitions}")

    # --- Backbone + Wrapper ---
    backbone = get_backbone("accusleep", num_classes=config.num_classes, in_channels=1)
    model = StageGuardWrapper(backbone, config)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # --- Synthetic data ---
    B, T, samples = 4, 50, 128
    x = torch.randn(B, T, 1, samples)
    targets = torch.randint(0, config.num_classes, (B, T))

    # --- Training step ---
    loss, details = model.training_step(x, targets)
    print(f"\nTraining loss: {loss.item():.4f}")
    print(f"  CE loss:    {details['ce_loss'].item():.4f}")
    print(f"  Trans loss: {details['trans_loss'].item():.4f}")

    # --- Inference ---
    predictions = model.predict(x)
    print(f"\nPredictions shape: {predictions.shape}")
    print(f"Sample predictions: {predictions[0, :20]}")

    # --- Metrics ---
    y_true = targets[0].numpy()
    y_pred = predictions[0]

    tvr = transition_violation_rate(y_pred, config.rare_transitions)
    fi = fragmentation_index(y_pred)
    cls_metrics = classification_metrics(y_true, y_pred, config.stage_names)
    arch = sleep_architecture(y_pred, epoch_sec=config.epoch_sec, stage_names=config.stage_names)

    print(f"\n--- Metrics ---")
    print(f"TVR:  {tvr:.4f}")
    print(f"FI:   {fi:.4f}")
    print(f"Acc:  {cls_metrics['accuracy']:.4f}")
    print(f"Kappa: {cls_metrics['kappa']:.4f}")
    print(f"F1:   {cls_metrics['macro_f1']:.4f}")
    print(f"\n--- Sleep Architecture ---")
    for k, v in arch.items():
        print(f"  {k}: {v:.2f}")

    print("\nDemo complete.")


if __name__ == "__main__":
    main()
