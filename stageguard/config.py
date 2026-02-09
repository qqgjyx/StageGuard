"""Modality configuration for StageGuard."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import yaml


@dataclass
class ModalityConfig:
    """Configuration for a sleep-staging modality.

    Each modality (EEG, actigraphy, cardiorespiratory, bioradar) has its own
    set of sleep stages, physiological constraints, and signal quality settings.
    """

    # --- Stage definitions ---
    stage_names: List[str]
    num_classes: int

    # --- Transition penalty (L_trans) ---
    rare_transitions: List[Tuple[int, int]]
    lambda_trans: float = 1.0

    # --- Semi-Markov decoder ---
    d_max: int = 30
    epsilon: float = 5.0
    gamma: float = 2.0
    k: int = 5
    d_min: List[int] = field(default_factory=list)

    # --- Signal quality ---
    sqi_method: str = "spectral_entropy"
    sqi_threshold: float = 0.5

    # --- Epoch ---
    epoch_sec: float = 30.0

    # --- Dataset info (optional) ---
    dataset_name: Optional[str] = None
    dataset_url: Optional[str] = None

    def __post_init__(self) -> None:
        if self.num_classes != len(self.stage_names):
            raise ValueError(
                f"num_classes ({self.num_classes}) != "
                f"len(stage_names) ({len(self.stage_names)})"
            )
        if self.d_min and len(self.d_min) != self.num_classes:
            raise ValueError(
                f"len(d_min) ({len(self.d_min)}) != "
                f"num_classes ({self.num_classes})"
            )
        # Convert rare_transitions from list-of-lists to list-of-tuples
        self.rare_transitions = [tuple(t) for t in self.rare_transitions]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ModalityConfig":
        """Load configuration from a YAML file."""
        path = Path(path)
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)
