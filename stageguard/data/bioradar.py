"""SLEEPBRL bioradar dataset loader."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

from .base import BaseSleepDataset


class SleepBRLDataset(BaseSleepDataset):
    """Loader for the SLEEPBRL contactless bioradar dataset.

    Expects pre-downloaded .npz files with bioradar signal features and
    3-class labels (Wake=0, Light=1, Deep=2) at 30-second epoch resolution.
    """

    DOWNLOAD_URL = None
    DATASET_NAME = "SLEEPBRL"

    def __init__(
        self,
        data_dir: str | Path,
        sequence_length: int = 100,
    ) -> None:
        super().__init__(data_dir, sequence_length)

    def _load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load .npz files with 'signals' and 'labels' arrays."""
        signals_list = []
        labels_list = []
        for fpath in sorted(self.data_dir.glob("*.npz")):
            data = np.load(fpath)
            signals_list.append(data["signals"])
            labels_list.append(data["labels"])

        if not signals_list:
            raise FileNotFoundError(
                f"No .npz files found in {self.data_dir}. "
                f"Contact the authors for SLEEPBRL data access."
            )
        signals = np.concatenate(signals_list, axis=0)
        labels = np.concatenate(labels_list, axis=0).astype(np.int64)
        if signals.ndim == 2:
            signals = signals[:, :, np.newaxis]
        return signals, labels
