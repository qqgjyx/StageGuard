"""SHHS cardiorespiratory dataset loader."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import h5py
import numpy as np

from .base import BaseSleepDataset


class SHHSDataset(BaseSleepDataset):
    """Loader for the SHHS cardiorespiratory dataset.

    Expects pre-downloaded HDF5 files with heart-rate and respiratory
    features and 3-class labels (Wake=0, Light=1, Deep=2) at 30-second
    epoch resolution.

    Download: https://sleepdata.org/datasets/shhs
    (Requires data use agreement)
    """

    DOWNLOAD_URL = "https://sleepdata.org/datasets/shhs"
    DATASET_NAME = "SHHS"

    def __init__(
        self,
        data_dir: str | Path,
        sequence_length: int = 100,
    ) -> None:
        super().__init__(data_dir, sequence_length)

    def _load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load HDF5 files with 'features' and 'labels' datasets."""
        signals_list = []
        labels_list = []
        for fpath in sorted(self.data_dir.glob("*.h5")):
            with h5py.File(fpath, "r") as f:
                signals_list.append(f["features"][:])
                labels_list.append(f["labels"][:])

        if not signals_list:
            raise FileNotFoundError(
                f"No .h5 files found in {self.data_dir}. "
                f"See: {self.DOWNLOAD_URL}"
            )
        signals = np.concatenate(signals_list, axis=0)
        labels = np.concatenate(labels_list, axis=0).astype(np.int64)
        # Ensure channel-first: (n_epochs, n_features, 1)
        if signals.ndim == 2:
            signals = signals[:, :, np.newaxis]
        return signals, labels
