"""Sleep-Accel wrist actigraphy dataset loader."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from .base import BaseSleepDataset


class SleepAccelDataset(BaseSleepDataset):
    """Loader for the Sleep-Accel wrist actigraphy dataset.

    Expects pre-downloaded CSV files with accelerometer data and
    2-class labels (Wake=0, Sleep=1) at 30-second epoch resolution.

    Download: https://physionet.org/content/sleep-accel/1.0.0/
    """

    DOWNLOAD_URL = "https://physionet.org/content/sleep-accel/1.0.0/"
    DATASET_NAME = "Sleep-Accel"

    def __init__(
        self,
        data_dir: str | Path,
        sequence_length: int = 100,
    ) -> None:
        super().__init__(data_dir, sequence_length)

    def _load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load CSV files: each with columns [timestamp, x, y, z, label].

        Expects one CSV per subject with 30-second epoch features.
        """
        signals_list = []
        labels_list = []
        for fpath in sorted(self.data_dir.glob("*.csv")):
            df = pd.read_csv(fpath)
            # Expect columns: x, y, z (acceleration), label
            accel = df[["x", "y", "z"]].values  # (n_epochs, 3)
            labels = df["label"].values.astype(np.int64)
            signals_list.append(accel)
            labels_list.append(labels)

        if not signals_list:
            raise FileNotFoundError(
                f"No .csv files found in {self.data_dir}. "
                f"See: {self.DOWNLOAD_URL}"
            )
        signals = np.concatenate(signals_list, axis=0)
        labels = np.concatenate(labels_list, axis=0)
        # Reshape to (n_epochs, 3, 1) — channel-first, single sample per epoch
        signals = signals[:, :, np.newaxis]
        return signals, labels
