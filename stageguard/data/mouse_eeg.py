"""AccuSleep mouse EEG/EMG dataset loader."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

from .base import BaseSleepDataset


class AccuSleepDataset(BaseSleepDataset):
    """Loader for the AccuSleep mouse EEG/EMG dataset.

    Expects pre-downloaded .mat or .npz files with EEG signals and
    3-class labels (Wake=0, NREM=1, REM=2) at 4-second epoch resolution.

    Download: https://zenodo.org/records/4079563
    """

    DOWNLOAD_URL = "https://zenodo.org/records/4079563"
    DATASET_NAME = "AccuSleep"

    def __init__(
        self,
        data_dir: str | Path,
        sequence_length: int = 100,
        fs: float = 256.0,
        epoch_sec: float = 4.0,
    ) -> None:
        self.fs = fs
        self.epoch_sec = epoch_sec
        self.samples_per_epoch = int(fs * epoch_sec)
        super().__init__(data_dir, sequence_length)

    def _load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load all .npz files from data_dir.

        Expected format per file:
            - 'eeg': (n_epochs, samples_per_epoch) EEG signal
            - 'labels': (n_epochs,) integer labels
        """
        signals_list = []
        labels_list = []
        for fpath in sorted(self.data_dir.glob("*.npz")):
            data = np.load(fpath)
            signals_list.append(data["eeg"])
            labels_list.append(data["labels"])

        if not signals_list:
            raise FileNotFoundError(
                f"No .npz files found in {self.data_dir}. "
                f"See: {self.DOWNLOAD_URL}"
            )
        signals = np.concatenate(signals_list, axis=0)
        labels = np.concatenate(labels_list, axis=0)
        # Reshape to (n_epochs, 1, samples_per_epoch) for conv input
        if signals.ndim == 2:
            signals = signals[:, np.newaxis, :]
        return signals, labels
