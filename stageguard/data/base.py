"""Abstract base class for sleep datasets."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from torch.utils.data import Dataset


class BaseSleepDataset(ABC, Dataset):
    """Base class for all sleep-staging datasets.

    Subclasses should set ``DOWNLOAD_URL`` and implement ``__len__``,
    ``__getitem__``, and ``_load_data``.

    Data is expected to be pre-downloaded. Use ``download_instructions()``
    to print how to obtain the data.
    """

    DOWNLOAD_URL: Optional[str] = None
    DATASET_NAME: str = "Unknown"

    def __init__(
        self,
        data_dir: str | Path,
        sequence_length: int = 100,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.sequence_length = sequence_length
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Data directory not found: {self.data_dir}\n"
                f"{self.download_instructions()}"
            )
        self.signals, self.labels = self._load_data()

    @abstractmethod
    def _load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load signals and labels from disk.

        Returns:
            signals: Array of signal data.
            labels: Array of integer stage labels.
        """
        ...

    def __len__(self) -> int:
        n_epochs = len(self.labels)
        return max(0, n_epochs - self.sequence_length + 1)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        end = idx + self.sequence_length
        return self.signals[idx:end], self.labels[idx:end]

    @classmethod
    def download_instructions(cls) -> str:
        """Return instructions for downloading the dataset."""
        url = cls.DOWNLOAD_URL or "URL not available"
        return (
            f"Dataset: {cls.DATASET_NAME}\n"
            f"Download from: {url}\n"
            f"Place the data in your chosen data_dir and pass it to the constructor."
        )
