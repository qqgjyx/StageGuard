"""Signal Quality Index (SQI) functions for different modalities.

Each function takes a 1-D signal segment and returns a scalar in [0, 1],
where higher values indicate better signal quality.
"""

from __future__ import annotations

from typing import Callable, Dict

import numpy as np
from scipy import signal as sig


def spectral_entropy(x: np.ndarray, fs: float = 256.0, **kwargs) -> float:
    """SQI for EEG: normalized spectral entropy.

    Low entropy → dominated by a few frequencies (good rhythmic EEG).
    High entropy → flat spectrum (noisy / artifact-heavy).
    We invert so that high quality → high SQI.
    """
    freqs, psd = sig.welch(x, fs=fs, nperseg=min(len(x), 256))
    psd = psd / (psd.sum() + 1e-12)
    entropy = -np.sum(psd * np.log(psd + 1e-12))
    max_entropy = np.log(len(psd) + 1e-12)
    normalized = entropy / (max_entropy + 1e-12)
    return float(1.0 - normalized)  # Invert: low entropy = good


def acceleration_variance(x: np.ndarray, **kwargs) -> float:
    """SQI for actigraphy: variance-based quality.

    Very low or very high variance suggests sensor detachment or artifact.
    """
    var = np.var(x)
    # Sigmoid-like mapping: moderate variance → high quality
    quality = float(2.0 / (1.0 + np.exp(-0.5 * var)) - 1.0)
    return np.clip(quality, 0.0, 1.0)


def rr_interval_quality(x: np.ndarray, **kwargs) -> float:
    """SQI for cardiorespiratory: RR-interval regularity.

    Coefficient of variation of successive differences (low = good).
    """
    if len(x) < 3:
        return 0.0
    diffs = np.abs(np.diff(x))
    mean_diff = np.mean(diffs)
    if mean_diff < 1e-12:
        return 1.0
    cv = np.std(diffs) / mean_diff
    # Lower CV → better quality
    quality = float(np.exp(-cv))
    return np.clip(quality, 0.0, 1.0)


def signal_amplitude(x: np.ndarray, **kwargs) -> float:
    """SQI for bioradar: amplitude-based quality.

    Very low amplitude suggests no subject or signal dropout.
    """
    amp = np.max(np.abs(x))
    if amp < 1e-12:
        return 0.0
    # Saturating mapping
    quality = float(1.0 - np.exp(-amp))
    return np.clip(quality, 0.0, 1.0)


# --- Dispatcher ---

SQI_REGISTRY: Dict[str, Callable] = {
    "spectral_entropy": spectral_entropy,
    "acceleration_variance": acceleration_variance,
    "rr_interval_quality": rr_interval_quality,
    "signal_amplitude": signal_amplitude,
}


def compute_sqi(x: np.ndarray, method: str, **kwargs) -> float:
    """Dispatch to the appropriate SQI function.

    Args:
        x: 1-D signal segment.
        method: Name of the SQI method (must be in SQI_REGISTRY).

    Returns:
        Scalar quality score in [0, 1].
    """
    if method not in SQI_REGISTRY:
        raise ValueError(
            f"Unknown SQI method '{method}'. "
            f"Available: {list(SQI_REGISTRY.keys())}"
        )
    return SQI_REGISTRY[method](x, **kwargs)
