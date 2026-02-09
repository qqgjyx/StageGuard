"""Evaluation metrics for sleep staging.

Includes physiological-constraint metrics (TVR, FI) and standard
classification metrics (accuracy, kappa, F1).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score


def transition_violation_rate(
    predictions: np.ndarray,
    rare_transitions: List[Tuple[int, int]],
) -> float:
    """Fraction of consecutive-epoch transitions that are physiologically rare.

    TVR = |{t : (ŷ_{t-1}, ŷ_t) ∈ R}| / (T - 1)

    Args:
        predictions: (T,) integer predictions.
        rare_transitions: List of (source, target) rare pairs.

    Returns:
        TVR in [0, 1].
    """
    if len(predictions) < 2 or not rare_transitions:
        return 0.0
    rare_set = set(rare_transitions)
    violations = sum(
        1
        for t in range(1, len(predictions))
        if (predictions[t - 1], predictions[t]) in rare_set
    )
    return violations / (len(predictions) - 1)


def fragmentation_index(predictions: np.ndarray) -> float:
    """Number of stage transitions per hour of recording.

    FI = n_transitions / (T * epoch_duration_hours). We report raw count
    normalized by number of epochs for epoch-agnostic comparison.

    Args:
        predictions: (T,) integer predictions.

    Returns:
        Transitions per epoch (multiply by 3600/epoch_sec for per-hour).
    """
    if len(predictions) < 2:
        return 0.0
    transitions = np.sum(predictions[1:] != predictions[:-1])
    return float(transitions / (len(predictions) - 1))


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    stage_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Standard classification metrics.

    Args:
        y_true: (T,) ground-truth labels.
        y_pred: (T,) predicted labels.
        stage_names: Optional names for reporting.

    Returns:
        Dict with accuracy, kappa, macro_f1, and per-class F1.
    """
    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }

    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    if stage_names and len(stage_names) == len(per_class_f1):
        for name, val in zip(stage_names, per_class_f1):
            metrics[f"f1_{name}"] = float(val)
    else:
        for i, val in enumerate(per_class_f1):
            metrics[f"f1_class_{i}"] = float(val)

    return metrics


def sleep_architecture(
    predictions: np.ndarray,
    epoch_sec: float = 30.0,
    wake_label: int = 0,
    stage_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Compute sleep architecture statistics.

    Args:
        predictions: (T,) integer predictions.
        epoch_sec: Duration of each epoch in seconds.
        wake_label: Integer label for the Wake stage.
        stage_names: Optional stage names for per-stage statistics.

    Returns:
        Dict with TST, SE, WASO, bout durations, and awakenings.
    """
    T = len(predictions)
    total_time_min = T * epoch_sec / 60.0

    is_sleep = predictions != wake_label
    total_sleep_epochs = int(np.sum(is_sleep))
    tst_min = total_sleep_epochs * epoch_sec / 60.0

    # Sleep efficiency: TST / total recording time
    se = tst_min / total_time_min if total_time_min > 0 else 0.0

    # Sleep onset: first sleep epoch
    sleep_indices = np.where(is_sleep)[0]
    if len(sleep_indices) > 0:
        sleep_onset = sleep_indices[0]
        sleep_offset = sleep_indices[-1]
        # WASO: wake time after sleep onset, before final awakening
        period = predictions[sleep_onset : sleep_offset + 1]
        waso_epochs = int(np.sum(period == wake_label))
        waso_min = waso_epochs * epoch_sec / 60.0
    else:
        waso_min = 0.0

    # Awakenings: transitions from sleep to wake
    awakenings = 0
    for t in range(1, T):
        if predictions[t] == wake_label and predictions[t - 1] != wake_label:
            awakenings += 1

    # Bout durations per stage
    bout_durations: Dict[int, List[float]] = {}
    i = 0
    while i < T:
        j = i + 1
        while j < T and predictions[j] == predictions[i]:
            j += 1
        stage = int(predictions[i])
        dur_min = (j - i) * epoch_sec / 60.0
        bout_durations.setdefault(stage, []).append(dur_min)
        i = j

    stats: Dict[str, float] = {
        "tst_min": tst_min,
        "sleep_efficiency": se,
        "waso_min": waso_min,
        "awakenings": float(awakenings),
        "total_time_min": total_time_min,
    }

    # Mean bout duration per stage
    names = stage_names or [str(s) for s in sorted(bout_durations.keys())]
    for stage_idx, bouts in bout_durations.items():
        label = names[stage_idx] if stage_idx < len(names) else str(stage_idx)
        stats[f"mean_bout_{label}_min"] = float(np.mean(bouts))

    return stats
