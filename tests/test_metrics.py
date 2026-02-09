"""Tests for evaluation metrics."""

import numpy as np

from stageguard.metrics import (
    classification_metrics,
    fragmentation_index,
    sleep_architecture,
    transition_violation_rate,
)


class TestTransitionViolationRate:
    def test_no_violations(self):
        preds = np.array([0, 0, 1, 1, 2, 2])
        tvr = transition_violation_rate(preds, [(0, 2), (2, 0)])
        assert tvr == 0.0

    def test_known_violations(self):
        # Transitions: 0→2, 2→1, 1→0
        preds = np.array([0, 2, 1, 0])
        tvr = transition_violation_rate(preds, [(0, 2), (2, 0)])
        # 1 violation (0→2) out of 3 transitions
        assert abs(tvr - 1 / 3) < 1e-10

    def test_all_violations(self):
        preds = np.array([0, 2, 0, 2])
        tvr = transition_violation_rate(preds, [(0, 2), (2, 0)])
        assert tvr == 1.0

    def test_empty_rare(self):
        preds = np.array([0, 1, 2, 0])
        assert transition_violation_rate(preds, []) == 0.0

    def test_single_element(self):
        assert transition_violation_rate(np.array([0]), [(0, 1)]) == 0.0


class TestFragmentationIndex:
    def test_constant(self):
        preds = np.array([0, 0, 0, 0])
        assert fragmentation_index(preds) == 0.0

    def test_alternating(self):
        preds = np.array([0, 1, 0, 1])
        assert fragmentation_index(preds) == 1.0

    def test_single_transition(self):
        preds = np.array([0, 0, 1, 1])
        assert abs(fragmentation_index(preds) - 1 / 3) < 1e-10


class TestClassificationMetrics:
    def test_perfect(self):
        y = np.array([0, 1, 2, 0, 1, 2])
        m = classification_metrics(y, y)
        assert m["accuracy"] == 1.0
        assert m["kappa"] == 1.0
        assert m["macro_f1"] == 1.0

    def test_with_stage_names(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 0, 0, 1])
        m = classification_metrics(y_true, y_pred, stage_names=["Wake", "Sleep"])
        assert "f1_Wake" in m
        assert "f1_Sleep" in m

    def test_accuracy_range(self):
        y_true = np.array([0, 1, 2, 0, 1])
        y_pred = np.array([0, 0, 2, 1, 1])
        m = classification_metrics(y_true, y_pred)
        assert 0.0 <= m["accuracy"] <= 1.0
        assert -1.0 <= m["kappa"] <= 1.0


class TestSleepArchitecture:
    def test_all_wake(self):
        preds = np.array([0, 0, 0, 0])
        stats = sleep_architecture(preds, epoch_sec=30.0, wake_label=0)
        assert stats["tst_min"] == 0.0
        assert stats["sleep_efficiency"] == 0.0

    def test_all_sleep(self):
        preds = np.array([1, 1, 1, 1])
        stats = sleep_architecture(preds, epoch_sec=30.0, wake_label=0)
        assert stats["tst_min"] == 2.0  # 4 * 30s = 120s = 2 min
        assert stats["sleep_efficiency"] == 1.0
        assert stats["waso_min"] == 0.0

    def test_awakenings(self):
        preds = np.array([1, 0, 1, 0, 1])
        stats = sleep_architecture(preds, epoch_sec=30.0, wake_label=0)
        assert stats["awakenings"] == 2.0

    def test_waso(self):
        # Sleep onset at t=0, wake at t=2, sleep at t=3
        preds = np.array([1, 1, 0, 1, 1])
        stats = sleep_architecture(preds, epoch_sec=60.0, wake_label=0)
        assert stats["waso_min"] == 1.0  # 1 epoch * 60s = 1 min
