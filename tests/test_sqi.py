"""Tests for signal quality index functions."""

import numpy as np
import pytest

from stageguard.sqi import (
    SQI_REGISTRY,
    acceleration_variance,
    compute_sqi,
    rr_interval_quality,
    signal_amplitude,
    spectral_entropy,
)


class TestSpectralEntropy:
    def test_output_range(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(512)
        val = spectral_entropy(x, fs=256.0)
        assert 0.0 <= val <= 1.0

    def test_pure_sine_high_quality(self):
        t = np.linspace(0, 1, 256, endpoint=False)
        x = np.sin(2 * np.pi * 10 * t)  # Pure 10 Hz sine
        val = spectral_entropy(x, fs=256.0)
        # Concentrated spectrum → low entropy → high quality
        assert val > 0.3


class TestAccelerationVariance:
    def test_output_range(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(100)
        val = acceleration_variance(x)
        assert 0.0 <= val <= 1.0

    def test_zero_input(self):
        x = np.zeros(100)
        val = acceleration_variance(x)
        assert 0.0 <= val <= 1.0


class TestRRIntervalQuality:
    def test_output_range(self):
        rng = np.random.default_rng(42)
        x = 0.8 + rng.standard_normal(50) * 0.05  # Regular RR intervals
        val = rr_interval_quality(x)
        assert 0.0 <= val <= 1.0

    def test_regular_high_quality(self):
        x = np.ones(50) * 0.8  # Perfectly regular
        val = rr_interval_quality(x)
        assert val == 1.0

    def test_short_input(self):
        assert rr_interval_quality(np.array([1.0, 2.0])) == 0.0


class TestSignalAmplitude:
    def test_output_range(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(100)
        val = signal_amplitude(x)
        assert 0.0 <= val <= 1.0

    def test_zero_signal(self):
        assert signal_amplitude(np.zeros(100)) == 0.0

    def test_large_amplitude(self):
        x = np.ones(100) * 100.0
        val = signal_amplitude(x)
        assert val > 0.9


class TestDispatcher:
    def test_all_methods_registered(self):
        expected = {
            "spectral_entropy",
            "acceleration_variance",
            "rr_interval_quality",
            "signal_amplitude",
        }
        assert set(SQI_REGISTRY.keys()) == expected

    def test_dispatch(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(256)
        for method in SQI_REGISTRY:
            val = compute_sqi(x, method)
            assert 0.0 <= val <= 1.0

    def test_unknown_method(self):
        with pytest.raises(ValueError, match="Unknown SQI method"):
            compute_sqi(np.zeros(10), "nonexistent")
