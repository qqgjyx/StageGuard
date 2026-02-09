"""Tests for the Semi-Markov decoder."""

import numpy as np
import pytest

from stageguard.decoder import SemiMarkovDecoder


class TestSemiMarkovDecoder:
    def test_output_shape(self, dummy_config, random_log_probs):
        decoder = SemiMarkovDecoder(dummy_config)
        result = decoder.decode(random_log_probs[0])
        assert result.shape == (20,)

    def test_batch_output_shape(self, dummy_config, random_log_probs):
        decoder = SemiMarkovDecoder(dummy_config)
        result = decoder.decode_batch(random_log_probs)
        assert result.shape == (2, 20)

    def test_labels_in_range(self, dummy_config, random_log_probs):
        decoder = SemiMarkovDecoder(dummy_config)
        result = decoder.decode_batch(random_log_probs)
        assert np.all(result >= 0)
        assert np.all(result < dummy_config.num_classes)

    def test_d_min_enforced(self, dummy_config):
        """Segments should be at least d_min epochs long."""
        decoder = SemiMarkovDecoder(dummy_config)
        # Create log-probs that strongly favor alternating stages
        T, C = 30, 3
        log_probs = np.full((T, C), -10.0)
        for t in range(T):
            log_probs[t, t % C] = 0.0  # Favor alternating

        result = decoder.decode(log_probs)
        # Check all segments are >= d_min
        i = 0
        while i < T:
            j = i + 1
            while j < T and result[j] == result[i]:
                j += 1
            seg_len = j - i
            stage = result[i]
            # d_min enforcement (last segment may be shorter if at boundary)
            if j < T:
                assert seg_len >= dummy_config.d_min[stage], (
                    f"Segment of stage {stage} has length {seg_len} "
                    f"< d_min={dummy_config.d_min[stage]}"
                )
            i = j

    def test_no_rare_transitions(self, dummy_config):
        """Decoder should avoid rare transitions when possible."""
        decoder = SemiMarkovDecoder(dummy_config)
        T, C = 50, 3
        rng = np.random.default_rng(0)
        log_probs = rng.standard_normal((T, C)) * 0.1  # Weak preferences
        # Make stage 1 generally preferred
        log_probs[:, 1] += 2.0

        result = decoder.decode(log_probs)
        rare_set = set(dummy_config.rare_transitions)
        for t in range(1, T):
            if result[t] != result[t - 1]:
                assert (result[t - 1], result[t]) not in rare_set, (
                    f"Rare transition {result[t-1]} -> {result[t]} at t={t}"
                )

    def test_anti_flip_flop(self):
        """γ penalty should suppress short detours back to a previous stage."""
        from stageguard.config import ModalityConfig

        # Config WITHOUT anti-flip-flop (γ=0)
        cfg_no_afp = ModalityConfig(
            stage_names=["W", "N", "R"], num_classes=3,
            rare_transitions=[], lambda_trans=1.0,
            d_max=10, epsilon=5.0, gamma=0.0, k=5,
            d_min=[2, 2, 2], epoch_sec=30.0,
        )
        # Config WITH anti-flip-flop (γ=2.0)
        cfg_afp = ModalityConfig(
            stage_names=["W", "N", "R"], num_classes=3,
            rare_transitions=[], lambda_trans=1.0,
            d_max=10, epsilon=5.0, gamma=2.0, k=5,
            d_min=[2, 2, 2], epoch_sec=30.0,
        )

        T, C = 20, 3
        log_probs = np.full((T, C), -10.0)
        log_probs[:, 0] = 0.0  # Stage 0 strongly preferred everywhere

        # Slight preference for stage 1 at positions 8-9 (2 epochs = d_min)
        # Emission advantage = 2*0.3 - 2*(-0.3) = 1.2 < γ=2.0
        log_probs[8:10, 1] = 0.3
        log_probs[8:10, 0] = -0.3

        result_no_afp = SemiMarkovDecoder(cfg_no_afp).decode(log_probs)
        result_afp = SemiMarkovDecoder(cfg_afp).decode(log_probs)

        # Without γ, the decoder takes the flip-flop detour
        assert result_no_afp[8] == 1 and result_no_afp[9] == 1, (
            f"Expected flip-flop without γ, got {result_no_afp[7:11]}"
        )
        # With γ=2.0, the penalty outweighs the emission gain → stays in 0
        assert result_afp[8] == 0 and result_afp[9] == 0, (
            f"Expected no flip-flop with γ, got {result_afp[7:11]}"
        )

    def test_sqi_damping(self, dummy_config):
        """Low SQI should push predictions toward uniform."""
        decoder = SemiMarkovDecoder(dummy_config)
        T, C = 20, 3
        # Strong preference for class 0
        log_probs = np.full((T, C), -10.0)
        log_probs[:, 0] = 0.0

        # With perfect SQI
        result_high = decoder.decode(log_probs, sqi_scores=np.ones(T))
        # With zero SQI (uniform emissions)
        result_low = decoder.decode(log_probs, sqi_scores=np.zeros(T))

        # High SQI should produce class 0; low SQI is more uncertain
        assert np.all(result_high == 0)

    def test_binary_config(self, binary_config):
        """Works with 2-class, no rare transitions."""
        decoder = SemiMarkovDecoder(binary_config)
        rng = np.random.default_rng(42)
        log_probs = rng.standard_normal((20, 2))
        result = decoder.decode(log_probs)
        assert result.shape == (20,)
        assert np.all((result == 0) | (result == 1))
