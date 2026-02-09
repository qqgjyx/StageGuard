"""Tests for StageGuard loss functions."""

import torch

from stageguard.losses import SoftTransitionPenalty, stageguard_loss


class TestSoftTransitionPenalty:
    def test_output_is_scalar(self, random_logits):
        penalty = SoftTransitionPenalty([(0, 2), (2, 0)])
        loss = penalty(random_logits)
        assert loss.dim() == 0

    def test_nonnegative(self, random_logits):
        penalty = SoftTransitionPenalty([(0, 2), (2, 0)])
        loss = penalty(random_logits)
        assert loss.item() >= 0.0

    def test_zero_when_no_rare_transitions(self, random_logits):
        penalty = SoftTransitionPenalty([])
        loss = penalty(random_logits)
        assert loss.item() == 0.0

    def test_gradient_flows(self, random_logits):
        logits = random_logits.clone().requires_grad_(True)
        penalty = SoftTransitionPenalty([(0, 2), (2, 0)])
        loss = penalty(logits)
        loss.backward()
        assert logits.grad is not None
        assert logits.grad.shape == logits.shape

    def test_single_timestep_returns_zero(self):
        logits = torch.randn(1, 1, 3)
        penalty = SoftTransitionPenalty([(0, 1)])
        assert penalty(logits).item() == 0.0


class TestStageGuardLoss:
    def test_output_shape(self, random_logits, random_targets):
        total, details = stageguard_loss(
            random_logits, random_targets, [(0, 2), (2, 0)]
        )
        assert total.dim() == 0
        assert "ce_loss" in details
        assert "trans_loss" in details

    def test_gradient_flows(self, random_logits, random_targets):
        logits = random_logits.clone().requires_grad_(True)
        total, _ = stageguard_loss(logits, random_targets, [(0, 2), (2, 0)])
        total.backward()
        assert logits.grad is not None

    def test_lambda_zero_equals_ce(self, random_logits, random_targets):
        total, details = stageguard_loss(
            random_logits, random_targets, [(0, 2)], lambda_trans=0.0
        )
        torch.testing.assert_close(total, details["ce_loss"])

    def test_higher_lambda_higher_loss(self, random_logits, random_targets):
        loss_low, _ = stageguard_loss(
            random_logits, random_targets, [(0, 2)], lambda_trans=0.1
        )
        loss_high, _ = stageguard_loss(
            random_logits, random_targets, [(0, 2)], lambda_trans=10.0
        )
        assert loss_high.item() >= loss_low.item()
