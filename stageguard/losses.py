"""StageGuard transition penalty loss.

Implements the soft transition penalty L_trans that discourages physiologically
rare stage transitions during training.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftTransitionPenalty(nn.Module):
    """Penalizes rare sleep-stage transitions in the softmax probability space.

    L_trans = (1 / (T-1)) * Σ_t Σ_{(s,s')∈R} p(y_{t-1}=s) * p(y_t=s')

    where R is the set of rare transitions and p = softmax(logits).
    """

    def __init__(self, rare_transitions: List[Tuple[int, int]]) -> None:
        super().__init__()
        self.rare_transitions = rare_transitions

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute the soft transition penalty.

        Args:
            logits: (B, T, C) raw logits from the backbone.

        Returns:
            Scalar penalty averaged over batch and time.
        """
        if not self.rare_transitions:
            return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

        probs = F.softmax(logits, dim=-1)  # (B, T, C)
        T = probs.shape[1]
        if T < 2:
            return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

        penalty = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
        for s, s_prime in self.rare_transitions:
            # p(y_{t-1}=s) * p(y_t=s')
            penalty = penalty + (probs[:, :-1, s] * probs[:, 1:, s_prime]).sum()

        # Average over batch and (T-1) transitions
        B = logits.shape[0]
        penalty = penalty / (B * (T - 1))
        return penalty


def stageguard_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    rare_transitions: List[Tuple[int, int]],
    lambda_trans: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Combined StageGuard loss: L_CE + λ * L_trans.

    Args:
        logits: (B, T, C) raw logits.
        targets: (B, T) integer class labels.
        rare_transitions: List of (source, target) rare transition pairs.
        lambda_trans: Weight for the transition penalty.

    Returns:
        total_loss: Scalar combined loss.
        details: Dict with 'ce_loss' and 'trans_loss' components.
    """
    B, T, C = logits.shape
    ce_loss = F.cross_entropy(logits.reshape(-1, C), targets.reshape(-1))

    trans_penalty = SoftTransitionPenalty(rare_transitions)
    trans_loss = trans_penalty(logits)

    total_loss = ce_loss + lambda_trans * trans_loss
    return total_loss, {"ce_loss": ce_loss, "trans_loss": trans_loss}
