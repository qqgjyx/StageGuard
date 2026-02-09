"""Semi-Markov decoder with augmented Viterbi for constrained sleep staging.

Implements physiological constraints via:
  1. Rare-transition penalties (ε) in the transition matrix.
  2. Minimum duration enforcement (d_min) through state augmentation.
  3. Anti-flip-flop penalty (γ, k) to discourage rapid stage alternation.

All computation is in NumPy (inference-only, no gradients needed).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .config import ModalityConfig


class SemiMarkovDecoder:
    """Augmented Viterbi decoder enforcing physiological sleep constraints.

    The augmented state space is S̃ = {(s_prev, s, d)} where:
      - s_prev: stage of the previous segment (for anti-flip-flop tracking)
      - s:      current stage, s ∈ {0..C-1}
      - d:      consecutive epochs in stage s, d ∈ {1..d_max}

    Constraints encoded in the transition matrix:
      - d_min: cannot leave state s before d >= d_min[s]
      - ε:     rare transitions (s, s') ∈ R incur a -ε penalty
      - γ:     returning to s_prev within d < k epochs incurs a -γ penalty

    Args:
        config: ModalityConfig with constraint parameters.
    """

    def __init__(self, config: ModalityConfig) -> None:
        self.num_classes = config.num_classes
        self.d_max = config.d_max
        self.epsilon = config.epsilon
        self.gamma = config.gamma
        self.k = config.k
        self.d_min = config.d_min if config.d_min else [1] * config.num_classes
        self.rare_transitions = set(config.rare_transitions)

        # Build augmented transition table
        self._build_transitions()

    def _aug_index(self, s_prev: int, s: int, d: int) -> int:
        """Compute flat index for augmented state (s_prev, s, d), d 1-based."""
        C, D = self.num_classes, self.d_max
        return s_prev * C * D + s * D + (d - 1)

    def _aug_stage(self, idx: int) -> int:
        """Extract current stage s from a flat augmented-state index."""
        C, D = self.num_classes, self.d_max
        return (idx % (C * D)) // D

    def _build_transitions(self) -> None:
        """Precompute valid transitions for efficient Viterbi.

        For each destination state, stores a list of (source_index, log_cost)
        pairs.  Only valid transitions are stored; all others are implicitly
        -∞ (forbidden).
        """
        C, D = self.num_classes, self.d_max
        self.n_aug = C * C * D
        self.valid_trans: list[list[tuple[int, float]]] = [
            [] for _ in range(self.n_aug)
        ]

        for s_prev in range(C):
            for s in range(C):
                for d in range(1, D + 1):
                    src = self._aug_index(s_prev, s, d)

                    # --- Stay in same state: (s_prev, s, d) → (s_prev, s, min(d+1, D)) ---
                    d_next = min(d + 1, D)
                    dst_stay = self._aug_index(s_prev, s, d_next)
                    self.valid_trans[dst_stay].append((src, 0.0))

                    # --- Switch to different state: (s_prev, s, d) → (s, s', 1) ---
                    if d < self.d_min[s]:
                        continue  # Cannot leave before d_min

                    for s_prime in range(C):
                        if s_prime == s:
                            continue
                        dst_switch = self._aug_index(s, s_prime, 1)
                        cost = 0.0

                        # Rare-transition penalty
                        if (s, s_prime) in self.rare_transitions:
                            cost -= self.epsilon

                        # Anti-flip-flop: returning to the previous stage
                        # after a short (<k epoch) segment incurs -γ
                        if s_prime == s_prev and d < self.k:
                            cost -= self.gamma

                        self.valid_trans[dst_switch].append((src, cost))

    def decode(
        self,
        log_probs: np.ndarray,
        sqi_scores: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Decode a single sequence with augmented Viterbi.

        Args:
            log_probs: (T, C) log-probabilities from the backbone (log-softmax).
            sqi_scores: (T,) optional signal quality scores in [0, 1].
                When SQI is low, emission scores are damped toward uniform.

        Returns:
            stages: (T,) integer stage labels.
        """
        T, C = log_probs.shape
        D = self.d_max
        n_aug = self.n_aug

        # Optionally damp emissions by SQI
        emissions = log_probs.copy()
        if sqi_scores is not None:
            uniform = np.full(C, -np.log(C))
            for t in range(T):
                q = sqi_scores[t]
                emissions[t] = q * emissions[t] + (1.0 - q) * uniform

        # Viterbi tables
        viterbi = np.full((T, n_aug), -np.inf)
        backptr = np.zeros((T, n_aug), dtype=np.int32)

        # Initialization: (s_prev=s, s, d=1) — sentinel s_prev=s means "no prior"
        for s in range(C):
            idx = self._aug_index(s, s, 1)
            viterbi[0, idx] = emissions[0, s]

        # Forward pass
        for t in range(1, T):
            for dst in range(n_aug):
                s_dst = self._aug_stage(dst)
                best_score = -np.inf
                best_src = 0

                for src, cost in self.valid_trans[dst]:
                    score = viterbi[t - 1, src] + cost
                    if score > best_score:
                        best_score = score
                        best_src = src

                viterbi[t, dst] = best_score + emissions[t, s_dst]
                backptr[t, dst] = best_src

        # Backtrace
        aug_path = np.zeros(T, dtype=np.int32)
        aug_path[T - 1] = int(np.argmax(viterbi[T - 1]))
        for t in range(T - 2, -1, -1):
            aug_path[t] = backptr[t + 1, aug_path[t + 1]]

        # Extract stage labels from augmented states
        return np.array([self._aug_stage(idx) for idx in aug_path])

    def decode_batch(
        self,
        log_probs: np.ndarray,
        sqi_scores: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Decode a batch of sequences.

        Args:
            log_probs: (B, T, C) log-probabilities.
            sqi_scores: (B, T) optional SQI scores.

        Returns:
            stages: (B, T) integer stage labels.
        """
        B = log_probs.shape[0]
        results = []
        for b in range(B):
            sqi_b = sqi_scores[b] if sqi_scores is not None else None
            results.append(self.decode(log_probs[b], sqi_b))
        return np.stack(results)
