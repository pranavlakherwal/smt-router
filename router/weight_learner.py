"""Robbins-Monro online weight learner for the S(M,T) router.

Implements stochastic approximation for phi function weights:
  w_{n+1} = w_n + a_n * gradient(feedback)

Where:
- a_n = 1 / (n + 1) (Robbins-Monro step size)
- gradient is derived from feedback signal quality vs prediction
- Weights are projected onto the simplex (sum to 1, all >= 0)

Convergence guaranteed by Hajek conditions (logarithmic cooling).
From regret_bounds proof: ~10,000 decisions to reach 95% optimal.
"""

import json
import logging
from typing import Dict, List, Optional

from router.models import (
    FeedbackSignal, ScoredEndpoint, PhiScore, TaskType,
)
from router.config import save_learned_weights, load_learned_weights

logger = logging.getLogger(__name__)


class WeightLearner:
    """Online weight updater using Robbins-Monro stochastic approximation."""

    def __init__(self, initial_weights: Optional[Dict[str, float]] = None):
        self._weights = initial_weights or load_learned_weights()
        self._decision_count = 0
        self._feedback_history: List[Dict] = []

    @property
    def weights(self) -> Dict[str, float]:
        return self._weights.copy()

    @property
    def decision_count(self) -> int:
        return self._decision_count

    def update(
        self,
        feedback: FeedbackSignal,
        scored_endpoint: ScoredEndpoint,
    ) -> Dict[str, float]:
        """Update weights based on feedback from a routing decision.

        The update rule:
        1. Compute prediction error: observed quality - predicted score
        2. Scale by step size a_n = 1 / (n + 1)
        3. Gradient: each phi weight moves proportionally to its contribution
           to the prediction error
        4. Project onto simplex (sum = 1, all >= 0)

        Returns updated weights.
        """
        self._decision_count += 1
        n = self._decision_count

        # Step size: Robbins-Monro a_n = 1 / (n + 1)
        step_size = 1.0 / (n + 1)

        # Prediction error: how far off was the router?
        predicted = scored_endpoint.weighted_phi_sum
        observed = feedback.quality_score

        error = observed - predicted

        # Gradient update: each weight is adjusted proportionally to
        # its phi_score * error (if phi was high and quality was low, reduce weight)
        phi_map = {ps.phi_id: ps.score for ps in scored_endpoint.phi_scores}

        new_weights = {}
        for phi_id, w in self._weights.items():
            phi_val = phi_map.get(phi_id, 0.5)
            # Gradient: phi_val * error tells us how much this phi
            # contributed to the prediction error
            gradient = phi_val * error
            new_w = w + step_size * gradient
            new_weights[phi_id] = new_w

        # Project onto probability simplex
        self._weights = self._project_simplex(new_weights)

        # Log the update
        logger.info(
            f"Weight update n={n}: step={step_size:.4f}, "
            f"error={error:.4f}, endpoint={feedback.endpoint_id}"
        )

        # Track history
        self._feedback_history.append({
            "n": n,
            "endpoint_id": feedback.endpoint_id,
            "task_type": feedback.task_type.value,
            "predicted": round(predicted, 4),
            "observed": round(observed, 4),
            "error": round(error, 4),
            "step_size": round(step_size, 4),
        })

        # Persist every 10 updates
        if n % 10 == 0:
            self.save()

        return self._weights

    def save(self):
        """Persist current weights to disk."""
        save_learned_weights(self._weights, self._decision_count)

    def _project_simplex(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Project weight vector onto the probability simplex.

        Ensures: all weights >= 0 and sum(weights) = 1.
        Uses the algorithm from Duchi et al. (2008).
        """
        keys = sorted(weights.keys())
        values = [weights[k] for k in keys]
        n = len(values)

        # Sort in descending order
        sorted_vals = sorted(values, reverse=True)

        # Find the threshold
        cumsum = 0.0
        threshold = 0.0
        for j in range(n):
            cumsum += sorted_vals[j]
            candidate = (cumsum - 1.0) / (j + 1)
            if sorted_vals[j] - candidate > 0:
                threshold = candidate

        # Project
        projected = {}
        for i, k in enumerate(keys):
            projected[k] = max(values[i] - threshold, 0.0)

        # Normalize to ensure exact sum = 1 (handle float precision)
        total = sum(projected.values())
        if total > 0:
            projected = {k: v / total for k, v in projected.items()}
        else:
            # Fallback to uniform
            projected = {k: 1.0 / n for k in keys}

        return projected

    def get_convergence_stats(self) -> Dict:
        """Get statistics about weight learning convergence."""
        if not self._feedback_history:
            return {"decisions": 0, "converged": False}

        recent = self._feedback_history[-100:]
        recent_errors = [abs(h["error"]) for h in recent]
        avg_error = sum(recent_errors) / len(recent_errors) if recent_errors else 1.0

        return {
            "decisions": self._decision_count,
            "avg_recent_error": round(avg_error, 4),
            "converged": avg_error < 0.05 and self._decision_count >= 100,
            "estimated_to_95pct": max(0, 10000 - self._decision_count),
            "step_size": round(1.0 / (self._decision_count + 1), 6),
        }
