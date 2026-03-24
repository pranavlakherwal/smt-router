"""S(M,T) scorer: the core routing equation.

Implements:
  S(M,T) = [Product_j gate_j(M,T)] * [Sum_i w_i * phi_i(M,T)] * e^(-beta * H_i)

With entropy regularization:
  S_reg = S(M,T) + lambda * H(pi)

Where:
- gate_j: binary hard constraints (0 or 1)
- w_i: phi function weights (sum to 1, from seed or learned)
- phi_i: phi function scores (0-1 per endpoint per task)
- beta: task-conditional temperature parameter
- H_i: entropy term for exploration
- lambda: entropy regularization coefficient (decays over time)
- H(pi): policy entropy over endpoint selection distribution
"""

import math
import logging
from typing import Dict, List, Optional

from agents.router.models import (
    Task, Endpoint, ScoredEndpoint, GateResult, PhiScore,
)
from agents.router.config import load_seed_weights, load_learned_weights

logger = logging.getLogger(__name__)


class SMTScorer:
    """Computes S(M,T) scores for endpoints given a task."""

    def __init__(self, decision_count: int = 0):
        self._seed_config = load_seed_weights()
        self._weights = load_learned_weights()
        self._decision_count = decision_count

        # Beta config
        beta_config = self._seed_config.get("beta", {})
        self._beta_c = beta_config.get("c", 0.1)
        self._beta_task = beta_config.get("task_conditional", {})

        # Entropy regularization config
        entropy_config = self._seed_config.get("entropy_regularization", {})
        self._lambda_initial = entropy_config.get("initial_lambda", 0.5)
        self._lambda_decay = entropy_config.get("decay_rate", 0.995)
        self._lambda_min = entropy_config.get("min_lambda", 0.01)

    @property
    def weights(self) -> Dict[str, float]:
        return self._weights

    @weights.setter
    def weights(self, new_weights: Dict[str, float]):
        self._weights = new_weights

    def score_endpoint(
        self,
        task: Task,
        endpoint: Endpoint,
        gate_results: List[GateResult],
        phi_scores: List[PhiScore],
    ) -> ScoredEndpoint:
        """Compute S(M,T) for a single (endpoint, task) pair."""

        # Gate product: if any gate fails, score is 0
        gates_passed = all(g.passed for g in gate_results)
        if not gates_passed:
            return ScoredEndpoint(
                endpoint=endpoint,
                gate_results=gate_results,
                gates_passed=False,
                phi_scores=phi_scores,
                weighted_phi_sum=0.0,
                temperature_factor=0.0,
                entropy_bonus=0.0,
                final_score=0.0,
            )

        # Weighted phi sum: Sum_i w_i * phi_i(M,T)
        weighted_sum = 0.0
        for ps in phi_scores:
            w = self._weights.get(ps.phi_id, 0.0)
            weighted_sum += w * ps.score

        # Temperature factor: e^(-beta * H_i)
        # beta is task-conditional
        beta = self._get_beta(task)
        # H_i: use model uncertainty (inverse of average confidence) as entropy proxy
        avg_confidence = sum(ps.confidence for ps in phi_scores) / max(len(phi_scores), 1)
        h_i = 1.0 - avg_confidence  # higher uncertainty = higher entropy
        temp_factor = math.exp(-beta * h_i)

        # Core S(M,T)
        smt_score = weighted_sum * temp_factor

        return ScoredEndpoint(
            endpoint=endpoint,
            gate_results=gate_results,
            gates_passed=True,
            phi_scores=phi_scores,
            weighted_phi_sum=round(weighted_sum, 6),
            temperature_factor=round(temp_factor, 6),
            entropy_bonus=0.0,  # computed in batch by score_all
            final_score=round(smt_score, 6),
        )

    def score_all(
        self,
        task: Task,
        scored_endpoints: List[ScoredEndpoint],
    ) -> List[ScoredEndpoint]:
        """Apply entropy regularization across all endpoints and sort by final score.

        S_reg = S(M,T) + lambda * H(pi)
        where H(pi) = -sum(p_i * log(p_i)) is the policy entropy.
        """
        # Filter to passed-gate endpoints
        passed = [se for se in scored_endpoints if se.gates_passed]
        if not passed:
            return scored_endpoints

        # Compute lambda (decays with decisions)
        lam = self._get_lambda()

        # Convert raw scores to probabilities via softmax
        raw_scores = [se.final_score for se in passed]
        max_s = max(raw_scores) if raw_scores else 0
        # Stable softmax
        exp_scores = [math.exp(s - max_s) for s in raw_scores]
        exp_sum = sum(exp_scores)
        probs = [e / exp_sum for e in exp_scores] if exp_sum > 0 else [1.0 / len(passed)] * len(passed)

        # Policy entropy: H(pi) = -sum(p_i * log(p_i))
        policy_entropy = 0.0
        for p in probs:
            if p > 1e-10:
                policy_entropy -= p * math.log(p)

        # Apply entropy bonus to each endpoint
        for i, se in enumerate(passed):
            # Per-endpoint entropy contribution: proportional to how much
            # selecting this endpoint contributes to policy diversity
            # Uniform would give max entropy, so bonus rewards diversity
            se.entropy_bonus = round(lam * policy_entropy, 6)
            se.final_score = round(se.final_score + se.entropy_bonus, 6)

        # Sort all (passed first, then failed) by final_score descending
        scored_endpoints.sort(key=lambda se: se.final_score, reverse=True)
        return scored_endpoints

    def _get_beta(self, task: Task) -> float:
        """Get task-conditional beta value.

        beta(t) = c * log(1 + t) with task-conditional multiplier.
        """
        t = self._decision_count
        base_beta = self._beta_c * math.log(1 + t) if t > 0 else self._seed_config.get("beta", {}).get("initial_beta", 0.1)
        task_multiplier = self._beta_task.get(task.task_type.value, 0.5)
        return base_beta * task_multiplier

    def _get_lambda(self) -> float:
        """Get current entropy regularization lambda.

        Decays exponentially: lambda(n) = max(lambda_min, lambda_0 * decay^n)
        """
        n = self._decision_count
        current = self._lambda_initial * (self._lambda_decay ** n)
        return max(current, self._lambda_min)

    def update_decision_count(self, n: int):
        """Update the decision counter (for beta/lambda scheduling)."""
        self._decision_count = n
