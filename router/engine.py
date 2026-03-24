"""Routing engine: the main orchestrator for S(M,T) endpoint selection.

Pipeline: Registry -> Gates -> Phi Evaluation -> Scoring -> Selection

Usage:
    engine = RoutingEngine()
    decision = engine.route(task)
    print(decision.selected.endpoint.name)
    # Later, after execution:
    engine.feedback(feedback_signal, decision)
"""

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from router.models import (
    Task, Endpoint, ScoredEndpoint, RoutingDecision,
    FeedbackSignal, TaskType,
)
from router.config import PIPELINE
from router.registry import EndpointRegistry
from router.gates import GateEvaluator
from router.phi_evaluators import PhiEvaluatorEngine
from router.scorer import SMTScorer
from router.weight_learner import WeightLearner

logger = logging.getLogger(__name__)


class RoutingEngine:
    """Main routing engine implementing the S(M,T) framework."""

    def __init__(
        self,
        registry: Optional[EndpointRegistry] = None,
        top_k: int = 3,
    ):
        self.registry = registry or EndpointRegistry()
        self.gates = GateEvaluator()
        self.phi_engine = PhiEvaluatorEngine()
        self.learner = WeightLearner()
        self.scorer = SMTScorer(decision_count=self.learner.decision_count)
        self.top_k = top_k or PIPELINE.get("top_k", 3)
        self._decision_log_path = PIPELINE.get("decision_log_path")

    def route(self, task: Task) -> RoutingDecision:
        """Route a task to the best endpoint(s).

        Pipeline:
        1. Get available endpoints from registry
        2. Run gates on each endpoint (hard constraint filtering)
        3. Evaluate 16 phi functions for each passing endpoint
        4. Compute S(M,T) scores
        5. Apply entropy regularization
        6. Select top-k
        """
        if not task.id:
            task.id = str(uuid.uuid4())[:8]

        endpoints = self.registry.get_available()
        if not endpoints:
            logger.warning("No available endpoints in registry")
            return RoutingDecision(task=task)

        logger.info(
            f"Routing task '{task.query[:50]}...' "
            f"(type={task.task_type.value}) across {len(endpoints)} endpoints"
        )

        # Step 1-3: Gate + Phi evaluation for each endpoint
        scored_endpoints = []
        for ep in endpoints:
            # Gates
            gate_results = self.gates.evaluate_all(task, ep)
            gates_passed = self.gates.all_passed(gate_results)

            # Phi evaluation (only if gates pass, for efficiency)
            if gates_passed:
                phi_scores = self.phi_engine.evaluate_all(task, ep)
            else:
                phi_scores = []

            # Score
            scored = self.scorer.score_endpoint(task, ep, gate_results, phi_scores)
            scored_endpoints.append(scored)

        # Step 4: Entropy regularization and sorting
        scored_endpoints = self.scorer.score_all(task, scored_endpoints)

        # Step 5: Select top-k
        passing = [se for se in scored_endpoints if se.gates_passed]
        top_k = passing[:self.top_k]
        selected = top_k[0] if top_k else None

        # Build decision
        decision = RoutingDecision(
            task=task,
            scored_endpoints=scored_endpoints,
            selected=selected,
            top_k=top_k,
            decision_number=self.learner.decision_count + 1,
        )

        # Log the decision
        self._log_decision(decision)

        if selected:
            logger.info(
                f"Selected: {selected.endpoint.name} "
                f"(score={selected.final_score:.4f}, "
                f"type={selected.endpoint.endpoint_type.value})"
            )
            if len(top_k) > 1:
                runners_up = ", ".join(
                    f"{se.endpoint.name}={se.final_score:.4f}"
                    for se in top_k[1:]
                )
                logger.info(f"Runners-up: {runners_up}")
        else:
            logger.warning("No endpoint passed all gates")

        return decision

    def feedback(self, signal: FeedbackSignal, decision: RoutingDecision):
        """Process feedback from a completed routing decision.

        Updates weights via Robbins-Monro and persists.
        """
        if not decision.selected:
            return

        signal.decision_id = decision.task.id
        signal.endpoint_id = decision.selected.endpoint.id
        signal.task_type = decision.task.task_type

        # Update weights
        new_weights = self.learner.update(signal, decision.selected)

        # Propagate updated weights to scorer
        self.scorer.weights = new_weights
        self.scorer.update_decision_count(self.learner.decision_count)

        logger.info(
            f"Feedback processed: quality={signal.quality_score:.2f}, "
            f"decisions={self.learner.decision_count}"
        )

    def explain(self, decision: RoutingDecision) -> str:
        """Generate human-readable explanation of a routing decision."""
        lines = []
        lines.append(f"Task: {decision.task.query[:80]}")
        lines.append(f"Type: {decision.task.task_type.value}")
        lines.append(f"Decision #{decision.decision_number}")
        lines.append("")

        if decision.selected:
            sel = decision.selected
            lines.append(f"Selected: {sel.endpoint.name}")
            lines.append(f"  Type: {sel.endpoint.endpoint_type.value}")
            lines.append(f"  Final Score: {sel.final_score:.4f}")
            lines.append(f"  Weighted Phi Sum: {sel.weighted_phi_sum:.4f}")
            lines.append(f"  Temperature Factor: {sel.temperature_factor:.4f}")
            lines.append(f"  Entropy Bonus: {sel.entropy_bonus:.4f}")
            lines.append("")

            # Top phi contributions
            lines.append("  Top Phi Contributions:")
            sorted_phis = sorted(
                sel.phi_scores,
                key=lambda ps: self.scorer.weights.get(ps.phi_id, 0) * ps.score,
                reverse=True,
            )
            for ps in sorted_phis[:5]:
                w = self.scorer.weights.get(ps.phi_id, 0)
                contribution = w * ps.score
                lines.append(
                    f"    {ps.name}: {ps.score:.3f} * w={w:.4f} = {contribution:.4f} "
                    f"[{ps.source}]"
                )
        else:
            lines.append("No endpoint selected (all gates failed)")

        # Runner-up summary
        if len(decision.top_k) > 1:
            lines.append("")
            lines.append("Alternatives:")
            for se in decision.top_k[1:]:
                lines.append(
                    f"  {se.endpoint.name}: {se.final_score:.4f} "
                    f"({se.endpoint.endpoint_type.value})"
                )

        # Gate failures
        failed = [se for se in decision.scored_endpoints if not se.gates_passed]
        if failed:
            lines.append("")
            lines.append(f"Filtered out ({len(failed)}):")
            for se in failed:
                reasons = [g.reason for g in se.gate_results if not g.passed]
                lines.append(f"  {se.endpoint.name}: {'; '.join(reasons)}")

        return "\n".join(lines)

    def stats(self) -> dict:
        """Get router statistics."""
        return {
            "endpoints": len(self.registry.get_all()),
            "available": len(self.registry.get_available()),
            "by_type": self.registry.count_by_type(),
            "decisions": self.learner.decision_count,
            "convergence": self.learner.get_convergence_stats(),
            "weights": {
                k: round(v, 4) for k, v in self.scorer.weights.items()
            },
        }

    def close(self):
        """Clean up resources."""
        self.phi_engine.close()
        self.learner.save()

    def _log_decision(self, decision: RoutingDecision):
        """Append decision to JSONL log."""
        if not self._decision_log_path:
            return

        try:
            log_entry = {
                "timestamp": decision.timestamp,
                "task_id": decision.task.id,
                "task_type": decision.task.task_type.value,
                "query_preview": decision.task.query[:100],
                "selected": decision.selected.endpoint.id if decision.selected else None,
                "selected_score": decision.selected.final_score if decision.selected else 0,
                "top_k": [
                    {"id": se.endpoint.id, "score": se.final_score}
                    for se in decision.top_k
                ],
                "total_evaluated": len(decision.scored_endpoints),
                "gates_failed": sum(
                    1 for se in decision.scored_endpoints if not se.gates_passed
                ),
                "decision_number": decision.decision_number,
            }

            log_path = Path(self._decision_log_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            logger.debug(f"Failed to log decision: {e}")
