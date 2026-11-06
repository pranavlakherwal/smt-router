"""Routing engine: the main orchestrator for S(M,T) endpoint selection.

Pipeline: Registry -> Gates -> Phi Evaluation -> Scoring -> Selection

Two ways to ask:

    engine = RoutingEngine()

    # 1. The cheapest endpoint that does the job. Name the kind of job and the bar.
    decision = engine.route(Task(query="...", kind="maths", bar=90))

    # 2. The highest S(M,T) score, as in the paper. No bar.
    decision = engine.route(Task(query="...", task_type=TaskType.MATH))

    print(decision.selected.endpoint.id, decision.rule)
    print(engine.explain(decision))

With a bar, an endpoint must have cleared it twice on measured tasks of that kind (gate 6),
and the cheapest by measured cost per task is selected. The S(M,T) score is still computed
and shown, and it does not choose. Without a bar, the highest score is selected.
"""

import json
import logging
import uuid
from typing import Any, Dict, Optional

from router.config import PIPELINE, decision_log_path
from router.gates import GateEvaluator
from router.measured import BARS, MeasuredTable
from router.models import FeedbackSignal, RoutingDecision, Task
from router.phi_evaluators import PhiEvaluatorEngine
from router.registry import EndpointRegistry
from router.scorer import SMTScorer
from router.weight_learner import WeightLearner

logger = logging.getLogger(__name__)


class RoutingEngine:
    """Main routing engine implementing the S(M,T) framework."""

    def __init__(
        self,
        registry: Optional[EndpointRegistry] = None,
        top_k: int = 3,
        measured: Optional[MeasuredTable] = None,
    ):
        self.measured = measured or MeasuredTable()
        self.registry = registry or EndpointRegistry(measured=self.measured)
        self.gates = GateEvaluator(measured=self.measured)
        self.phi_engine = PhiEvaluatorEngine(measured=self.measured)
        self.learner = WeightLearner()
        self.scorer = SMTScorer(decision_count=self.learner.decision_count)
        self.top_k = top_k or PIPELINE.get("top_k", 3)

    def options(self, kind: str, bars=BARS) -> Optional[Dict[str, Any]]:
        """Up to three options for a kind of job: the cheapest setting at each bar.

        None when the kind was never measured.
        """
        return self.measured.options(kind, bars)

    def route(self, task: Task) -> RoutingDecision:
        """Route a task to an endpoint.

        Pipeline:
        1. Get available endpoints from registry
        2. Run gates on each endpoint (hard constraint filtering)
        3. Evaluate 16 phi functions for each passing endpoint
        4. Compute S(M,T) scores
        5. Apply entropy regularization
        6. Select: the cheapest that cleared the bar when the task sets one, else the top score

        Raises ValueError on an empty query or a kind that was never measured by name.
        """
        if not task.query or not task.query.strip():
            raise ValueError("Task.query is empty: there is nothing to route")
        if not task.id:
            task.id = str(uuid.uuid4())[:8]

        # The measured kind of job this task stands for, if any.
        if task.kind:
            kind = self.measured.resolve_kind(task.kind)
            if kind is None:
                known = ", ".join(sorted(self.measured.kinds))
                raise ValueError(f"kind {task.kind!r} was never measured; measured kinds: {known}")
        else:
            kind = self.measured.kind_for_task_type(task.task_type)
        rule = "cheapest_at_bar" if task.bar is not None else "score"

        if task.bar is not None and kind is None:
            return RoutingDecision(
                task=task,
                rule=rule,
                reason=(
                    f"NOT MEASURED: no tasks of type '{task.task_type.value}' have been measured, "
                    "so no endpoint can show it clears a bar"
                ),
            )

        endpoints = self.registry.get_available()
        if not endpoints:
            logger.warning("No available endpoints in registry")
            return RoutingDecision(task=task, rule=rule, reason="no available endpoints in the registry")

        logger.info(
            f"Routing task '{task.query[:50]}...' (type={task.task_type.value}) across {len(endpoints)} endpoints"
        )

        # Step 1-3: Gate + Phi evaluation for each endpoint
        scored_endpoints = []
        for ep in endpoints:
            # Gates
            gate_results = self.gates.evaluate_all(task, ep, kind)
            gates_passed = self.gates.all_passed(gate_results)

            # Phi evaluation (only if gates pass, for efficiency)
            if gates_passed:
                phi_scores = self.phi_engine.evaluate_all(task, ep, kind)
            else:
                phi_scores = []

            # Score
            scored = self.scorer.score_endpoint(task, ep, gate_results, phi_scores)
            if kind:
                scored.measured = dict(self.measured.row(ep.id, kind) or {})
            scored_endpoints.append(scored)

        # Step 4: Entropy regularization and sorting
        scored_endpoints = self.scorer.score_all(task, scored_endpoints)

        # Step 5: Select
        passing = [se for se in scored_endpoints if se.gates_passed]

        def cost(se):
            # Measured cost per task when there is one, else the list price per million tokens.
            m = se.measured.get("usd_per_task")
            list_price = se.endpoint.cost_per_million_input + se.endpoint.cost_per_million_output
            return (0, m) if m is not None else (1, list_price)

        if task.bar is not None:
            # The cheapest endpoint that cleared the bar twice, by measured cost per task.
            passing.sort(key=lambda se: (cost(se), se.endpoint.id))
        else:
            # The highest score. Equal scores go to the cheaper endpoint, not to the first name.
            passing.sort(key=lambda se: (-se.final_score, cost(se), se.endpoint.id))
        top_k = passing[: self.top_k]
        selected = top_k[0] if top_k else None
        reason = ""
        if selected is None:
            reason = (
                f"no endpoint cleared {task.bar} twice on {self.measured.label(kind).lower()}"
                if task.bar is not None
                else "no endpoint passed all gates"
            )

        # Build decision
        decision = RoutingDecision(
            task=task,
            scored_endpoints=scored_endpoints,
            selected=selected,
            top_k=top_k,
            decision_number=self.learner.decision_count + 1,
            rule=rule,
            reason=reason,
            metadata={"kind": kind, "measured_as_of": self.measured.as_of if kind else None},
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
                runners_up = ", ".join(f"{se.endpoint.name}={se.final_score:.4f}" for se in top_k[1:])
                logger.info(f"Runners-up: {runners_up}")
        else:
            logger.warning(f"Nothing selected: {reason}")

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

        logger.info(f"Feedback processed: quality={signal.quality_score:.2f}, decisions={self.learner.decision_count}")

    def explain(self, decision: RoutingDecision, max_filtered: Optional[int] = None) -> str:
        """Generate human-readable explanation of a routing decision.

        `max_filtered` caps how many filtered-out endpoints are listed.
        """
        lines = []
        kind = decision.metadata.get("kind")
        lines.append(f"Task: {decision.task.query[:80]}")
        lines.append(f"Type: {decision.task.task_type.value}")
        if kind:
            lines.append(f"Kind of job: {self.measured.label(kind)} (measured {self.measured.as_of})")
        else:
            lines.append("Kind of job: not measured, so capability scores are priors")
        if decision.rule == "cheapest_at_bar":
            lines.append(f"Rule: the cheapest endpoint that cleared {decision.task.bar} right in 100, twice")
        else:
            lines.append("Rule: the highest S(M,T) score")
        lines.append(f"Decision #{decision.decision_number}")
        lines.append("")

        if decision.selected:
            sel = decision.selected
            lines.append(f"Selected: {sel.endpoint.name}")
            lines.append(f"  Type: {sel.endpoint.endpoint_type.value}")
            if sel.measured.get("n"):
                m = sel.measured
                fresh = f", then {m['fresh_right']} of {m['fresh_n']} on fresh tasks" if m.get("fresh_n") else ""
                lines.append(
                    f"  Measured: {m['right']} of {m['n']}{fresh}; ${m['usd_per_task'] * 1000:.2f} per 1,000 tasks"
                )
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
                lines.append(f"    {ps.name}: {ps.score:.3f} * w={w:.4f} = {contribution:.4f} [{ps.source}]")
        else:
            lines.append(f"No endpoint selected: {decision.reason or 'all gates failed'}")

        # Runner-up summary
        if len(decision.top_k) > 1:
            lines.append("")
            lines.append("Alternatives:")
            for se in decision.top_k[1:]:
                cost = (
                    f", ${se.measured['usd_per_task'] * 1000:.2f} per 1,000 tasks"
                    if se.measured.get("usd_per_task") is not None
                    else ""
                )
                lines.append(f"  {se.endpoint.name}: {se.final_score:.4f} ({se.endpoint.endpoint_type.value}{cost})")

        # Gate failures
        failed = [se for se in decision.scored_endpoints if not se.gates_passed]
        if failed:
            lines.append("")
            lines.append(f"Filtered out ({len(failed)}):")
            shown = failed if max_filtered is None else failed[:max_filtered]
            for se in shown:
                reasons = [g.reason for g in se.gate_results if not g.passed]
                lines.append(f"  {se.endpoint.name}: {'; '.join(reasons)}")
            if len(shown) < len(failed):
                lines.append(f"  ... and {len(failed) - len(shown)} more")

        return "\n".join(lines)

    def stats(self) -> dict:
        """Get router statistics."""
        return {
            "endpoints": len(self.registry.get_all()),
            "available": len(self.registry.get_available()),
            "by_type": self.registry.count_by_type(),
            "decisions": self.learner.decision_count,
            "convergence": self.learner.get_convergence_stats(),
            "weights": {k: round(v, 4) for k, v in self.scorer.weights.items()},
            "data_sources": self.phi_engine.data_sources(),
        }

    def close(self):
        """Clean up resources."""
        self.phi_engine.close()
        self.learner.save()

    def _log_decision(self, decision: RoutingDecision):
        """Append the decision to a JSONL log, only when SMT_ROUTER_STATE_DIR is set."""
        log_path = decision_log_path()
        if log_path is None:
            return

        try:
            log_entry = {
                "timestamp": decision.timestamp,
                "task_id": decision.task.id,
                "task_type": decision.task.task_type.value,
                "query_preview": decision.task.query[:100],
                "selected": decision.selected.endpoint.id if decision.selected else None,
                "selected_score": decision.selected.final_score if decision.selected else 0,
                "top_k": [{"id": se.endpoint.id, "score": se.final_score} for se in decision.top_k],
                "total_evaluated": len(decision.scored_endpoints),
                "gates_failed": sum(1 for se in decision.scored_endpoints if not se.gates_passed),
                "decision_number": decision.decision_number,
                "rule": decision.rule,
                "bar": decision.task.bar,
                "kind": decision.metadata.get("kind"),
            }

            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
        except OSError as e:
            logger.warning(f"Could not write the decision log at {log_path}: {e}")
