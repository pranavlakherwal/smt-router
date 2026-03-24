"""Binary gate evaluators for the S(M,T) router.

Gates are hard constraints that filter out endpoints before phi scoring.
If any gate returns False, the endpoint is excluded (Product_j gate_j = 0).

Five gates:
1. context_window: Does the endpoint support the required context length?
2. format_support: Does the endpoint support required output format?
3. availability: Is the endpoint currently available?
4. budget: Does the endpoint fit within the cost budget?
5. thinking_support: If task requires thinking, does endpoint support it?
"""

import logging
from typing import List

from agents.router.models import Task, Endpoint, GateResult

logger = logging.getLogger(__name__)


class GateEvaluator:
    """Evaluates binary gates for endpoint filtering."""

    def evaluate_all(self, task: Task, endpoint: Endpoint) -> List[GateResult]:
        """Run all gates. Returns list of GateResult."""
        results = [
            self._gate_context_window(task, endpoint),
            self._gate_format_support(task, endpoint),
            self._gate_availability(task, endpoint),
            self._gate_budget(task, endpoint),
            self._gate_thinking_support(task, endpoint),
        ]
        return results

    def all_passed(self, results: List[GateResult]) -> bool:
        """Check if all gates passed."""
        return all(r.passed for r in results)

    # ----- Individual gates -----

    def _gate_context_window(self, task: Task, endpoint: Endpoint) -> GateResult:
        """Gate 1: Context window must be large enough for the task."""
        if task.context_length <= 0:
            return GateResult(gate_name="context_window", passed=True)

        passed = endpoint.context_window >= task.context_length
        reason = "" if passed else (
            f"Need {task.context_length:,} tokens, "
            f"endpoint supports {endpoint.context_window:,}"
        )
        return GateResult(gate_name="context_window", passed=passed, reason=reason)

    def _gate_format_support(self, task: Task, endpoint: Endpoint) -> GateResult:
        """Gate 2: Structured output support if required."""
        if not task.requires_structured_output:
            return GateResult(gate_name="format_support", passed=True)

        passed = endpoint.supports_structured_output
        reason = "" if passed else "Task requires structured output, endpoint lacks support"
        return GateResult(gate_name="format_support", passed=passed, reason=reason)

    def _gate_availability(self, task: Task, endpoint: Endpoint) -> GateResult:
        """Gate 3: Endpoint must be marked available."""
        passed = endpoint.available
        reason = "" if passed else "Endpoint is currently unavailable"
        return GateResult(gate_name="availability", passed=passed, reason=reason)

    def _gate_budget(self, task: Task, endpoint: Endpoint) -> GateResult:
        """Gate 4: Estimated cost must fit within budget."""
        if task.max_budget_usd is None:
            return GateResult(gate_name="budget", passed=True)

        # Rough cost estimate: assume ~1000 input tokens + ~500 output tokens
        # if no context_length given, otherwise use context_length + 500 output
        input_tokens = max(task.context_length, 1000)
        output_tokens = 500
        estimated_cost = (
            input_tokens * endpoint.cost_per_million_input
            + output_tokens * endpoint.cost_per_million_output
        ) / 1_000_000

        passed = estimated_cost <= task.max_budget_usd
        reason = "" if passed else (
            f"Estimated ${estimated_cost:.4f} exceeds budget ${task.max_budget_usd:.4f}"
        )
        return GateResult(gate_name="budget", passed=passed, reason=reason)

    def _gate_thinking_support(self, task: Task, endpoint: Endpoint) -> GateResult:
        """Gate 5: If task requires chain-of-thought, endpoint must support it."""
        if not task.requires_thinking:
            return GateResult(gate_name="thinking_support", passed=True)

        passed = endpoint.supports_thinking
        reason = "" if passed else "Task requires thinking/CoT, endpoint lacks support"
        return GateResult(gate_name="thinking_support", passed=passed, reason=reason)
