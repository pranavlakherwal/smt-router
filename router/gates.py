"""Binary gate evaluators for the S(M,T) router.

Gates are hard constraints that filter out endpoints before phi scoring.
If any gate returns False, the endpoint is excluded (Product_j gate_j = 0).

Six gates:
1. context_window: Does the endpoint support the required context length?
2. format_support: Does the endpoint support required output format?
3. availability: Is the endpoint currently available?
4. budget: Does the endpoint fit within the cost budget?
5. thinking_support: If task requires thinking, does endpoint support it?
6. measured_bar: If the task sets an accuracy bar, has the endpoint cleared it twice on
   measured tasks of that kind? An endpoint with no measurement fails: a bar asks for evidence.
"""

import logging
from typing import List, Optional

from router.models import Endpoint, EndpointType, GateResult, Task

logger = logging.getLogger(__name__)


class GateEvaluator:
    """Evaluates binary gates for endpoint filtering."""

    def __init__(self, measured=None):
        # A router.measured.MeasuredTable, or None. Without one, a task that sets a bar
        # fails gate 6 for every endpoint.
        self.measured = measured

    def evaluate_all(self, task: Task, endpoint: Endpoint, kind: Optional[str] = None) -> List[GateResult]:
        """Run all gates. `kind` is the measured kind of job the task resolved to, if any."""
        results = [
            self._gate_context_window(task, endpoint),
            self._gate_format_support(task, endpoint),
            self._gate_availability(task, endpoint),
            self._gate_budget(task, endpoint),
            self._gate_thinking_support(task, endpoint),
            self._gate_measured_bar(task, endpoint, kind),
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
        reason = (
            "" if passed else (f"Need {task.context_length:,} tokens, endpoint supports {endpoint.context_window:,}")
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
            input_tokens * endpoint.cost_per_million_input + output_tokens * endpoint.cost_per_million_output
        ) / 1_000_000

        passed = estimated_cost <= task.max_budget_usd
        reason = "" if passed else (f"Estimated ${estimated_cost:.4f} exceeds budget ${task.max_budget_usd:.4f}")
        return GateResult(gate_name="budget", passed=passed, reason=reason)

    def _gate_thinking_support(self, task: Task, endpoint: Endpoint) -> GateResult:
        """Gate 5: If task requires chain-of-thought, endpoint must support it."""
        if not task.requires_thinking:
            return GateResult(gate_name="thinking_support", passed=True)

        passed = endpoint.supports_thinking
        reason = "" if passed else "Task requires thinking/CoT, endpoint lacks support"
        return GateResult(gate_name="thinking_support", passed=passed, reason=reason)

    def _gate_measured_bar(self, task: Task, endpoint: Endpoint, kind: Optional[str]) -> GateResult:
        """Gate 6: with a bar set, the endpoint must have cleared it twice on this kind of job."""
        if task.bar is None:
            return GateResult(gate_name="measured_bar", passed=True)
        if self.measured is None:
            return GateResult(gate_name="measured_bar", passed=False, reason="no measured table loaded")
        if not kind:
            return GateResult(gate_name="measured_bar", passed=False, reason="this kind of job was never measured")
        if endpoint.endpoint_type != EndpointType.LLM:
            return GateResult(gate_name="measured_bar", passed=False, reason="no measured pass rate for this endpoint")
        passed, why = self.measured.status(endpoint.id, kind, task.bar)
        return GateResult(gate_name="measured_bar", passed=passed, reason=why)
