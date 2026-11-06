"""The gates, the scorer and the weight learner, each in a passing world and a failing one."""

from router.gates import GateEvaluator
from router.models import Endpoint, GateResult, PhiScore, Task
from router.scorer import SMTScorer
from router.weight_learner import WeightLearner


def ep(**kw):
    base = {
        "id": "m",
        "name": "m",
        "context_window": 1000,
        "cost_per_million_input": 1.0,
        "cost_per_million_output": 2.0,
    }
    base.update(kw)
    return Endpoint(**base)


def gate(task, endpoint, name):
    return [g for g in GateEvaluator().evaluate_all(task, endpoint) if g.gate_name == name][0].passed


def test_context_window_gate():
    assert gate(Task(query="q", context_length=900), ep(), "context_window")
    assert not gate(Task(query="q", context_length=1100), ep(), "context_window")


def test_budget_gate():
    # 1,000 input tokens at $1 per million and 500 output tokens at $2 per million cost $0.002.
    assert gate(Task(query="q", max_budget_usd=0.0021), ep(), "budget")
    assert not gate(Task(query="q", max_budget_usd=0.0019), ep(), "budget")


def test_thinking_and_format_gates():
    assert gate(Task(query="q", requires_thinking=True), ep(supports_thinking=True), "thinking_support")
    assert not gate(Task(query="q", requires_thinking=True), ep(), "thinking_support")
    assert gate(Task(query="q", requires_structured_output=True), ep(supports_structured_output=True), "format_support")
    assert not gate(Task(query="q", requires_structured_output=True), ep(), "format_support")


def test_availability_gate():
    assert gate(Task(query="q"), ep(), "availability")
    assert not gate(Task(query="q"), ep(available=False), "availability")


def test_a_bar_with_no_measured_table_fails_every_endpoint():
    assert gate(Task(query="q"), ep(), "measured_bar")
    assert not gate(Task(query="q", bar=90), ep(), "measured_bar")


def test_a_failed_gate_zeroes_the_score():
    s = SMTScorer()
    phis = [PhiScore(phi_id="phi_4", name="performance_predictor", score=1.0)]
    ok = s.score_endpoint(Task(query="q"), ep(), [GateResult(gate_name="g", passed=True)], phis)
    bad = s.score_endpoint(Task(query="q"), ep(), [GateResult(gate_name="g", passed=False)], phis)
    assert ok.final_score > 0 and bad.final_score == 0.0 and not bad.gates_passed


def test_the_seed_weights_are_sixteen_and_sum_to_one():
    w = SMTScorer().weights
    assert len(w) == 16 and abs(sum(w.values()) - 1.0) < 1e-4


def test_the_learner_keeps_weights_on_the_simplex():
    learner = WeightLearner()
    projected = learner._project_simplex({"a": 2.0, "b": -1.0, "c": 0.5})
    assert abs(sum(projected.values()) - 1.0) < 1e-9 and min(projected.values()) >= 0.0
    assert projected["a"] > projected["c"] >= projected["b"]
