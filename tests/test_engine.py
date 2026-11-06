"""The engine: a bar picks the cheapest endpoint that cleared it twice; no bar picks the top score."""

import json

import pytest
from conftest import make_table, row
from pydantic import ValidationError

import router.config as config
from router.engine import RoutingEngine
from router.measured import MeasuredTable
from router.models import FeedbackSignal, Task, TaskType
from router.registry import EndpointRegistry


def engine_for(mid_fresh=95, include_paper_examples=False):
    t = make_table(
        {
            "cheap": {"gsm8k_math": row(91, 0.00001, fresh_right=90)},
            "mid": {"gsm8k_math": row(96, 0.00002, fresh_right=mid_fresh)},
            "dear": {"gsm8k_math": row(97, 0.00009, fresh_right=96)},
            "best": {"gsm8k_math": row(99, 0.00050)},
        }
    )
    reg = EndpointRegistry(measured=t, include_paper_examples=include_paper_examples)
    return RoutingEngine(registry=reg, measured=t)


def pick(engine, **kw):
    d = engine.route(Task(query="What is 2 + 2?", **kw))
    return d.selected.endpoint.id if d.selected else None


def test_bar_picks_the_cheapest_that_cleared_it_twice(no_state):
    e = engine_for()
    assert pick(e, kind="maths", bar=90) == "cheap"
    assert pick(e, kind="maths", bar=95) == "mid"
    assert pick(e, kind="maths", bar=98) is None


def test_a_slip_on_fresh_tasks_moves_the_pick(no_state):
    assert pick(engine_for(mid_fresh=94), kind="maths", bar=95) == "dear"


def test_no_bar_picks_the_highest_score_not_the_cheapest(no_state):
    e = engine_for()
    d = e.route(Task(query="What is 2 + 2?", task_type=TaskType.MATH))
    assert d.rule == "score" and d.selected.endpoint.id == "best"
    phi4 = [p for p in d.selected.phi_scores if p.phi_id == "phi_4"][0]
    assert (phi4.source, phi4.score) == ("measured", 0.99)


def test_equal_scores_go_to_the_cheaper_endpoint(no_state):
    for cheap, dear in (("a", "b"), ("b", "a")):
        t = make_table({cheap: {"gsm8k_math": row(95, 0.00001)}, dear: {"gsm8k_math": row(95, 0.00009)}})
        e = RoutingEngine(registry=EndpointRegistry(measured=t), measured=t)
        assert pick(e, task_type=TaskType.MATH) == cheap


def test_the_engine_and_the_options_table_agree_on_the_shipped_data(no_state):
    table = MeasuredTable()
    e = RoutingEngine(measured=table)
    for kind in table.kinds:
        for o in table.options(kind)["options"]:
            assert pick(e, kind=kind, bar=o["bar"]) == o["entry"], (kind, o["bar"])


def test_an_empty_query_is_refused(no_state):
    e = engine_for()
    for q in ("", "   "):
        with pytest.raises(ValueError, match="empty"):
            e.route(Task(query=q))


def test_a_misnamed_field_is_refused():
    with pytest.raises(ValidationError):
        Task(prompt="What is 2 + 2?")


def test_an_unknown_kind_is_refused_by_name(no_state):
    with pytest.raises(ValueError, match="never measured"):
        engine_for().route(Task(query="x", kind="poetry", bar=90))


def test_a_bar_on_an_unmeasured_task_type_selects_nothing_and_says_why(no_state):
    d = engine_for().route(Task(query="Write a function.", task_type=TaskType.CODING, bar=90))
    assert d.selected is None and d.reason.startswith("NOT MEASURED")


def test_without_a_bar_an_unmeasured_task_type_scores_from_priors_and_says_so(no_state):
    e = engine_for()
    d = e.route(Task(query="Write a function.", task_type=TaskType.CODING))
    assert d.selected is not None
    phi4 = [p for p in d.selected.phi_scores if p.phi_id == "phi_4"][0]
    assert phi4.source == "prior"
    assert "not measured" in e.explain(d)


def test_an_endpoint_with_no_measurement_cannot_pass_a_bar(no_state):
    e = engine_for(include_paper_examples=True)
    d = e.route(Task(query="What is 2 + 2?", kind="maths", bar=90))
    assert d.selected.endpoint.id == "cheap"
    failed = {se.endpoint.id: se for se in d.scored_endpoints if not se.gates_passed}
    assert "thinking" in failed and "claude-opus-4" in failed
    assert any(g.gate_name == "measured_bar" and not g.passed for g in failed["thinking"].gate_results)


def test_nothing_is_written_unless_a_state_folder_is_set(no_state, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    before = sorted(p.name for p in config.DATA_DIR.iterdir())
    e = engine_for()
    d = e.route(Task(query="What is 2 + 2?", kind="maths", bar=90))
    for _ in range(10):
        e.feedback(FeedbackSignal(quality_score=1.0), d)
    e.close()
    assert sorted(p.name for p in config.DATA_DIR.iterdir()) == before
    assert list(tmp_path.iterdir()) == []


def test_with_a_state_folder_the_log_and_the_weights_go_there(tmp_path, monkeypatch):
    monkeypatch.setenv("SMT_ROUTER_STATE_DIR", str(tmp_path / "state"))
    e = engine_for()
    d = e.route(Task(query="What is 2 + 2?", kind="maths", bar=90))
    e.feedback(FeedbackSignal(quality_score=1.0), d)
    e.close()
    log = [json.loads(x) for x in (tmp_path / "state" / "routing_decisions.jsonl").read_text().splitlines()]
    assert log[0]["selected"] == "cheap" and log[0]["rule"] == "cheapest_at_bar" and log[0]["bar"] == 90
    weights = json.loads((tmp_path / "state" / "learned_weights.json").read_text())
    assert abs(sum(weights["weights"].values()) - 1.0) < 1e-9 and weights["decision_count"] == 1


def test_missing_seed_weights_is_an_error_not_a_fallback(monkeypatch, tmp_path, no_state):
    monkeypatch.setattr(config, "SEED_WEIGHTS_PATH", tmp_path / "absent.json")
    with pytest.raises(FileNotFoundError, match="Seed weights"):
        engine_for()


def test_a_missing_measured_table_is_an_error_not_a_fallback(tmp_path):
    with pytest.raises(FileNotFoundError, match="Measured table"):
        MeasuredTable(path=tmp_path / "absent.json")
