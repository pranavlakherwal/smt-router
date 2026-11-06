"""The files that ship: the table is whole, the matrix recounts to it, and the commands run."""

import json

import router.config as config
from router.__main__ import main
from router.measured import MeasuredTable


def test_every_entry_answered_every_task_of_every_kind():
    t = MeasuredTable()
    assert len(t.kinds) == 4 and len(t.entries) >= 60
    for name, e in t.entries.items():
        assert set(e["kinds"]) == set(t.kinds), name
        for kind, r in e["kinds"].items():
            assert r["n"] == 100 and 0 <= r["right"] <= 100, (name, kind)
            assert (r["fresh_n"] is None) == (r["fresh_right"] is None), (name, kind)


def test_the_right_or_wrong_matrix_recounts_to_the_table():
    t = MeasuredTable()
    files = sorted(config.DATA_DIR.glob("outcomes_*.json"))
    assert files, "no outcomes file shipped"
    o = json.loads(files[-1].read_text())
    assert o["as_of"] == t.as_of
    for name, e in t.entries.items():
        for kind, r in e["kinds"].items():
            first = o["first"]["entries"][name][kind]
            assert (len(first), first.count("1")) == (r["n"], r["right"]), (name, kind)
            fresh = o["fresh"]["entries"].get(name, {}).get(kind, "").replace("-", "")
            assert (len(fresh), fresh.count("1")) == (r["fresh_n"] or 0, r["fresh_right"] or 0), (name, kind)


def test_the_commands_run_and_say_not_measured(capsys):
    assert main(["options"]) == 0
    assert "Maths word problems" in capsys.readouterr().out
    assert main(["options", "--kind", "coding"]) == 1
    assert "NOT MEASURED" in capsys.readouterr().out
    assert main(["table", "--kind", "exam", "--top", "3"]) == 0
    assert main(["route", "--kind", "maths", "--bar", "90", "What is 2 + 2?"]) == 0
    assert "cheapest endpoint that cleared 90" in capsys.readouterr().out
    assert main(["route", "--type", "coding", "--bar", "90", "Write a function."]) == 1
    assert main(["route", "   "]) == 2


def test_the_tables_in_the_note_and_the_readme_match_the_data():
    from router.report import fill

    for name in ("docs/ROUTE_OPTIONS_2026-11.md", "README.md"):
        text = (config.PROJECT_ROOT / name).read_text()
        assert "<!-- " in text, name
        assert fill(text) == text, f"{name} is out of step with data/: run python -m router report --fill {name}"
