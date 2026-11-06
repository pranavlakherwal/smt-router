"""The rule: the option at a bar is the cheapest setting that hit the bar and held it on fresh tasks."""

from conftest import make_table, row


def world(b_fresh):
    return make_table(
        {
            "cheap": {"k": row(91, 0.00001, fresh_right=90)},
            "mid": {"k": row(96, 0.00002, fresh_right=b_fresh)},
            "dear": {"k": row(97, 0.00009, fresh_right=96)},
            "untested": {"k": row(99, 0.000001)},
            "weak": {"k": row(70, 0.0000001)},
        }
    )


def picks(table):
    return {o["bar"]: o["entry"] for o in table.options("k")["options"]}


def test_cheapest_that_held_is_the_option():
    assert picks(world(b_fresh=95)) == {90: "cheap", 95: "mid", 98: None}


def test_a_setting_that_slips_on_fresh_tasks_loses_the_bar():
    # Same first-task scores. Only the fresh result of "mid" changes, and the 95 option moves.
    assert picks(world(b_fresh=94)) == {90: "cheap", 95: "dear", 98: None}


def test_a_setting_never_sent_to_fresh_tasks_is_not_an_option():
    t = world(b_fresh=95)
    assert picks(t)[98] is None
    assert t.status("untested", "k", 98) == (False, "99 of 100 on the first tasks, never sent to fresh tasks")
    o98 = t.options("k")["options"][2]
    assert (o98["settings_that_hit"], o98["settings_retested"], o98["settings_that_held"]) == (1, 0, 0)


def test_the_cheapest_wrong_setting_is_never_an_option():
    assert "weak" not in picks(world(b_fresh=95)).values()


def test_cost_is_the_measured_cost_per_task():
    a = make_table({"x": {"k": row(95, 0.00005, fresh_right=95)}, "y": {"k": row(95, 0.00001, fresh_right=95)}})
    b = make_table({"x": {"k": row(95, 0.00001, fresh_right=95)}, "y": {"k": row(95, 0.00005, fresh_right=95)}})
    assert picks(a)[90] == "y" and picks(b)[90] == "x"


def test_a_kind_never_measured_has_no_options():
    assert world(95).options("coding") is None


def test_plain_names_resolve_only_to_measured_kinds():
    t = make_table({"x": {"gsm8k_math": row(95, 0.00001, fresh_right=95)}})
    assert t.resolve_kind("maths") == "gsm8k_math"
    assert t.resolve_kind("science") is None
