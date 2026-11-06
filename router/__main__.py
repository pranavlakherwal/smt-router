"""Command line for the S(M,T) router.

  python -m router options                 the three options for every measured kind of job
  python -m router options --kind maths    the three options for one kind
  python -m router route --kind maths --bar 90 "A train covers 120 km in 2 hours. Its speed?"
  python -m router route --type math "A train covers 120 km in 2 hours. Its speed?"
  python -m router table --kind science    every measured setting on one kind, cheapest first
  python -m router report                  every table of the research note, recounted from data/

Kinds: maths, science, claims, exam. Nothing here calls a model or the network.
"""

import argparse
import sys

from router.engine import RoutingEngine
from router.measured import MeasuredTable
from router.models import Task, TaskType


def _print_options(res) -> None:
    print(f"{res['label']}, measured {res['as_of']}")
    for o in res["options"]:
        if o["entry"]:
            print(
                f"  {o['bar']}: {o['entry']}  {o['right']} of {o['n']}, then {o['fresh_right']} of "
                f"{o['fresh_n']} on fresh tasks  ${o['usd_per_1000']:.2f} per 1,000 tasks"
            )
        else:
            print(f"  {o['bar']}: none. {o['why']}")
        print(
            f"      {o['settings_that_hit']} settings hit {o['bar']} on the first tasks; "
            f"{o['settings_retested']} of them were re-tested on fresh tasks; {o['settings_that_held']} held"
        )


def cmd_options(a) -> int:
    table = MeasuredTable()
    kinds = [a.kind] if a.kind else list(table.kinds)
    for i, k in enumerate(kinds):
        res = table.options(k)
        if res is None:
            print(f"NOT MEASURED: no measured kind called {k!r}. Measured kinds: maths, science, claims, exam")
            return 1
        if i:
            print()
        _print_options(res)
    return 0


def cmd_route(a) -> int:
    engine = RoutingEngine()
    task = Task(query=a.query, kind=a.kind, bar=a.bar, task_type=TaskType(a.type))
    try:
        decision = engine.route(task)
    except ValueError as e:
        print(f"REFUSED: {e}")
        return 2
    print(engine.explain(decision, max_filtered=a.show_filtered))
    return 0 if decision.selected else 1


def cmd_table(a) -> int:
    table = MeasuredTable()
    kind = table.resolve_kind(a.kind)
    if kind is None:
        print(f"NOT MEASURED: no measured kind called {a.kind!r}. Measured kinds: maths, science, claims, exam")
        return 1
    rows = table.ranking(kind)
    print(f"{table.label(kind)}, measured {table.as_of}: {len(rows)} settings, cheapest first")
    print(f"  {'setting':52s} {'right':>9s} {'fresh':>9s} {'$ per 1,000':>12s}")
    for r in rows[: a.top] if a.top else rows:
        fresh = f"{r['fresh_right']}/{r['fresh_n']}" if r.get("fresh_n") else "-"
        print(f"  {r['entry']:52s} {r['right']:>5d}/{r['n']:<3d} {fresh:>9s} {r['usd_per_task'] * 1000:>12.3f}")
    return 0


def cmd_report(a) -> int:
    from pathlib import Path

    from router.report import fill, render

    if a.fill:
        path = Path(a.fill)
        path.write_text(fill(path.read_text()))
        print(f"filled the table blocks in {path}")
        return 0
    print(render(a.only))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m router", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd")

    o = sub.add_parser("options", help="the cheapest setting at 90, 95 and 98 right in 100")
    o.add_argument("--kind", help="maths, science, claims or exam; all four when left out")
    o.set_defaults(fn=cmd_options)

    r = sub.add_parser("route", help="route one task and explain the choice")
    r.add_argument("query")
    r.add_argument("--kind", help="maths, science, claims or exam")
    r.add_argument("--bar", type=int, help="right answers wanted out of 100, for example 90")
    r.add_argument(
        "--type", default="general", choices=[t.value for t in TaskType], help="task type, used when no kind is given"
    )
    r.add_argument("--show-filtered", type=int, default=5, help="how many filtered-out endpoints to list")
    r.set_defaults(fn=cmd_route)

    t = sub.add_parser("table", help="every measured setting on one kind, cheapest first")
    t.add_argument("--kind", required=True)
    t.add_argument("--top", type=int, default=0, help="show only the cheapest N")
    t.set_defaults(fn=cmd_table)

    p = sub.add_parser("report", help="every table of the research note, recounted from data/")
    p.add_argument("--only", choices=["t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8"])
    p.add_argument("--fill", help="rewrite the marked table blocks in this Markdown file from data/")
    p.set_defaults(fn=cmd_report)

    a = ap.parse_args(argv)
    if not getattr(a, "fn", None):
        ap.print_help()
        return 2
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
