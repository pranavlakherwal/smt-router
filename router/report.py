"""Every table in docs/ROUTE_OPTIONS_2026-11.md, printed from the files in data/.

  python -m router report            all tables, as Markdown
  python -m router report --only t3  one table

Nothing here calls a model. A reader can recount every number from data/measured_capability.json
and data/outcomes_*.json.
"""

import json
import math
from typing import Any, Dict, List, Optional, Tuple

from router.config import DATA_DIR
from router.measured import BARS, MeasuredTable
from router.models import Task

OFF = "@think-off"


def wilson(right: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """The 95% Wilson interval for right out of n, as two percentages."""
    if not n:
        return (0.0, 100.0)
    p = right / n
    d = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * (mid - half), 100 * (mid + half))


def _md(head: List[str], rows: List[List[Any]]) -> str:
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _usd(per_task: float) -> str:
    return f"{per_task * 1000:.3f}"


def _outcomes() -> Dict[str, Any]:
    files = sorted(DATA_DIR.glob("outcomes_*.json"))
    if not files:
        raise FileNotFoundError(f"no outcomes file in {DATA_DIR}")
    return json.loads(files[-1].read_text())


def t1_options(t: MeasuredTable) -> str:
    rows = []
    for kind in t.kinds:
        for o in t.options(kind)["options"]:
            if o["entry"]:
                lo, hi = wilson(o["right"] + o["fresh_right"], o["n"] + o["fresh_n"])
                rows.append(
                    [
                        t.label(kind),
                        o["bar"],
                        f"`{o['entry']}`",
                        f"{o['right']} of {o['n']}",
                        f"{o['fresh_right']} of {o['fresh_n']}",
                        f"{lo:.0f} to {hi:.0f}",
                        f"{o['usd_per_1000']:.3f}",
                    ]
                )
            else:
                rows.append([t.label(kind), o["bar"], "none", "", "", "", ""])
    return _md(
        [
            "Kind of job",
            "Bar",
            "Cheapest setting that cleared it twice",
            "First tasks",
            "Fresh tasks",
            "95% interval, both runs",
            "$ per 1,000 tasks",
        ],
        rows,
    )


def t2_counts(t: MeasuredTable) -> str:
    rows = []
    for kind in t.kinds:
        for o in t.options(kind)["options"]:
            rows.append(
                [t.label(kind), o["bar"], o["settings_that_hit"], o["settings_retested"], o["settings_that_held"]]
            )
    return _md(
        [
            "Kind of job",
            "Bar",
            "Settings that hit it on the first tasks",
            "Of those, re-tested on fresh tasks",
            "Held it on fresh tasks",
        ],
        rows,
    )


def retested(t: MeasuredTable) -> List[Dict[str, Any]]:
    """Every (kind, setting) pair that hit a bar and was sent to fresh tasks, with the top bar it hit and held."""
    out = []
    for kind in t.kinds:
        for name in t.entries:
            r = t.row(name, kind)
            if not r or not r.get("fresh_n"):
                continue
            hit = [b for b in BARS if r["right"] * 100 >= b * r["n"]]
            if not hit:
                continue
            held = [b for b in BARS if r["fresh_right"] * 100 >= b * r["fresh_n"]]
            out.append(
                {
                    "kind": kind,
                    "entry": name,
                    "right": r["right"],
                    "n": r["n"],
                    "fresh_right": r["fresh_right"],
                    "fresh_n": r["fresh_n"],
                    "top_hit": max(hit),
                    "top_held": max(held) if held else None,
                    "slipped": (max(held) if held else 0) < max(hit),
                    "usd_per_task": r["usd_per_task"],
                }
            )
    return sorted(out, key=lambda x: (list(t.kinds).index(x["kind"]), x["usd_per_task"], x["entry"]))


def t3_retested(t: MeasuredTable) -> str:
    rows = [
        [
            t.label(x["kind"]),
            f"`{x['entry']}`",
            f"{x['right']} of {x['n']}",
            f"{x['fresh_right']} of {x['fresh_n']}",
            x["top_hit"],
            x["top_held"] or "none",
            "slipped" if x["slipped"] else "held",
        ]
        for x in retested(t)
    ]
    return _md(["Kind of job", "Setting", "First tasks", "Fresh tasks", "Top bar hit", "Top bar held", "Result"], rows)


def totals(t: MeasuredTable) -> List[Dict[str, Any]]:
    out = []
    for name, e in t.entries.items():
        ks = e["kinds"]
        out.append(
            {
                "entry": name,
                **{k: ks[k]["right"] for k in t.kinds},
                "total": sum(ks[k]["right"] for k in t.kinds),
                "n": sum(ks[k]["n"] for k in t.kinds),
                "usd": sum(ks[k]["usd_per_task"] * ks[k]["n"] for k in t.kinds),
                "empty": sum(ks[k].get("empty") or 0 for k in t.kinds),
                "cut": sum(ks[k].get("cut") or 0 for k in t.kinds),
            }
        )
    return sorted(out, key=lambda x: (-x["total"], x["usd"], x["entry"]))


def t4_all(t: MeasuredTable) -> str:
    rows = [
        [f"`{x['entry']}`", *[x[k] for k in t.kinds], x["total"], f"{x['usd']:.4f}", x["empty"], x["cut"]]
        for x in totals(t)
    ]
    return _md(
        [
            "Setting",
            *[t.label(k) for k in t.kinds],
            "Right of 400",
            "$ for the 400 tasks",
            "Empty replies",
            "Cut at the limit",
        ],
        rows,
    )


def pairs(t: MeasuredTable) -> List[Dict[str, Any]]:
    by = {x["entry"]: x for x in totals(t)}
    out = []
    for name in sorted(by):
        if name.endswith(OFF) and name[: -len(OFF)] in by:
            on, off = by[name[: -len(OFF)]], by[name]
            out.append(
                {
                    "model": name[: -len(OFF)],
                    "on": on["total"],
                    "off": off["total"],
                    "on_usd": on["usd"],
                    "off_usd": off["usd"],
                    "ratio": off["usd"] / on["usd"] if on["usd"] else None,
                }
            )
    return out


def t5_thinking(t: MeasuredTable) -> str:
    rows = [
        [
            f"`{x['model']}`",
            x["on"],
            x["off"],
            x["off"] - x["on"],
            f"{x['on_usd']:.4f}",
            f"{x['off_usd']:.4f}",
            f"{x['ratio']:.2f}" if x["ratio"] is not None else "",
        ]
        for x in pairs(t)
    ]
    return _md(
        [
            "Model",
            "Right of 400, thinking on",
            "Thinking off",
            "Change",
            "$ for 400, on",
            "$ for 400, off",
            "Off cost as a share of on",
        ],
        rows,
    )


def score_picks(t: MeasuredTable) -> List[Dict[str, Any]]:
    """What the paper's score selects with no bar, beside the option at 90, for each kind."""
    from router.engine import RoutingEngine

    engine = RoutingEngine(measured=t)
    out = []
    for kind in t.kinds:
        d = engine.route(Task(query="a task of this kind", kind=kind))
        sel = d.selected
        tied = [se for se in d.scored_endpoints if se.gates_passed and se.final_score == sel.final_score]
        o90 = t.options(kind)["options"][0]
        out.append(
            {
                "kind": kind,
                "pick": sel.endpoint.id,
                "right": sel.measured["right"],
                "n": sel.measured["n"],
                "usd_per_task": sel.measured["usd_per_task"],
                "tied": len(tied),
                "option_90": o90["entry"],
                "option_90_right": o90.get("right"),
                "option_90_usd_per_1000": o90.get("usd_per_1000"),
            }
        )
    return out


def t6_score_vs_bar(t: MeasuredTable) -> str:
    rows = []
    for x in score_picks(t):
        opt = (
            f"`{x['option_90']}`, {x['option_90_right']} of {x['n']}, ${x['option_90_usd_per_1000']:.3f}"
            if x["option_90"]
            else "none"
        )
        rows.append(
            [t.label(x["kind"]), f"`{x['pick']}`", f"{x['right']} of {x['n']}", _usd(x["usd_per_task"]), x["tied"], opt]
        )
    return _md(
        [
            "Kind of job",
            "The score's pick, no bar",
            "First tasks",
            "$ per 1,000 tasks, first tasks",
            "Settings tied on the top score",
            "The option at 90: setting, first tasks, $ per 1,000",
        ],
        rows,
    )


def price_tiers(t: MeasuredTable) -> Dict[str, int]:
    """How many settings fall in each price tier of the cost term (phi_14), by list price in plus out per million."""
    tiers = {"free": 0, "under $1": 0, "$1 to $5": 0, "$5 to $20": 0, "$20 and over": 0}
    for e in t.entries.values():
        p = (e.get("price_in_per_m") or 0) + (e.get("price_out_per_m") or 0)
        key = (
            "free"
            if p <= 0
            else "under $1"
            if p < 1
            else "$1 to $5"
            if p < 5
            else "$5 to $20"
            if p < 20
            else "$20 and over"
        )
        tiers[key] += 1
    return tiers


def hard_tasks(t: MeasuredTable, strongest: int = 5, most: float = 0.9) -> List[Dict[str, Any]]:
    """Per kind: tasks all of the strongest settings got wrong, and tasks nine in ten settings got wrong."""
    o = _outcomes()["first"]
    out = []
    for kind in t.kinds:
        ids = o["tasks"][kind]
        marks = {n: e[kind] for n, e in o["entries"].items() if kind in e}
        best = sorted(marks, key=lambda n: (-marks[n].count("1"), n))[:strongest]
        all_best_wrong = [i for i in range(len(ids)) if all(marks[n][i] == "0" for n in best)]
        most_wrong = [i for i in range(len(ids)) if sum(m[i] == "0" for m in marks.values()) >= most * len(marks)]
        out.append(
            {
                "kind": kind,
                "tasks": len(ids),
                "settings": len(marks),
                "best_score": marks[best[0]].count("1"),
                "all_best_wrong": len(all_best_wrong),
                "most_wrong": len(most_wrong),
            }
        )
    return out


def t7_hard(t: MeasuredTable) -> str:
    rows = [[t.label(x["kind"]), x["best_score"], x["all_best_wrong"], x["most_wrong"]] for x in hard_tasks(t)]
    return _md(
        [
            "Kind of job",
            "Best score of any setting, of 100",
            "Tasks all five strongest settings got wrong",
            "Tasks nine in ten settings got wrong",
        ],
        rows,
    )


def suspect_keys() -> Dict[str, Any]:
    files = sorted(DATA_DIR.glob("suspect_keys_*.json"))
    if not files:
        raise FileNotFoundError(f"no suspect_keys file in {DATA_DIR}")
    return json.loads(files[-1].read_text())


def t8_suspect(t: MeasuredTable) -> str:
    rows = []
    for kind, v in suspect_keys()["kinds"].items():
        if not v["suspect"]:
            rows.append([t.label(kind), "none", "", "", "", ""])
        for r in v["suspect"]:
            rows.append(
                [
                    t.label(kind),
                    f"`{r['task_id']}`",
                    r["key"],
                    r["strongest_answer"],
                    f"{r['settings_giving_that_answer']} of {r['settings']}",
                    f"{r['settings_matching_the_key']} of {r['settings']}",
                ]
            )
    return _md(
        [
            "Kind of job",
            "Task",
            "The key says",
            "The strongest settings say",
            "Settings that say the same",
            "Settings that match the key",
        ],
        rows,
    )


def compact_options(t: MeasuredTable) -> str:
    """One row per kind of job, for the README."""
    rows = []
    for kind in t.kinds:
        cells = []
        for o in t.options(kind)["options"]:
            cells.append(f"`{o['entry']}`, ${o['usd_per_1000']:.3f}" if o["entry"] else "none")
        rows.append([t.label(kind), *cells])
    return _md(["Kind of job", "Cheapest at 90", "Cheapest at 95", "Cheapest at 98"], rows)


def fill(text: str, table: Optional[MeasuredTable] = None) -> str:
    """Rewrite every block between <!-- name --> and <!-- /name --> markers with the table of that name."""
    t = table or MeasuredTable()
    blocks = {key: fn for key, (_title, fn) in TABLES.items()}
    blocks["options"] = compact_options
    for key, fn in blocks.items():
        a, b = f"<!-- {key} -->", f"<!-- /{key} -->"
        if a in text and b in text:
            head, rest = text.split(a, 1)
            _old, tail = rest.split(b, 1)
            text = f"{head}{a}\n{fn(t)}\n{b}{tail}"
    return text


TABLES = {
    "t1": ("Table 1. The options", t1_options),
    "t2": ("Table 2. How many settings reached each bar", t2_counts),
    "t3": ("Table 3. Every hit that was re-tested on fresh tasks", t3_retested),
    "t4": ("Table 4. Every setting on the first 400 tasks", t4_all),
    "t5": ("Table 5. The same model with thinking on and off", t5_thinking),
    "t6": ("Table 6. What the paper's score picks when given the measured table", t6_score_vs_bar),
    "t7": ("Table 7. Tasks that almost no setting gets right", t7_hard),
    "t8": ("Table 8. Answer keys the strongest settings disagree with", t8_suspect),
}


def render(only: Optional[str] = None, table: Optional[MeasuredTable] = None) -> str:
    t = table or MeasuredTable()
    parts = []
    for key, (title, fn) in TABLES.items():
        if only and key != only:
            continue
        parts.append(f"### {title}\n\n{fn(t)}")
    return "\n\n".join(parts)
