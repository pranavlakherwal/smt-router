"""Measured capability: what each model setting actually scored on each kind of job.

A model's capability for a kind of job is its measured pass rate on tasks of that kind:
right answers out of n, with a date. It is not a leaderboard number and not a guess from price.

The rule this module implements:
  1. A setting "hits" a bar when its right answers on the first tasks reach the bar.
  2. It "holds" the bar when it reaches it again on fresh tasks it had not seen.
  3. The option at a bar is the cheapest setting that hit and held, by measured cost per task.
So a kind of job has up to three options, one each at 90, 95 and 98 right in 100.

A setting is a model with its request settings. An entry whose name ends "@think-off" is the
same model with thinking switched off, which is cheaper and sometimes just as right.

The table is data/measured_capability.json. Its "as_of" date is the day of the last run.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from router.config import DATA_DIR
from router.models import Endpoint, EndpointType, TaskType

MEASURED_PATH = DATA_DIR / "measured_capability.json"
BARS = (90, 95, 98)

# Plain names for the measured kinds.
ALIASES = {
    "maths": "gsm8k_math",
    "math": "gsm8k_math",
    "science": "arc_reasoning",
    "claims": "scifact_claims",
    "exam": "mmlu_knowledge",
}

# The measured kind that stands in for a task type when a task names no kind. A task type
# that is not here has no measured tasks: coding, writing and long documents among them.
KIND_FOR_TASK_TYPE = {
    TaskType.MATH: "gsm8k_math",
    TaskType.REASONING: "arc_reasoning",
    TaskType.KNOWLEDGE: "mmlu_knowledge",
}


def _hit(row: Optional[Dict[str, Any]], bar: int) -> bool:
    return bool(row and row.get("n")) and row["right"] * 100 >= bar * row["n"]


def _held(row: Dict[str, Any], bar: int) -> bool:
    return (
        bool(row.get("fresh_n"))
        and row.get("fresh_right") is not None
        and row["fresh_right"] * 100 >= bar * row["fresh_n"]
    )


class MeasuredTable:
    """The measured pass rates, and the options they support."""

    def __init__(self, path: Optional[Path] = None, table: Optional[Dict[str, Any]] = None):
        if table is None:
            path = Path(path) if path else MEASURED_PATH
            if not path.exists():
                raise FileNotFoundError(f"Measured table not found at {path}. The file ships with the repository.")
            table = json.loads(path.read_text())
        for key in ("as_of", "kinds", "entries"):
            if key not in table:
                raise ValueError(f"measured table has no '{key}'")
        self._t = table

    # ----- what is in the table -----

    @property
    def as_of(self) -> str:
        return self._t["as_of"]

    @property
    def kinds(self) -> Dict[str, Any]:
        return self._t["kinds"]

    @property
    def entries(self) -> Dict[str, Any]:
        return self._t["entries"]

    def label(self, kind: str) -> str:
        k = self.kinds.get(kind)
        return k.get("label", kind) if isinstance(k, dict) else str(k or kind)

    def resolve_kind(self, name: Optional[str]) -> Optional[str]:
        """A kind's key from its key or its plain name. None when it was never measured."""
        if not name:
            return None
        key = ALIASES.get(name.strip().lower(), name.strip())
        return key if key in self.kinds else None

    def kind_for_task_type(self, task_type: TaskType) -> Optional[str]:
        kind = KIND_FOR_TASK_TYPE.get(task_type)
        return kind if kind in self.kinds else None

    def row(self, entry: str, kind: str) -> Optional[Dict[str, Any]]:
        e = self.entries.get(entry)
        return (e.get("kinds") or {}).get(kind) if e else None

    # ----- the bar -----

    def status(self, entry: str, kind: str, bar: int) -> Tuple[bool, str]:
        """Did this entry clear the bar twice on this kind of job? Returns (yes or no, why)."""
        r = self.row(entry, kind)
        if not r or not r.get("n"):
            return False, f"not measured on {self.label(kind).lower()}"
        first = f"{r['right']} of {r['n']}"
        if not _hit(r, bar):
            return False, f"{first} on the first tasks, under {bar}"
        if not r.get("fresh_n"):
            return False, f"{first} on the first tasks, never sent to fresh tasks"
        fresh = f"{r['fresh_right']} of {r['fresh_n']}"
        if not _held(r, bar):
            return False, f"{first} on the first tasks, then {fresh} on fresh tasks, under {bar}"
        return True, f"{first}, then {fresh} on fresh tasks"

    def qualifiers(self, kind: str, bar: int) -> List[str]:
        """Entries that cleared the bar twice, cheapest first by measured cost per task."""
        ok = [n for n in self.entries if self.status(n, kind, bar)[0]]
        return sorted(ok, key=lambda n: (self.row(n, kind)["usd_per_task"], n))

    def options(self, kind: str, bars: Tuple[int, ...] = BARS) -> Optional[Dict[str, Any]]:
        """Up to three options for one kind of job. None when the kind was never measured."""
        kind = self.resolve_kind(kind) or ""
        if not kind:
            return None
        out = []
        for b in bars:
            hits = [n for n in self.entries if _hit(self.row(n, kind), b)]
            retested = [n for n in hits if self.row(n, kind).get("fresh_n")]
            twice = self.qualifiers(kind, b)
            if not twice:
                why = "no setting cleared this bar twice" if hits else "no setting reached this bar"
                out.append(
                    {
                        "bar": b,
                        "entry": None,
                        "settings_that_hit": len(hits),
                        "settings_retested": len(retested),
                        "settings_that_held": 0,
                        "why": why,
                    }
                )
                continue
            n = twice[0]
            r = self.row(n, kind)
            e = self.entries[n]
            both = r["n"] + r["fresh_n"]
            usd = (
                r["usd_per_task"] * r["n"] + (r.get("fresh_usd_per_task") or r["usd_per_task"]) * r["fresh_n"]
            ) / both
            out.append(
                {
                    "bar": b,
                    "entry": n,
                    "model": e.get("model", n),
                    "extra_body": e.get("extra_body") or {},
                    "right": r["right"],
                    "n": r["n"],
                    "fresh_right": r["fresh_right"],
                    "fresh_n": r["fresh_n"],
                    "usd_per_1000": round(usd * 1000, 4),
                    "settings_that_hit": len(hits),
                    "settings_retested": len(retested),
                    "settings_that_held": len(twice),
                }
            )
        return {"kind": kind, "label": self.label(kind), "as_of": self.as_of, "options": out}

    def ranking(self, kind: str) -> List[Dict[str, Any]]:
        """Every measured entry on one kind, cheapest first."""
        kind = self.resolve_kind(kind) or ""
        rows = []
        for n in self.entries:
            r = self.row(n, kind)
            if r and r.get("n"):
                rows.append({"entry": n, **r})
        return sorted(rows, key=lambda x: (x["usd_per_task"], x["entry"]))

    # ----- endpoints -----

    def endpoints(self) -> List[Endpoint]:
        """One LLM endpoint per measured entry, with the list price of the day of the run."""
        thinkers = {n[: -len("@think-off")] for n in self.entries if n.endswith("@think-off")}
        eps = []
        for name, e in self.entries.items():
            off = name.endswith("@think-off")
            thinks = (not off) and (name in thinkers or (e.get("mean_thinking_tokens") or 0) > 0)
            eps.append(
                Endpoint(
                    id=name,
                    name=name,
                    endpoint_type=EndpointType.LLM,
                    provider="openrouter",
                    model_id=e.get("model", name),
                    context_window=int(e.get("context_length") or 0) or 128000,
                    supports_thinking=thinks,
                    supports_vision=bool(e.get("supports_vision")),
                    cost_per_million_input=float(e.get("price_in_per_m") or 0.0),
                    cost_per_million_output=float(e.get("price_out_per_m") or 0.0),
                    tags=["measured"],
                    metadata={"extra_body": e.get("extra_body") or {}, "measured_as_of": self.as_of},
                )
            )
        return eps
