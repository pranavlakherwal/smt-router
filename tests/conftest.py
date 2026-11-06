"""Shared test fixtures. Every rule is tested on a made-up table, in a true world and a flipped one."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from router.measured import MeasuredTable  # noqa: E402


def row(right, usd, fresh_right=None, n=100):
    return {
        "right": right,
        "n": n,
        "usd_per_task": usd,
        "empty": 0,
        "cut": 0,
        "fresh_right": fresh_right,
        "fresh_n": n if fresh_right is not None else None,
        "fresh_usd_per_task": usd if fresh_right is not None else None,
    }


def make_table(rows):
    """rows: {entry: {kind: row}} -> a MeasuredTable. List prices are set dear for cheap-to-run
    entries on purpose, so a test can tell measured cost from list price."""
    kinds = sorted({k for e in rows.values() for k in e})
    entries = {
        name: {
            "model": name.split("@")[0],
            "extra_body": {},
            "price_in_per_m": 0.5,
            "price_out_per_m": 0.4,
            "context_length": 100000,
            "supports_vision": False,
            "mean_thinking_tokens": 0,
            "kinds": ks,
        }
        for name, ks in rows.items()
    }
    return MeasuredTable(table={"as_of": "2000-01-01", "kinds": {k: {"label": k} for k in kinds}, "entries": entries})


@pytest.fixture
def no_state(monkeypatch):
    monkeypatch.delenv("SMT_ROUTER_STATE_DIR", raising=False)
