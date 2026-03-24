"""Unified data processing pipeline for bilinear phi training.

Converts all raw datasets into two canonical formats:
  Tier 1 (per-prompt):  (prompt_text, canonical_model_id, score, task_type, source)
  Tier 2 (aggregate):   (canonical_model_id, benchmark, score, metric_name, source)

All model IDs are resolved through the canonical ModelRegistry.
Output: SQLite database at data/bilinear_training.db

Usage:
    PYTHONPATH=. python -m router.data_pipeline
    PYTHONPATH=. python -m router.data_pipeline --dry-run
    PYTHONPATH=. python -m router.data_pipeline --dataset arena_55k
"""

import argparse
import json
import logging
import pickle
import sqlite3
import time
from pathlib import Path
from typing import Optional

import pandas as pd

from router.model_registry import ModelRegistry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
BILINEAR_DIR = RAW_DIR / "bilinear"
DB_PATH = DATA_DIR / "bilinear_training.db"

# RouterBench lives in a sibling repo
ROUTERBENCH_PKL = Path(__file__).parent.parent.parent.parent / "routerbench" / "data" / "hf_dataset" / "routerbench_raw.pkl"


def init_db(db_path: Path) -> sqlite3.Connection:
    """Create the unified training database schema."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    conn.executescript("""
        CREATE TABLE IF NOT EXISTS tier1_prompts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prompt_text TEXT NOT NULL,
            canonical_model_id TEXT NOT NULL,
            score REAL NOT NULL,
            task_type TEXT,
            source TEXT NOT NULL,
            source_id TEXT,
            UNIQUE(source, source_id, canonical_model_id)
        );

        CREATE TABLE IF NOT EXISTS tier2_aggregate (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            canonical_model_id TEXT NOT NULL,
            benchmark TEXT NOT NULL,
            score REAL NOT NULL,
            metric_name TEXT NOT NULL,
            source TEXT NOT NULL,
            UNIQUE(source, canonical_model_id, benchmark, metric_name)
        );

        CREATE TABLE IF NOT EXISTS unresolved_models (
            raw_model_id TEXT NOT NULL,
            source TEXT NOT NULL,
            count INTEGER DEFAULT 1,
            UNIQUE(raw_model_id, source)
        );

        CREATE TABLE IF NOT EXISTS pipeline_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,
            tier INTEGER NOT NULL,
            rows_processed INTEGER,
            rows_inserted INTEGER,
            rows_skipped_unresolved INTEGER,
            elapsed_sec REAL,
            timestamp TEXT DEFAULT (datetime('now'))
        );

        CREATE INDEX IF NOT EXISTS idx_t1_model ON tier1_prompts(canonical_model_id);
        CREATE INDEX IF NOT EXISTS idx_t1_source ON tier1_prompts(source);
        CREATE INDEX IF NOT EXISTS idx_t2_model ON tier2_aggregate(canonical_model_id);
        CREATE INDEX IF NOT EXISTS idx_t2_source ON tier2_aggregate(source);
    """)
    conn.commit()
    return conn


def ingest_routerbench(conn: sqlite3.Connection, registry: ModelRegistry, dry_run: bool = False) -> dict:
    """Ingest RouterBench raw data: per-prompt scores for 11 models."""
    source = "routerbench"
    log.info(f"[{source}] Loading {ROUTERBENCH_PKL}")

    if not ROUTERBENCH_PKL.exists():
        log.error(f"[{source}] File not found: {ROUTERBENCH_PKL}")
        return {"status": "error", "reason": "file_not_found"}

    start = time.time()
    df = pd.read_pickle(str(ROUTERBENCH_PKL))
    log.info(f"[{source}] Loaded {len(df):,} rows, {df['model_name'].nunique()} models")

    inserted = 0
    skipped = 0
    unresolved = {}

    rows_to_insert = []
    for _, row in df.iterrows():
        raw_model = row["model_name"]
        canonical = registry.resolve(raw_model)
        if canonical is None:
            unresolved[raw_model] = unresolved.get(raw_model, 0) + 1
            skipped += 1
            continue

        prompt_text = str(row["prompt"])[:10000]  # cap at 10K chars
        score = float(row["performance"])
        task_type = str(row["eval_name"])
        source_id = str(row["sample_id"])

        rows_to_insert.append((prompt_text, canonical, score, task_type, source, source_id))

    if not dry_run and rows_to_insert:
        conn.executemany(
            "INSERT OR IGNORE INTO tier1_prompts (prompt_text, canonical_model_id, score, task_type, source, source_id) VALUES (?,?,?,?,?,?)",
            rows_to_insert,
        )
        inserted = conn.total_changes
        conn.commit()
    elif dry_run:
        inserted = len(rows_to_insert)

    _log_unresolved(conn, unresolved, source, dry_run)
    elapsed = time.time() - start
    _log_pipeline(conn, source, 1, len(df), inserted, skipped, elapsed, dry_run)
    return {"inserted": inserted, "skipped": skipped, "unresolved": len(unresolved)}


def ingest_routellm(conn: sqlite3.Connection, registry: ModelRegistry, dry_run: bool = False) -> dict:
    """Ingest RouteLLM data: binary routing between gpt-4 and mixtral."""
    source = "routellm"
    train_path = RAW_DIR / "routellm" / "train.jsonl"

    if not train_path.exists():
        log.error(f"[{source}] File not found: {train_path}")
        return {"status": "error", "reason": "file_not_found"}

    start = time.time()
    log.info(f"[{source}] Loading {train_path}")

    # RouteLLM has fixed models: gpt-4 (strong) and mixtral (weak)
    gpt4_canonical = registry.resolve("gpt-4")
    mixtral_canonical = registry.resolve("mixtral")

    if not gpt4_canonical or not mixtral_canonical:
        log.error(f"[{source}] Cannot resolve gpt-4 or mixtral in registry")
        return {"status": "error", "reason": "model_resolution_failed"}

    inserted = 0
    skipped = 0
    total = 0
    rows_to_insert = []

    with open(train_path) as f:
        for line_num, line in enumerate(f):
            row = json.loads(line)
            total += 1
            prompt_text = str(row["prompt"])[:10000]
            mixtral_score = float(row["mixtral_score"])  # 1-5 scale

            # Normalize to 0-1: score 5 means mixtral matched gpt-4
            # We create entries for BOTH models from each row
            source_id = f"routellm_train_{line_num}"

            # Mixtral gets its raw score normalized to 0-1
            mixtral_norm = mixtral_score / 5.0
            rows_to_insert.append((prompt_text, mixtral_canonical, mixtral_norm, "general", source, f"{source_id}_mixtral"))

            # GPT-4 is the reference (always scores 1.0 in this dataset)
            rows_to_insert.append((prompt_text, gpt4_canonical, 1.0, "general", source, f"{source_id}_gpt4"))

    if not dry_run and rows_to_insert:
        conn.executemany(
            "INSERT OR IGNORE INTO tier1_prompts (prompt_text, canonical_model_id, score, task_type, source, source_id) VALUES (?,?,?,?,?,?)",
            rows_to_insert,
        )
        inserted = conn.total_changes
        conn.commit()
    elif dry_run:
        inserted = len(rows_to_insert)

    elapsed = time.time() - start
    _log_pipeline(conn, source, 1, total, inserted, skipped, elapsed, dry_run)
    return {"inserted": inserted, "skipped": skipped, "total_rows": total}


def ingest_arena_55k(conn: sqlite3.Connection, registry: ModelRegistry, dry_run: bool = False) -> dict:
    """Ingest LMSYS Arena 55K: pairwise preferences -> per-model scores."""
    source = "arena_55k"
    parquet_path = BILINEAR_DIR / "arena_55k" / "train.parquet"

    if not parquet_path.exists():
        log.error(f"[{source}] File not found: {parquet_path}")
        return {"status": "error", "reason": "file_not_found"}

    start = time.time()
    df = pd.read_parquet(str(parquet_path))
    log.info(f"[{source}] Loaded {len(df):,} pairwise comparisons")

    inserted = 0
    skipped = 0
    unresolved = {}
    rows_to_insert = []

    for idx, row in df.iterrows():
        raw_a = str(row["model_a"])
        raw_b = str(row["model_b"])
        canonical_a = registry.resolve(raw_a)
        canonical_b = registry.resolve(raw_b)

        if canonical_a is None:
            unresolved[raw_a] = unresolved.get(raw_a, 0) + 1
        if canonical_b is None:
            unresolved[raw_b] = unresolved.get(raw_b, 0) + 1

        if canonical_a is None and canonical_b is None:
            skipped += 1
            continue

        prompt_text = str(row["prompt"])[:10000]
        source_id = str(row.get("id", idx))

        # Convert pairwise to per-model scores:
        # winner gets 1.0, loser gets 0.0, tie gets 0.5
        win_a = bool(row["winner_model_a"])
        win_b = bool(row["winner_model_b"])
        is_tie = bool(row["winner_tie"])

        if is_tie:
            score_a, score_b = 0.5, 0.5
        elif win_a:
            score_a, score_b = 1.0, 0.0
        else:
            score_a, score_b = 0.0, 1.0

        if canonical_a:
            rows_to_insert.append((prompt_text, canonical_a, score_a, "general", source, f"{source_id}_a"))
        if canonical_b:
            rows_to_insert.append((prompt_text, canonical_b, score_b, "general", source, f"{source_id}_b"))

    if not dry_run and rows_to_insert:
        conn.executemany(
            "INSERT OR IGNORE INTO tier1_prompts (prompt_text, canonical_model_id, score, task_type, source, source_id) VALUES (?,?,?,?,?,?)",
            rows_to_insert,
        )
        inserted = conn.total_changes
        conn.commit()
    elif dry_run:
        inserted = len(rows_to_insert)

    _log_unresolved(conn, unresolved, source, dry_run)
    elapsed = time.time() - start
    _log_pipeline(conn, source, 1, len(df), inserted, skipped, elapsed, dry_run)
    return {"inserted": inserted, "skipped": skipped, "unresolved": len(unresolved)}


def ingest_mt_bench(conn: sqlite3.Connection, registry: ModelRegistry, dry_run: bool = False) -> dict:
    """Ingest MT-Bench human judgments: pairwise preferences."""
    source = "mt_bench"
    parquet_path = BILINEAR_DIR / "mt_bench" / "human.parquet"

    if not parquet_path.exists():
        log.error(f"[{source}] File not found: {parquet_path}")
        return {"status": "error", "reason": "file_not_found"}

    start = time.time()
    df = pd.read_parquet(str(parquet_path))
    log.info(f"[{source}] Loaded {len(df):,} human judgments")

    inserted = 0
    skipped = 0
    unresolved = {}
    rows_to_insert = []

    for idx, row in df.iterrows():
        raw_a = str(row["model_a"])
        raw_b = str(row["model_b"])
        canonical_a = registry.resolve(raw_a)
        canonical_b = registry.resolve(raw_b)

        if canonical_a is None:
            unresolved[raw_a] = unresolved.get(raw_a, 0) + 1
        if canonical_b is None:
            unresolved[raw_b] = unresolved.get(raw_b, 0) + 1

        if canonical_a is None and canonical_b is None:
            skipped += 1
            continue

        # MT-Bench has conversation_a/conversation_b, extract the prompt
        conv_a = row["conversation_a"]
        if isinstance(conv_a, list) and len(conv_a) > 0:
            first_turn = conv_a[0]
            if isinstance(first_turn, dict):
                prompt_text = str(first_turn.get("content", ""))[:10000]
            else:
                prompt_text = str(first_turn)[:10000]
        else:
            prompt_text = str(conv_a)[:10000]

        source_id = f"mt_{row['question_id']}_{row['turn']}_{idx}"
        winner = str(row["winner"])

        if winner == "model_a":
            score_a, score_b = 1.0, 0.0
        elif winner == "model_b":
            score_a, score_b = 0.0, 1.0
        else:  # tie
            score_a, score_b = 0.5, 0.5

        task_type = "multi_turn"
        if canonical_a:
            rows_to_insert.append((prompt_text, canonical_a, score_a, task_type, source, f"{source_id}_a"))
        if canonical_b:
            rows_to_insert.append((prompt_text, canonical_b, score_b, task_type, source, f"{source_id}_b"))

    if not dry_run and rows_to_insert:
        conn.executemany(
            "INSERT OR IGNORE INTO tier1_prompts (prompt_text, canonical_model_id, score, task_type, source, source_id) VALUES (?,?,?,?,?,?)",
            rows_to_insert,
        )
        inserted = conn.total_changes
        conn.commit()
    elif dry_run:
        inserted = len(rows_to_insert)

    _log_unresolved(conn, unresolved, source, dry_run)
    elapsed = time.time() - start
    _log_pipeline(conn, source, 1, len(df), inserted, skipped, elapsed, dry_run)
    return {"inserted": inserted, "skipped": skipped, "unresolved": len(unresolved)}


def ingest_reward_bench(conn: sqlite3.Connection, registry: ModelRegistry, dry_run: bool = False) -> dict:
    """Ingest RewardBench: chosen/rejected model pairs with named models."""
    source = "reward_bench"
    parquet_path = BILINEAR_DIR / "reward_bench" / "filtered.parquet"

    if not parquet_path.exists():
        log.error(f"[{source}] File not found: {parquet_path}")
        return {"status": "error", "reason": "file_not_found"}

    start = time.time()
    df = pd.read_parquet(str(parquet_path))
    log.info(f"[{source}] Loaded {len(df):,} rows")

    inserted = 0
    skipped = 0
    unresolved = {}
    rows_to_insert = []

    for idx, row in df.iterrows():
        raw_chosen = str(row["chosen_model"])
        raw_rejected = str(row["rejected_model"])
        canonical_chosen = registry.resolve(raw_chosen)
        canonical_rejected = registry.resolve(raw_rejected)

        if canonical_chosen is None:
            unresolved[raw_chosen] = unresolved.get(raw_chosen, 0) + 1
        if canonical_rejected is None:
            unresolved[raw_rejected] = unresolved.get(raw_rejected, 0) + 1

        if canonical_chosen is None and canonical_rejected is None:
            skipped += 1
            continue

        prompt_text = str(row["prompt"])[:10000]
        task_type = str(row.get("subset", "general"))
        source_id = str(row.get("id", idx))

        # Chosen = 1.0, Rejected = 0.0
        if canonical_chosen:
            rows_to_insert.append((prompt_text, canonical_chosen, 1.0, task_type, source, f"{source_id}_chosen"))
        if canonical_rejected:
            rows_to_insert.append((prompt_text, canonical_rejected, 0.0, task_type, source, f"{source_id}_rejected"))

    if not dry_run and rows_to_insert:
        conn.executemany(
            "INSERT OR IGNORE INTO tier1_prompts (prompt_text, canonical_model_id, score, task_type, source, source_id) VALUES (?,?,?,?,?,?)",
            rows_to_insert,
        )
        inserted = conn.total_changes
        conn.commit()
    elif dry_run:
        inserted = len(rows_to_insert)

    _log_unresolved(conn, unresolved, source, dry_run)
    elapsed = time.time() - start
    _log_pipeline(conn, source, 1, len(df), inserted, skipped, elapsed, dry_run)
    return {"inserted": inserted, "skipped": skipped, "unresolved": len(unresolved)}


def ingest_ultrafeedback(conn: sqlite3.Connection, registry: ModelRegistry, dry_run: bool = False) -> dict:
    """Ingest UltraFeedback: quality scores (no model names, but has score_chosen/score_rejected).

    UltraFeedback doesn't have explicit model names in the binarized version.
    We treat it as: each prompt has a quality score pair indicating the score gap.
    Since we don't know which model generated each response, we use this as
    a prompt difficulty signal, not a per-model training signal.

    Strategy: skip for Tier 1 (no model IDs). Could use for prompt encoding pre-training later.
    """
    source = "ultrafeedback"
    log.info(f"[{source}] SKIPPED: No model identity columns in binarized version")
    log.info(f"[{source}] Has score_chosen/score_rejected but no model names")
    log.info(f"[{source}] Could be used for prompt difficulty pre-training in future")
    return {"inserted": 0, "skipped": 0, "reason": "no_model_ids"}


def ingest_arena_hard(conn: sqlite3.Connection, registry: ModelRegistry, dry_run: bool = False) -> dict:
    """Ingest Arena-Hard: challenging queries (prompts only, judgments need separate download).

    Arena-Hard v0.1 on HuggingFace contains the 500 challenge questions and categories,
    but per-model judgment data requires running the evaluation pipeline.
    We store the prompts for potential use as hard-prompt augmentation.
    """
    source = "arena_hard"
    log.info(f"[{source}] SKIPPED: Contains prompts only, no per-model judgments in HF dataset")
    log.info(f"[{source}] 500 challenge queries available for prompt augmentation")
    return {"inserted": 0, "skipped": 0, "reason": "prompts_only"}


def ingest_bigcodebench_aggregate(conn: sqlite3.Connection, registry: ModelRegistry, dry_run: bool = False) -> dict:
    """Ingest BigCodeBench as Tier 2 aggregate: task definitions for code benchmarking.

    BigCodeBench contains 1,140 code tasks. Per-model pass rates are published
    on their leaderboard but not in the HF dataset. We store task metadata
    as Tier 2 aggregate once we have model scores.
    """
    source = "bigcodebench"
    log.info(f"[{source}] SKIPPED: HF dataset has tasks, not per-model scores")
    log.info(f"[{source}] 1,140 code tasks available, need leaderboard scrape for model scores")
    return {"inserted": 0, "skipped": 0, "reason": "tasks_only"}


def ingest_mixeval_aggregate(conn: sqlite3.Connection, registry: ModelRegistry, dry_run: bool = False) -> dict:
    """Ingest MixEval as Tier 2 aggregate: evaluation prompts with ground truth.

    MixEval contains prompts + ground truth answers. Per-model scores are on
    their leaderboard. We store task metadata.
    """
    source = "mixeval"
    log.info(f"[{source}] SKIPPED: HF dataset has questions + ground truth, not per-model scores")
    log.info(f"[{source}] 4,000 questions available, need leaderboard scrape for model scores")
    return {"inserted": 0, "skipped": 0, "reason": "questions_only"}


def _log_unresolved(conn: sqlite3.Connection, unresolved: dict, source: str, dry_run: bool):
    """Log unresolved model IDs."""
    if not unresolved:
        return
    log.warning(f"[{source}] {len(unresolved)} unresolved model IDs ({sum(unresolved.values())} total rows)")
    for model_id, count in sorted(unresolved.items(), key=lambda x: -x[1])[:10]:
        log.warning(f"  {model_id}: {count:,} rows")

    if not dry_run:
        for model_id, count in unresolved.items():
            conn.execute(
                "INSERT OR REPLACE INTO unresolved_models (raw_model_id, source, count) VALUES (?,?,?)",
                (model_id, source, count),
            )
        conn.commit()


def _log_pipeline(conn: sqlite3.Connection, source: str, tier: int, processed: int,
                  inserted: int, skipped: int, elapsed: float, dry_run: bool):
    """Log pipeline execution."""
    log.info(f"[{source}] T{tier}: {processed:,} processed, {inserted:,} inserted, "
             f"{skipped:,} skipped (unresolved), {elapsed:.1f}s")
    if not dry_run:
        conn.execute(
            "INSERT INTO pipeline_log (source, tier, rows_processed, rows_inserted, rows_skipped_unresolved, elapsed_sec) VALUES (?,?,?,?,?,?)",
            (source, tier, processed, inserted, skipped, round(elapsed, 2)),
        )
        conn.commit()


# Map of dataset name -> ingest function
INGEST_MAP = {
    "routerbench": ingest_routerbench,
    "routellm": ingest_routellm,
    "arena_55k": ingest_arena_55k,
    "mt_bench": ingest_mt_bench,
    "reward_bench": ingest_reward_bench,
    "ultrafeedback": ingest_ultrafeedback,
    "arena_hard": ingest_arena_hard,
    "bigcodebench": ingest_bigcodebench_aggregate,
    "mixeval": ingest_mixeval_aggregate,
}


def main():
    parser = argparse.ArgumentParser(description="Build unified bilinear phi training database")
    parser.add_argument("--dataset", type=str, help="Ingest a single dataset")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    parser.add_argument("--db-path", type=str, default=str(DB_PATH), help="Output DB path")
    parser.add_argument("--reset", action="store_true", help="Drop and recreate all tables")
    args = parser.parse_args()

    db_path = Path(args.db_path)

    if args.reset and db_path.exists():
        log.warning(f"Resetting database: {db_path}")
        db_path.unlink()

    registry = ModelRegistry()
    log.info(f"Registry: {registry}")

    conn = init_db(db_path)
    log.info(f"Database: {db_path}")

    # Select datasets to ingest
    if args.dataset:
        if args.dataset not in INGEST_MAP:
            log.error(f"Unknown dataset: {args.dataset}. Available: {list(INGEST_MAP.keys())}")
            return
        targets = {args.dataset: INGEST_MAP[args.dataset]}
    else:
        targets = INGEST_MAP

    log.info(f"\n=== Bilinear Phi Data Pipeline ===")
    log.info(f"Datasets: {list(targets.keys())}")
    log.info("")

    results = {}
    for name, ingest_fn in targets.items():
        results[name] = ingest_fn(conn, registry, dry_run=args.dry_run)
        log.info("")

    # Summary
    log.info("=== Pipeline Summary ===")
    total_inserted = sum(r.get("inserted", 0) for r in results.values())
    total_skipped = sum(r.get("skipped", 0) for r in results.values())

    for name, r in results.items():
        status = "OK" if r.get("inserted", 0) > 0 else r.get("reason", "no_data")
        log.info(f"  {name:20s} inserted={r.get('inserted', 0):>8,}  skipped={r.get('skipped', 0):>6,}  [{status}]")

    log.info(f"\n  TOTAL: {total_inserted:,} inserted, {total_skipped:,} skipped")

    # Show DB stats
    if not args.dry_run:
        t1_count = conn.execute("SELECT COUNT(*) FROM tier1_prompts").fetchone()[0]
        t1_models = conn.execute("SELECT COUNT(DISTINCT canonical_model_id) FROM tier1_prompts").fetchone()[0]
        t1_sources = conn.execute("SELECT COUNT(DISTINCT source) FROM tier1_prompts").fetchone()[0]
        unresolved_count = conn.execute("SELECT COUNT(*) FROM unresolved_models").fetchone()[0]

        log.info(f"\n=== Database Stats ===")
        log.info(f"  Tier 1 rows: {t1_count:,}")
        log.info(f"  Tier 1 models: {t1_models}")
        log.info(f"  Tier 1 sources: {t1_sources}")
        log.info(f"  Unresolved model IDs: {unresolved_count}")

        # Per-model counts
        log.info(f"\n=== Per-Model Row Counts ===")
        rows = conn.execute(
            "SELECT canonical_model_id, COUNT(*) as cnt FROM tier1_prompts GROUP BY canonical_model_id ORDER BY cnt DESC"
        ).fetchall()
        for model_id, cnt in rows:
            log.info(f"  {model_id:45s} {cnt:>8,}")

    conn.close()
    log.info(f"\nDone. Database: {db_path}")


if __name__ == "__main__":
    main()
