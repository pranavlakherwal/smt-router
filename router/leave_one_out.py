"""Leave-one-out source ablation experiments.

For each of the 5 training data sources, trains a model excluding that source
and evaluates on held-out + standard benchmarks to measure:
  1. Per-source contribution to overall performance
  2. Cross-source generalization (can arena_55k data help on routerbench prompts?)
  3. Source complementarity (which pairs of sources overlap?)

Uses the pre-computed embedding cache from precompute_cache.py for fast iteration.
Each ablation run shares the same prompt embeddings, only filtering which rows
are included in training.

Usage:
    PYTHONPATH=. python -m router.leave_one_out
    PYTHONPATH=. python -m router.leave_one_out --source routerbench
    PYTHONPATH=. python -m router.leave_one_out --epochs 30 --patience 8
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from router.learned_phi import (
    BilinearPhiEngine,
    PrecomputedEmbeddingDataset,
)
from router.model_registry import ModelRegistry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent / "data"
DB_PATH = DATA_DIR / "bilinear_training.db"
CACHE_PATH = DATA_DIR / "embedding_cache.pt"
RESULTS_DIR = DATA_DIR / "leave_one_out"

ALL_SOURCES = ["routerbench", "routellm", "arena_55k", "mt_bench", "reward_bench"]


def load_source_filtered_data(
    db_path: Path,
    model_to_idx: dict,
    exclude_source: str | None = None,
    include_only_source: str | None = None,
) -> tuple[list[str], list[int], list[float]]:
    """Load rows from DB with optional source filtering.

    Args:
        exclude_source: If set, exclude all rows from this source (leave-one-out training).
        include_only_source: If set, include ONLY rows from this source (held-out evaluation).
    """
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    if exclude_source:
        rows = conn.execute(
            "SELECT prompt_text, canonical_model_id, score FROM tier1_prompts WHERE source != ?",
            (exclude_source,),
        ).fetchall()
    elif include_only_source:
        rows = conn.execute(
            "SELECT prompt_text, canonical_model_id, score FROM tier1_prompts WHERE source = ?",
            (include_only_source,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT prompt_text, canonical_model_id, score FROM tier1_prompts"
        ).fetchall()
    conn.close()

    prompts, model_idxs, score_vals = [], [], []
    skipped = 0
    for row in rows:
        mid = row["canonical_model_id"]
        if mid not in model_to_idx:
            skipped += 1
            continue
        prompts.append(str(row["prompt_text"])[:512])
        model_idxs.append(model_to_idx[mid])
        score_vals.append(float(row["score"]))

    if skipped:
        log.warning(f"Skipped {skipped} rows with unknown model IDs")
    return prompts, model_idxs, score_vals


def build_datasets_from_cache(
    cache_path: Path,
    db_path: Path,
    model_to_idx: dict,
    exclude_source: str | None = None,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, PrecomputedEmbeddingDataset]:
    """Build train/val/test datasets by filtering the full embedding cache by source.

    The embedding cache stores unique-prompt embeddings in the same order as
    dict.fromkeys(all_prompts). We look up each filtered row's prompt in that
    deduplication map to get the pre-computed embedding.

    This avoids re-encoding anything.
    """
    # Load the full unique-prompt embedding matrix
    log.info(f"Loading embedding cache from {cache_path}")
    cached = torch.load(str(cache_path), map_location="cpu", weights_only=False)

    # We need the original full prompt list to build the uid mapping.
    # Load ALL rows (unfiltered) to reconstruct the deduplication map.
    all_prompts_full, _, _ = load_source_filtered_data(db_path, model_to_idx)
    unique_prompts_full = list(dict.fromkeys(all_prompts_full))
    prompt_to_uid = {p: i for i, p in enumerate(unique_prompts_full)}

    # The cache was built with train/val/test splits over the FULL dataset.
    # We need the raw unique embeddings. Reconstruct them:
    # The cache has split_embeddings which are indexed copies.
    # Better approach: load from the precompute_cache output which stores
    # unique embeddings. But the cache format stores train/val/test splits.
    # We need to reconstruct the unique embedding matrix.

    # Since the cache stores split data, let's just load from DB + encode mapping.
    # Actually, the precompute_cache.py stores the full unique embeddings in the
    # cache before splitting. Let me check what keys are in the cache.

    cache_keys = list(cached.keys())
    log.info(f"Cache keys: {cache_keys}")

    # The cache from precompute_cache.py stores {split}_embeddings, {split}_model_indices, {split}_scores.
    # These are ALREADY split and index-expanded (not unique).
    # We need a different approach: reconstruct unique embeddings from the splits.

    # Concatenate all splits back together in the original order
    # Note: the splits were created with a deterministic shuffle (seed=42).
    # We can reconstruct the full dataset, then re-filter.
    all_emb = torch.cat([cached["train_embeddings"], cached["val_embeddings"], cached["test_embeddings"]], dim=0)
    all_mid = torch.cat([cached["train_model_indices"], cached["val_model_indices"], cached["test_model_indices"]], dim=0)
    all_scores = torch.cat([cached["train_scores"], cached["val_scores"], cached["test_scores"]], dim=0)
    n_total = all_emb.shape[0]
    log.info(f"Reconstructed full dataset: {n_total:,} samples")

    # The original splits were: shuffle(arange(N)) with seed=42, then split.
    # Reconstruct the original order by inverting the shuffle.
    rng = np.random.RandomState(42)
    original_indices = np.arange(n_total)
    rng.shuffle(original_indices)

    n_test_full = int(n_total * test_ratio)
    n_val_full = int(n_total * val_ratio)
    n_train_full = n_total - n_test_full - n_val_full

    # original_indices was used to CREATE the splits:
    # train = data[original_indices[:n_train]]
    # val = data[original_indices[n_train:n_train+n_val]]
    # test = data[original_indices[n_train+n_val:]]
    # So the concatenated data (train+val+test) is in shuffled order.
    # To get back to original DB order, we need to invert the permutation.
    inv_perm = np.zeros(n_total, dtype=np.int64)
    inv_perm[original_indices] = np.arange(n_total)

    # Restore original DB order
    db_order_emb = all_emb[inv_perm]
    db_order_mid = all_mid[inv_perm]
    db_order_scores = all_scores[inv_perm]

    # Now load the source labels for each row (in DB order)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT prompt_text, canonical_model_id, score, source FROM tier1_prompts"
    ).fetchall()
    conn.close()

    # Build source labels aligned with the embedding data (same filtering as original)
    source_labels = []
    for row in rows:
        mid = row["canonical_model_id"]
        if mid not in model_to_idx:
            continue
        source_labels.append(row["source"])

    assert len(source_labels) == n_total, (
        f"Source labels ({len(source_labels)}) != total samples ({n_total})"
    )

    # Filter by source
    if exclude_source:
        mask = np.array([s != exclude_source for s in source_labels])
        label = f"excluding {exclude_source}"
    else:
        mask = np.ones(n_total, dtype=bool)
        label = "all sources"

    filtered_emb = db_order_emb[mask]
    filtered_mid = db_order_mid[mask]
    filtered_scores = db_order_scores[mask]

    n_filtered = filtered_emb.shape[0]
    log.info(f"Filtered dataset ({label}): {n_filtered:,} / {n_total:,} samples")

    # Split the filtered data
    rng2 = np.random.RandomState(seed + hash(exclude_source or "") % 10000)
    indices = np.arange(n_filtered)
    rng2.shuffle(indices)

    n_test = int(n_filtered * test_ratio)
    n_val = int(n_filtered * val_ratio)

    splits = {
        "train": indices[:n_filtered - n_test - n_val],
        "val": indices[n_filtered - n_test - n_val:n_filtered - n_test],
        "test": indices[n_filtered - n_test:],
    }

    datasets = {}
    for split, sidx in splits.items():
        datasets[split] = PrecomputedEmbeddingDataset(
            filtered_emb[sidx],
            filtered_mid[sidx],
            filtered_scores[sidx],
        )
        log.info(f"  [{split}]: {len(datasets[split]):,} samples")

    return datasets


def build_heldout_dataset(
    cache_path: Path,
    db_path: Path,
    model_to_idx: dict,
    source: str,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> PrecomputedEmbeddingDataset:
    """Build a dataset containing ONLY rows from a specific source.

    Uses the same cache reconstruction approach to avoid re-encoding.
    """
    cached = torch.load(str(cache_path), map_location="cpu", weights_only=False)
    all_emb = torch.cat([cached["train_embeddings"], cached["val_embeddings"], cached["test_embeddings"]], dim=0)
    all_mid = torch.cat([cached["train_model_indices"], cached["val_model_indices"], cached["test_model_indices"]], dim=0)
    all_scores = torch.cat([cached["train_scores"], cached["val_scores"], cached["test_scores"]], dim=0)
    n_total = all_emb.shape[0]

    # Invert the shuffle to get DB order
    rng = np.random.RandomState(42)
    original_indices = np.arange(n_total)
    rng.shuffle(original_indices)
    inv_perm = np.zeros(n_total, dtype=np.int64)
    inv_perm[original_indices] = np.arange(n_total)

    db_order_emb = all_emb[inv_perm]
    db_order_mid = all_mid[inv_perm]
    db_order_scores = all_scores[inv_perm]

    # Get source labels
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT canonical_model_id, source FROM tier1_prompts"
    ).fetchall()
    conn.close()

    source_labels = []
    for row in rows:
        if row["canonical_model_id"] in model_to_idx:
            source_labels.append(row["source"])

    mask = np.array([s == source for s in source_labels])
    return PrecomputedEmbeddingDataset(
        db_order_emb[mask],
        db_order_mid[mask],
        db_order_scores[mask],
    )


def train_ablation(
    datasets: dict[str, PrecomputedEmbeddingDataset],
    num_models: int,
    config: dict,
    device: torch.device,
    checkpoint_path: Path,
) -> dict:
    """Train a BilinearPhiEngine and return training history."""
    engine = BilinearPhiEngine(num_models=num_models).to(device)

    train_loader = DataLoader(
        datasets["train"], batch_size=config["batch_size"], shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        datasets["val"], batch_size=config["batch_size"], shuffle=False, num_workers=0,
    )

    optimizer = torch.optim.AdamW([
        {"params": [engine.phi_matrices], "lr": config["lr_phi"]},
        {"params": [engine.model_embeddings.weight], "lr": config["lr_embed"]},
        {"params": [engine.phi_biases, engine.phi_logits], "lr": config["lr_weights"]},
    ], weight_decay=1e-4)

    total_steps = len(train_loader) * config["epochs"]
    warmup = 500

    def lr_lambda(step):
        if step < warmup:
            return step / max(warmup, 1)
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_loss = float("inf")
    patience_counter = 0
    global_step = 0
    best_state = None

    log.info(f"Training: {config['epochs']} epochs, {len(train_loader)} batches/epoch")

    for epoch in range(config["epochs"]):
        engine.train()
        epoch_loss = 0.0
        n_batches = 0

        for prompt_emb, model_idx, target_score in train_loader:
            prompt_emb = prompt_emb.to(device)
            model_idx = model_idx.to(device)
            target_score = target_score.to(device)
            optimizer.zero_grad()

            predicted = engine.predict_score(prompt_emb, model_idx)
            mse_loss = F.mse_loss(predicted, target_score)
            div_loss = engine.phi_diversity_loss()
            total_loss = mse_loss + config["diversity_lambda"] * div_loss

            total_loss.backward()
            optimizer.step()
            scheduler.step()

            # Norm clipping
            with torch.no_grad():
                norms = engine.model_embeddings.weight.norm(dim=1, keepdim=True)
                mask = norms > engine.norm_clip
                if mask.any():
                    engine.model_embeddings.weight.data = torch.where(
                        mask,
                        engine.model_embeddings.weight.data * (engine.norm_clip / norms),
                        engine.model_embeddings.weight.data,
                    )

            epoch_loss += mse_loss.item()
            n_batches += 1
            global_step += 1

        avg_train = epoch_loss / max(n_batches, 1)

        # Validation
        engine.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for prompt_emb, model_idx, target_score in val_loader:
                prompt_emb = prompt_emb.to(device)
                model_idx = model_idx.to(device)
                target_score = target_score.to(device)
                predicted = engine.predict_score(prompt_emb, model_idx)
                val_loss += F.mse_loss(predicted, target_score).item()
                val_batches += 1
        avg_val = val_loss / max(val_batches, 1)

        log.info(f"  Epoch {epoch+1:>3d}/{config['epochs']}  train={avg_train:.4f}  val={avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in engine.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                log.info(f"  Early stopping at epoch {epoch+1}")
                break

    # Restore best model
    if best_state:
        engine.load_state_dict(best_state)
        engine = engine.to(device)

    # Save checkpoint
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": engine.state_dict(),
        "num_models": num_models,
        "d_prompt": engine.d_prompt,
        "d_model": engine.d_model,
        "n_phi": engine.n_phi,
    }, checkpoint_path)

    # Evaluate on test split
    test_loader = DataLoader(
        datasets["test"], batch_size=config["batch_size"], shuffle=False, num_workers=0,
    )
    engine.eval()
    test_loss = 0.0
    test_batches = 0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for prompt_emb, model_idx, target_score in test_loader:
            prompt_emb = prompt_emb.to(device)
            model_idx = model_idx.to(device)
            target_score = target_score.to(device)
            predicted = engine.predict_score(prompt_emb, model_idx)
            test_loss += F.mse_loss(predicted, target_score).item()
            test_batches += 1
            all_preds.append(predicted.cpu())
            all_targets.append(target_score.cpu())

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    test_mse = test_loss / max(test_batches, 1)
    correlation = float(np.corrcoef(preds.numpy(), targets.numpy())[0, 1])

    return {
        "best_val_loss": float(best_val_loss),
        "test_mse": float(test_mse),
        "test_correlation": float(correlation),
        "epochs_trained": epoch + 1,
        "engine": engine,
    }


def eval_on_heldout(
    engine: BilinearPhiEngine,
    heldout_ds: PrecomputedEmbeddingDataset,
    device: torch.device,
    batch_size: int = 256,
) -> dict:
    """Evaluate a trained engine on a held-out source dataset."""
    if len(heldout_ds) == 0:
        return {"mse": None, "mae": None, "correlation": None, "n_samples": 0}

    loader = DataLoader(heldout_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    engine.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for prompt_emb, model_idx, target_score in loader:
            prompt_emb = prompt_emb.to(device)
            model_idx = model_idx.to(device)
            predicted = engine.predict_score(prompt_emb, model_idx)
            all_preds.append(predicted.cpu())
            all_targets.append(target_score)

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)

    mse = F.mse_loss(preds, targets).item()
    mae = (preds - targets).abs().mean().item()
    corr = float(np.corrcoef(preds.numpy(), targets.numpy())[0, 1])

    return {
        "mse": round(mse, 6),
        "mae": round(mae, 6),
        "correlation": round(corr, 4),
        "n_samples": len(heldout_ds),
    }


def main():
    parser = argparse.ArgumentParser(description="Leave-one-out source ablation")
    parser.add_argument("--source", type=str, default=None,
                        help="Run ablation for a single source only (default: all)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--lr-phi", type=float, default=1e-4)
    parser.add_argument("--lr-embed", type=float, default=1e-3)
    parser.add_argument("--lr-weights", type=float, default=1e-3)
    parser.add_argument("--diversity-lambda", type=float, default=0.01)
    args = parser.parse_args()

    if not CACHE_PATH.exists():
        log.error(f"Embedding cache not found: {CACHE_PATH}")
        log.error("Run precompute_cache.py first.")
        return
    if not DB_PATH.exists():
        log.error(f"Training DB not found: {DB_PATH}")
        return

    registry = ModelRegistry()
    model_to_idx = registry.export_for_training()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    log.info(f"Device: {device}, Models: {registry.num_models}")

    config = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "patience": args.patience,
        "lr_phi": args.lr_phi,
        "lr_embed": args.lr_embed,
        "lr_weights": args.lr_weights,
        "diversity_lambda": args.diversity_lambda,
    }

    sources_to_ablate = [args.source] if args.source else ALL_SOURCES
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for source in sources_to_ablate:
        log.info(f"\n{'='*60}")
        log.info(f"ABLATION: Excluding '{source}'")
        log.info(f"{'='*60}\n")

        t0 = time.time()

        # Build training data (exclude this source)
        datasets = build_datasets_from_cache(
            CACHE_PATH, DB_PATH, model_to_idx,
            exclude_source=source,
        )

        # Build held-out dataset (only this source)
        heldout = build_heldout_dataset(
            CACHE_PATH, DB_PATH, model_to_idx, source=source,
        )
        log.info(f"Held-out ({source}): {len(heldout):,} samples")

        # Train
        ckpt_path = RESULTS_DIR / f"ablation_no_{source}.pt"
        result = train_ablation(
            datasets, registry.num_models, config, device, ckpt_path,
        )
        engine = result.pop("engine")
        elapsed = time.time() - t0

        # Evaluate on held-out source
        heldout_metrics = eval_on_heldout(engine, heldout, device)

        # Evaluate on each OTHER source too (cross-source generalization)
        cross_source = {}
        for other_source in ALL_SOURCES:
            other_ds = build_heldout_dataset(
                CACHE_PATH, DB_PATH, model_to_idx, source=other_source,
            )
            cross_source[other_source] = eval_on_heldout(engine, other_ds, device)

        result.update({
            "excluded_source": source,
            "train_samples": len(datasets["train"]),
            "val_samples": len(datasets["val"]),
            "test_samples": len(datasets["test"]),
            "heldout_eval": heldout_metrics,
            "cross_source_eval": cross_source,
            "elapsed_sec": round(elapsed, 1),
        })

        all_results[source] = result
        log.info(f"\nResults for excluding '{source}':")
        log.info(f"  Test MSE (own split): {result['test_mse']:.4f}")
        log.info(f"  Test Correlation: {result['test_correlation']:.4f}")
        log.info(f"  Held-out MSE ({source}): {heldout_metrics['mse']}")
        log.info(f"  Held-out Correlation ({source}): {heldout_metrics['correlation']}")
        log.info(f"  Time: {elapsed/60:.1f} min")

        # Save incremental results
        results_path = RESULTS_DIR / "ablation_results.json"
        results_path.write_text(json.dumps(all_results, indent=2))

    # Print summary table
    log.info(f"\n{'='*80}")
    log.info("LEAVE-ONE-OUT ABLATION SUMMARY")
    log.info(f"{'='*80}")
    log.info(f"{'Excluded':<15} {'Train N':>10} {'Val MSE':>10} {'Test MSE':>10} "
             f"{'Heldout MSE':>12} {'Heldout r':>10}")
    log.info("-" * 80)
    for source, r in all_results.items():
        ho = r["heldout_eval"]
        log.info(
            f"{source:<15} {r['train_samples']:>10,} {r['best_val_loss']:>10.4f} "
            f"{r['test_mse']:>10.4f} {ho['mse'] or 'N/A':>12} "
            f"{ho['correlation'] or 'N/A':>10}"
        )

    log.info(f"\n{'='*80}")
    log.info("CROSS-SOURCE GENERALIZATION MATRIX")
    log.info(f"{'='*80}")
    header = "Excluded / Eval"
    log.info(f"{header:<15} " + " ".join(f"{s[:10]:>12}" for s in ALL_SOURCES))
    log.info("-" * 80)
    for excl, r in all_results.items():
        row = f"{excl:<15} "
        for eval_src in ALL_SOURCES:
            m = r["cross_source_eval"].get(eval_src, {})
            val = m.get("mse")
            if val is not None:
                row += f"{val:>12.4f} "
            else:
                row += f"{'N/A':>12} "
        log.info(row)

    results_path = RESULTS_DIR / "ablation_results.json"
    results_path.write_text(json.dumps(all_results, indent=2))
    log.info(f"\nResults saved: {results_path}")


if __name__ == "__main__":
    main()
