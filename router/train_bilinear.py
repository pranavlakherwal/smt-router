"""Train the BilinearPhiEngine on unified training data.

Usage:
    PYTHONPATH=. python -m router.train_bilinear
    PYTHONPATH=. python -m router.train_bilinear --epochs 10 --batch-size 128
    PYTHONPATH=. python -m router.train_bilinear --dry-run
"""

import argparse
import json
import logging
import time
from pathlib import Path

import torch

from router.learned_phi import BilinearPhiEngine, BilinearTrainer
from router.model_registry import ModelRegistry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent / "data"
DB_PATH = DATA_DIR / "bilinear_training.db"


def main():
    parser = argparse.ArgumentParser(description="Train BilinearPhiEngine")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr-phi", type=float, default=1e-4)
    parser.add_argument("--lr-embed", type=float, default=1e-3)
    parser.add_argument("--lr-weights", type=float, default=1e-3)
    parser.add_argument("--diversity-lambda", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--db-path", type=str, default=str(DB_PATH))
    parser.add_argument("--dry-run", action="store_true", help="Check data loading only")
    args = parser.parse_args()

    db_path = Path(args.db_path)
    if not db_path.exists():
        log.error(f"Training DB not found: {db_path}")
        log.error("Run data_pipeline.py first to create it.")
        return

    # Load model registry
    registry = ModelRegistry()
    model_to_idx = registry.export_for_training()
    log.info(f"Registry: {registry}")
    log.info(f"Model-to-index mapping: {len(model_to_idx)} models")

    # Create engine
    engine = BilinearPhiEngine(num_models=registry.num_models)

    if args.dry_run:
        log.info("\n=== Dry Run: Loading data only ===")
        from sentence_transformers import SentenceTransformer
        from router.learned_phi import build_precomputed_datasets

        prompt_encoder = SentenceTransformer("all-mpnet-base-v2")
        cache_path = DATA_DIR / "embedding_cache.pt"
        datasets = build_precomputed_datasets(
            db_path, prompt_encoder, model_to_idx, cache_path=cache_path,
        )
        for split, ds in datasets.items():
            log.info(f"  {split}: {len(ds):,} samples")
        log.info(f"Sample: emb_shape={datasets['train'][0][0].shape}, model_idx={datasets['train'][0][1].item()}, score={datasets['train'][0][2].item():.3f}")
        log.info("Dry run complete.")
        return

    # Training config
    config = {
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr_phi_matrices": args.lr_phi,
        "lr_model_embeddings": args.lr_embed,
        "lr_weights": args.lr_weights,
        "diversity_lambda": args.diversity_lambda,
        "patience": args.patience,
    }

    # Create trainer
    trainer = BilinearTrainer(
        engine=engine,
        db_path=db_path,
        model_to_idx=model_to_idx,
        config=config,
    )

    # Train
    start = time.time()
    history = trainer.train()
    elapsed = time.time() - start

    # Save history
    history["elapsed_sec"] = round(elapsed, 1)
    history["config"] = config
    history["num_models"] = registry.num_models
    history["num_aliases"] = registry.num_aliases

    history_path = DATA_DIR / "training_history.json"
    # Convert any non-serializable values
    def make_serializable(obj):
        if isinstance(obj, (torch.Tensor,)):
            return obj.item() if obj.numel() == 1 else obj.tolist()
        if isinstance(obj, float):
            return round(obj, 6)
        return obj

    serializable_history = {
        k: [make_serializable(v) for v in vals] if isinstance(vals, list) else make_serializable(vals)
        for k, vals in history.items()
    }

    history_path.write_text(json.dumps(serializable_history, indent=2))
    log.info(f"\nTraining history saved: {history_path}")
    log.info(f"Total time: {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    main()
