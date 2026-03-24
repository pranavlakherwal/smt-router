"""Bilinear Phi Engine for S(M,T) routing.

Implements learned bilinear phi functions:
  phi_i(M,T) = sigmoid(e_T^T @ W_i @ e_M + b_i)

where:
  e_T = frozen sentence-transformers encoder (768-dim)
  e_M = learned model embedding (256-dim)
  W_i = learned bilinear matrix (768 x 256)
  b_i = learned scalar bias

Architecture:
  - 16 bilinear phi heads
  - Frozen prompt encoder (all-mpnet-base-v2)
  - Learnable model embeddings (nn.Embedding)
  - ~3.15M trainable parameters

Usage:
    engine = BilinearPhiEngine(num_models=48)
    phi_scores = engine(prompt_text="Explain quantum computing", model_idx=3)
    # phi_scores: (16,) tensor of phi values in [0, 1]
"""

import logging
import math
import sqlite3
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)

# Defaults from architecture plan
D_PROMPT = 768   # all-mpnet-base-v2 output dim
D_MODEL = 256    # model embedding dim
N_PHI = 16       # number of phi heads
NORM_CLIP = 10.0 # max L2 norm for model embeddings


class BilinearPhiEngine(nn.Module):
    """16 learned bilinear phi functions with model embeddings.

    Each phi head computes:
      phi_i(prompt, model) = sigmoid(e_T^T @ W_i @ e_M + b_i)

    The prompt encoder is frozen (all-mpnet-base-v2).
    Model embeddings and bilinear matrices are learned.
    """

    def __init__(
        self,
        num_models: int,
        d_prompt: int = D_PROMPT,
        d_model: int = D_MODEL,
        n_phi: int = N_PHI,
        norm_clip: float = NORM_CLIP,
    ):
        super().__init__()
        self.num_models = num_models
        self.d_prompt = d_prompt
        self.d_model = d_model
        self.n_phi = n_phi
        self.norm_clip = norm_clip

        # Model embeddings: each model gets a d_model-dim vector
        self.model_embeddings = nn.Embedding(num_models, d_model)
        nn.init.normal_(self.model_embeddings.weight, mean=0.0, std=0.1)

        # Bilinear matrices: W_i of shape (d_prompt, d_model) for each phi head
        # Use a single 3D parameter for efficiency: (n_phi, d_prompt, d_model)
        self.phi_matrices = nn.Parameter(
            torch.randn(n_phi, d_prompt, d_model) * (2.0 / math.sqrt(d_prompt + d_model))
        )

        # Biases: one per phi head
        self.phi_biases = nn.Parameter(torch.zeros(n_phi))

        # Phi weights: learned weights for combining phi scores (softmax-normalized)
        self.phi_logits = nn.Parameter(torch.zeros(n_phi))

        logger.info(
            f"BilinearPhiEngine: {num_models} models, "
            f"{n_phi} phi heads, "
            f"{self._count_params():,} trainable params"
        )

    def _count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        prompt_embeddings: torch.Tensor,
        model_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Compute phi scores for a batch of (prompt, model) pairs.

        Args:
            prompt_embeddings: (batch, d_prompt) pre-computed prompt embeddings
            model_indices: (batch,) integer model indices

        Returns:
            phi_scores: (batch, n_phi) tensor of phi values in [0, 1]
        """
        # Get model embeddings: (batch, d_model)
        e_M = self.model_embeddings(model_indices)

        # Clip embedding norms (anti-stuck safeguard #1)
        with torch.no_grad():
            norms = e_M.norm(dim=1, keepdim=True)
            mask = norms > self.norm_clip
            if mask.any():
                e_M = torch.where(mask, e_M * (self.norm_clip / norms), e_M)

        # Bilinear product: e_T^T @ W_i @ e_M for all phi heads at once
        # prompt_embeddings: (batch, d_prompt)
        # phi_matrices: (n_phi, d_prompt, d_model)
        # e_M: (batch, d_model)

        # Step 1: e_T @ W_i -> (batch, n_phi, d_model)
        # einsum: b=batch, p=d_prompt, n=n_phi, m=d_model
        intermediate = torch.einsum("bp,npm->bnm", prompt_embeddings, self.phi_matrices)

        # Step 2: intermediate @ e_M -> (batch, n_phi)
        # einsum: b=batch, n=n_phi, m=d_model
        logits = torch.einsum("bnm,bm->bn", intermediate, e_M)

        # Add biases: (n_phi,) broadcast to (batch, n_phi)
        logits = logits + self.phi_biases.unsqueeze(0)

        # Sigmoid to get phi scores in [0, 1]
        phi_scores = torch.sigmoid(logits)

        return phi_scores

    def predict_score(
        self,
        prompt_embeddings: torch.Tensor,
        model_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Predict overall quality score from phi scores.

        Returns:
            scores: (batch,) predicted quality scores in [0, 1]
        """
        phi_scores = self.forward(prompt_embeddings, model_indices)

        # Softmax-normalized phi weights
        weights = F.softmax(self.phi_logits, dim=0)  # (n_phi,)

        # Weighted sum: (batch, n_phi) @ (n_phi,) -> (batch,)
        scores = (phi_scores * weights.unsqueeze(0)).sum(dim=1)

        return scores

    def phi_diversity_loss(self) -> torch.Tensor:
        """Regularizer: penalize cosine similarity between phi matrices.

        Anti-stuck safeguard #2: prevents all phi heads from converging
        to the same function.
        """
        # Flatten each phi matrix to a vector: (n_phi, d_prompt * d_model)
        flat = self.phi_matrices.view(self.n_phi, -1)

        # Normalize
        flat_norm = F.normalize(flat, dim=1)

        # Pairwise cosine similarity matrix
        sim = flat_norm @ flat_norm.T  # (n_phi, n_phi)

        # Penalize off-diagonal elements (want them close to 0)
        mask = ~torch.eye(self.n_phi, dtype=torch.bool, device=sim.device)
        diversity_loss = sim[mask].pow(2).mean()

        return diversity_loss

    def get_model_embedding(self, model_idx: int) -> torch.Tensor:
        """Get a single model's embedding vector."""
        idx = torch.tensor([model_idx], device=self.model_embeddings.weight.device)
        return self.model_embeddings(idx).squeeze(0)

    def init_embeddings_from_pca(self, benchmark_scores: np.ndarray):
        """Initialize model embeddings from PCA of benchmark scores.

        Anti-stuck safeguard #4: warm-start from aggregate data structure.

        Args:
            benchmark_scores: (num_models, num_benchmarks) array of scores.
                              Rows with all NaN are initialized randomly.
        """
        from sklearn.decomposition import PCA

        valid_mask = ~np.isnan(benchmark_scores).all(axis=1)
        valid_scores = benchmark_scores[valid_mask]

        # Fill remaining NaN with column means
        col_means = np.nanmean(valid_scores, axis=0)
        for j in range(valid_scores.shape[1]):
            nan_mask = np.isnan(valid_scores[:, j])
            valid_scores[nan_mask, j] = col_means[j]

        n_components = min(self.d_model, valid_scores.shape[1], valid_scores.shape[0])
        pca = PCA(n_components=n_components)
        reduced = pca.fit_transform(valid_scores)

        # Pad to d_model if needed
        if reduced.shape[1] < self.d_model:
            padding = np.random.randn(reduced.shape[0], self.d_model - reduced.shape[1]) * 0.01
            reduced = np.hstack([reduced, padding])

        # Scale to reasonable range
        reduced = reduced * 0.1

        with torch.no_grad():
            valid_indices = np.where(valid_mask)[0]
            for i, idx in enumerate(valid_indices):
                self.model_embeddings.weight[idx] = torch.from_numpy(reduced[i]).float()

        logger.info(
            f"PCA warm-start: initialized {len(valid_indices)}/{self.num_models} "
            f"model embeddings from {valid_scores.shape[1]} benchmarks "
            f"({n_components} components, {pca.explained_variance_ratio_.sum():.1%} variance)"
        )


class PrecomputedEmbeddingDataset(Dataset):
    """Dataset with pre-computed prompt embeddings for fast training.

    Pre-encodes all prompts once, then training batches are pure tensor ops.
    """

    def __init__(
        self,
        prompt_embeddings: torch.Tensor,
        model_indices: torch.Tensor,
        scores: torch.Tensor,
    ):
        assert len(prompt_embeddings) == len(model_indices) == len(scores)
        self.prompt_embeddings = prompt_embeddings
        self.model_indices = model_indices
        self.scores = scores

    def __len__(self):
        return len(self.scores)

    def __getitem__(self, idx):
        return (
            self.prompt_embeddings[idx],
            self.model_indices[idx],
            self.scores[idx],
        )


def build_precomputed_datasets(
    db_path: Path,
    prompt_encoder,
    model_to_idx: dict[str, int],
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    max_prompt_length: int = 512,
    encode_batch_size: int = 256,
    cache_path: Optional[Path] = None,
) -> dict[str, PrecomputedEmbeddingDataset]:
    """Load data from SQLite, encode all prompts, split into train/val/test.

    Returns dict with keys 'train', 'val', 'test'.
    Caches encoded embeddings to disk for fast reload.
    """
    # Check cache first
    if cache_path and cache_path.exists():
        logger.info(f"Loading cached embeddings from {cache_path}")
        cached = torch.load(str(cache_path), map_location="cpu", weights_only=False)
        datasets = {}
        for split in ["train", "val", "test"]:
            datasets[split] = PrecomputedEmbeddingDataset(
                cached[f"{split}_embeddings"],
                cached[f"{split}_model_indices"],
                cached[f"{split}_scores"],
            )
            logger.info(f"  [{split}]: {len(datasets[split]):,} samples")
        return datasets

    # Load from SQLite
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT prompt_text, canonical_model_id, score FROM tier1_prompts"
    ).fetchall()
    conn.close()

    # Filter and collect
    prompts = []
    model_idxs = []
    score_vals = []
    skipped = 0

    for row in rows:
        model_id = row["canonical_model_id"]
        if model_id not in model_to_idx:
            skipped += 1
            continue
        prompts.append(str(row["prompt_text"])[:max_prompt_length])
        model_idxs.append(model_to_idx[model_id])
        score_vals.append(float(row["score"]))

    if skipped > 0:
        logger.warning(f"Skipped {skipped} rows with unknown model IDs")
    logger.info(f"Loaded {len(prompts):,} samples from DB")

    # Deduplicate prompts before encoding (many prompts appear across multiple models)
    unique_prompts = list(dict.fromkeys(prompts))  # preserves order
    prompt_to_uid = {p: i for i, p in enumerate(unique_prompts)}
    prompt_uid_per_row = [prompt_to_uid[p] for p in prompts]
    logger.info(
        f"Deduplication: {len(unique_prompts):,} unique prompts "
        f"from {len(prompts):,} total ({len(unique_prompts)/len(prompts):.1%})"
    )

    # Encode only unique prompts
    logger.info(f"Encoding {len(unique_prompts):,} unique prompts (batch_size={encode_batch_size})...")
    with torch.no_grad():
        unique_embeddings = prompt_encoder.encode(
            unique_prompts,
            batch_size=encode_batch_size,
            convert_to_tensor=True,
            show_progress_bar=True,
        ).cpu().clone()  # (N_unique, 768)
    logger.info(f"Unique embeddings shape: {unique_embeddings.shape}")

    # Map back to full dataset
    uid_indices = torch.tensor(prompt_uid_per_row, dtype=torch.long)
    all_embeddings = unique_embeddings[uid_indices]  # (N_total, 768)
    model_indices = torch.tensor(model_idxs, dtype=torch.long)
    scores = torch.tensor(score_vals, dtype=torch.float32)
    logger.info(f"Full embeddings shape: {all_embeddings.shape}")

    # Deterministic split
    rng = np.random.RandomState(seed)
    indices = np.arange(len(scores))
    rng.shuffle(indices)

    n_test = int(len(indices) * test_ratio)
    n_val = int(len(indices) * val_ratio)
    n_train = len(indices) - n_test - n_val

    splits = {
        "train": indices[:n_train],
        "val": indices[n_train:n_train + n_val],
        "test": indices[n_train + n_val:],
    }

    datasets = {}
    cache_data = {}
    for split, split_indices in splits.items():
        emb = all_embeddings[split_indices]
        midx = model_indices[split_indices]
        sc = scores[split_indices]
        datasets[split] = PrecomputedEmbeddingDataset(emb, midx, sc)
        cache_data[f"{split}_embeddings"] = emb
        cache_data[f"{split}_model_indices"] = midx
        cache_data[f"{split}_scores"] = sc
        logger.info(f"  [{split}]: {len(split_indices):,} samples")

    # Save cache
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(cache_data, str(cache_path))
        size_mb = cache_path.stat().st_size / 1e6
        logger.info(f"Saved embedding cache: {cache_path} ({size_mb:.0f}MB)")

    return datasets


class BilinearTrainer:
    """Training loop for BilinearPhiEngine.

    Implements:
      - Prompt pathway training (Tier 1 per-prompt data)
      - Anti-stuck safeguards (diversity loss, gradient monitoring, norm clipping)
      - Per-component learning rates
      - Cosine warmup scheduler
      - Early stopping
    """

    def __init__(
        self,
        engine: BilinearPhiEngine,
        db_path: Path,
        model_to_idx: dict[str, int],
        config: Optional[dict] = None,
    ):
        self.engine = engine
        self.db_path = db_path
        self.model_to_idx = model_to_idx
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        # Default config from architecture plan
        self.config = {
            "batch_size": 64,
            "epochs": 50,
            "lr_phi_matrices": 1e-4,
            "lr_model_embeddings": 1e-3,
            "lr_weights": 1e-3,
            "weight_decay": 1e-4,
            "diversity_lambda": 0.01,
            "val_split": 0.1,
            "test_split": 0.1,
            "patience": 10,
            "warmup_steps": 500,
            "log_every": 100,
            "gradient_alert_threshold": 1e-7,
            "gradient_alert_window": 100,
        }
        if config:
            self.config.update(config)

        self.engine = self.engine.to(self.device)
        logger.info(f"Trainer initialized on device: {self.device}")
        logger.info(f"Config: {self.config}")

    def train(self) -> dict:
        """Run the full training loop. Returns training history."""
        cfg = self.config

        # Load prompt encoder for embedding pre-computation
        from sentence_transformers import SentenceTransformer
        prompt_encoder = SentenceTransformer("all-mpnet-base-v2")
        prompt_encoder.eval()

        # Pre-compute all embeddings (one-time cost, then training is fast)
        cache_path = self.db_path.parent / "embedding_cache.pt"
        datasets = build_precomputed_datasets(
            self.db_path,
            prompt_encoder,
            self.model_to_idx,
            val_ratio=cfg["val_split"],
            test_ratio=cfg["test_split"],
            cache_path=cache_path,
        )

        # Free the encoder after pre-computation
        del prompt_encoder
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        device = self.device

        def to_device(batch):
            return (b.to(device) for b in batch)

        train_loader = DataLoader(
            datasets["train"], batch_size=cfg["batch_size"], shuffle=True, num_workers=0,
        )
        val_loader = DataLoader(
            datasets["val"], batch_size=cfg["batch_size"], shuffle=False, num_workers=0,
        )

        # Per-component learning rates (anti-stuck safeguard #5)
        optimizer = torch.optim.AdamW([
            {"params": [self.engine.phi_matrices], "lr": cfg["lr_phi_matrices"]},
            {"params": [self.engine.model_embeddings.weight], "lr": cfg["lr_model_embeddings"]},
            {"params": [self.engine.phi_biases, self.engine.phi_logits], "lr": cfg["lr_weights"]},
        ], weight_decay=cfg["weight_decay"])

        # Cosine warmup scheduler
        total_steps = len(train_loader) * cfg["epochs"]
        warmup = cfg["warmup_steps"]

        def lr_lambda(step):
            if step < warmup:
                return step / max(warmup, 1)
            progress = (step - warmup) / max(total_steps - warmup, 1)
            return 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        # Training loop
        history = {"train_loss": [], "val_loss": [], "diversity_loss": []}
        best_val_loss = float("inf")
        patience_counter = 0
        global_step = 0
        gradient_low_count = {i: 0 for i in range(self.engine.n_phi)}

        logger.info(f"\n=== Training: {cfg['epochs']} epochs, {len(train_loader)} batches/epoch ===\n")

        for epoch in range(cfg["epochs"]):
            self.engine.train()
            epoch_loss = 0.0
            epoch_diversity = 0.0
            n_batches = 0

            for prompt_emb, model_idx, target_score in train_loader:
                prompt_emb = prompt_emb.to(device)
                model_idx = model_idx.to(device)
                target_score = target_score.to(device)
                optimizer.zero_grad()

                # Forward pass
                predicted = self.engine.predict_score(prompt_emb, model_idx)
                mse_loss = F.mse_loss(predicted, target_score)

                # Diversity regularizer
                div_loss = self.engine.phi_diversity_loss()
                total_loss = mse_loss + cfg["diversity_lambda"] * div_loss

                # Backward pass
                total_loss.backward()

                # Gradient monitoring (anti-stuck safeguard #3)
                if global_step % cfg["log_every"] == 0:
                    for i in range(self.engine.n_phi):
                        grad_norm = self.engine.phi_matrices.grad[i].norm().item()
                        if grad_norm < cfg["gradient_alert_threshold"]:
                            gradient_low_count[i] += 1
                            if gradient_low_count[i] >= cfg["gradient_alert_window"]:
                                logger.warning(
                                    f"ALERT: phi_{i} gradient norm < {cfg['gradient_alert_threshold']} "
                                    f"for {gradient_low_count[i]} consecutive checks"
                                )
                        else:
                            gradient_low_count[i] = 0

                optimizer.step()
                scheduler.step()

                # Norm clipping on model embeddings (applied after optimizer step)
                with torch.no_grad():
                    norms = self.engine.model_embeddings.weight.norm(dim=1, keepdim=True)
                    mask = norms > self.engine.norm_clip
                    if mask.any():
                        self.engine.model_embeddings.weight.data = torch.where(
                            mask,
                            self.engine.model_embeddings.weight.data * (self.engine.norm_clip / norms),
                            self.engine.model_embeddings.weight.data,
                        )

                epoch_loss += mse_loss.item()
                epoch_diversity += div_loss.item()
                n_batches += 1
                global_step += 1

                if global_step % cfg["log_every"] == 0:
                    lr_current = scheduler.get_last_lr()[0]
                    logger.info(
                        f"  step {global_step:>6d}  loss={mse_loss.item():.4f}  "
                        f"div={div_loss.item():.4f}  lr={lr_current:.2e}"
                    )

            avg_train_loss = epoch_loss / max(n_batches, 1)
            avg_diversity = epoch_diversity / max(n_batches, 1)

            # Validation
            val_loss = self._validate(val_loader)

            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(val_loss)
            history["diversity_loss"].append(avg_diversity)

            logger.info(
                f"Epoch {epoch+1:>3d}/{cfg['epochs']}  "
                f"train_loss={avg_train_loss:.4f}  val_loss={val_loss:.4f}  "
                f"diversity={avg_diversity:.4f}"
            )

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                self._save_checkpoint("best")
            else:
                patience_counter += 1
                if patience_counter >= cfg["patience"]:
                    logger.info(f"Early stopping at epoch {epoch+1} (patience={cfg['patience']})")
                    break

        # Load best model
        self._load_checkpoint("best")

        # Final test evaluation
        test_loader = DataLoader(
            datasets["test"], batch_size=cfg["batch_size"], shuffle=False, num_workers=0,
        )
        test_loss = self._validate(test_loader)

        history["test_loss"] = test_loss
        history["best_val_loss"] = best_val_loss
        history["epochs_trained"] = epoch + 1
        history["total_steps"] = global_step

        logger.info(f"\n=== Training Complete ===")
        logger.info(f"Best val loss: {best_val_loss:.4f}")
        logger.info(f"Test loss: {test_loss:.4f}")
        logger.info(f"Epochs: {epoch+1}, Steps: {global_step}")

        return history

    def _validate(self, loader) -> float:
        """Compute average loss on a validation/test set."""
        self.engine.eval()
        total_loss = 0.0
        n_batches = 0
        device = self.device

        with torch.no_grad():
            for prompt_emb, model_idx, target_score in loader:
                prompt_emb = prompt_emb.to(device)
                model_idx = model_idx.to(device)
                target_score = target_score.to(device)
                predicted = self.engine.predict_score(prompt_emb, model_idx)
                loss = F.mse_loss(predicted, target_score)
                total_loss += loss.item()
                n_batches += 1

        self.engine.train()
        return total_loss / max(n_batches, 1)

    def _save_checkpoint(self, tag: str):
        """Save model checkpoint."""
        checkpoint_dir = self.db_path.parent / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        path = checkpoint_dir / f"bilinear_phi_{tag}.pt"
        torch.save({
            "model_state": self.engine.state_dict(),
            "num_models": self.engine.num_models,
            "d_prompt": self.engine.d_prompt,
            "d_model": self.engine.d_model,
            "n_phi": self.engine.n_phi,
            "model_to_idx": self.model_to_idx,
        }, path)
        logger.debug(f"Checkpoint saved: {path}")

    def _load_checkpoint(self, tag: str):
        """Load model checkpoint."""
        path = self.db_path.parent / "checkpoints" / f"bilinear_phi_{tag}.pt"
        if path.exists():
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            self.engine.load_state_dict(checkpoint["model_state"])
            logger.info(f"Loaded checkpoint: {path}")


def load_trained_engine(checkpoint_path: Path, device: Optional[str] = None) -> tuple[BilinearPhiEngine, dict]:
    """Load a trained BilinearPhiEngine from checkpoint.

    Returns:
        (engine, model_to_idx) tuple
    """
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    engine = BilinearPhiEngine(
        num_models=checkpoint["num_models"],
        d_prompt=checkpoint["d_prompt"],
        d_model=checkpoint["d_model"],
        n_phi=checkpoint["n_phi"],
    )
    engine.load_state_dict(checkpoint["model_state"])
    engine = engine.to(device)
    engine.eval()

    return engine, checkpoint["model_to_idx"]
