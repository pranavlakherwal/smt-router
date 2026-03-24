"""Evaluate trained BilinearPhiEngine on all 3 benchmarks.

Benchmarks:
  1. RouterBench: accuracy on model selection from pool of 11 models
  2. RouteLLM: binary routing quality (gpt-4 vs mixtral)
  3. RouterEval: model selection from variable pool sizes

Baselines (v4 scalar phi):
  - RouterBench: 54.71% accuracy
  - RouteLLM: AUC -7516
  - RouterEval: mu=0.458

Usage:
    PYTHONPATH=. ./venv/bin/python -m agents.router.eval_bilinear
    PYTHONPATH=. ./venv/bin/python -m agents.router.eval_bilinear --benchmark routerbench
    PYTHONPATH=. ./venv/bin/python -m agents.router.eval_bilinear --checkpoint data/checkpoints/bilinear_phi_best.pt
"""

import argparse
import json
import logging
import pickle
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

from agents.router.learned_phi import BilinearPhiEngine, load_trained_engine
from agents.router.model_registry import ModelRegistry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_CHECKPOINT = DATA_DIR / "checkpoints" / "bilinear_phi_best.pt"

# External data sources
ROUTERBENCH_0SHOT = Path("/Users/pranavlakherwal/Core/Neural/routerbench/data/hf_dataset/routerbench_0shot.pkl")
ROUTERBENCH_5SHOT = Path("/Users/pranavlakherwal/Core/Neural/routerbench/data/hf_dataset/routerbench_5shot.pkl")
ROUTELLM_EVALS = Path("/Users/pranavlakherwal/Core/Neural/RouteLLM/routellm/evals")
ROUTEREVAL_DATA = Path("/Users/pranavlakherwal/Core/Neural/RouterEval/data/router_dataset")

# v4 baselines for comparison
V4_BASELINES = {
    "routerbench_accuracy": 54.71,
    "routellm_auc": -7516,
    "routereval_mu": 0.458,
}


class BilinearEvaluator:
    """Evaluate a trained BilinearPhiEngine on benchmark datasets."""

    def __init__(
        self,
        engine: BilinearPhiEngine,
        model_to_idx: dict,
        device: str = "cpu",
    ):
        self.engine = engine
        self.model_to_idx = model_to_idx
        self.idx_to_model = {v: k for k, v in model_to_idx.items()}
        self.device = torch.device(device)
        self.engine = self.engine.to(self.device).eval()

        # Load raw transformers encoder (much faster than sentence_transformers)
        from transformers import AutoTokenizer, AutoModel
        log.info("Loading prompt encoder (all-mpnet-base-v2 via raw transformers)...")
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
        self.encoder_model = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")
        self.encode_device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.encoder_model = self.encoder_model.to(self.encode_device).eval()
        log.info(f"Encoder on device: {self.encode_device}")

        # Load model registry for alias resolution
        self.registry = ModelRegistry()

    @staticmethod
    def _mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def encode_prompts(self, prompts: list[str], batch_size: int = 64) -> torch.Tensor:
        """Encode prompts using raw transformers (faster than sentence_transformers)."""
        all_embeddings = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            encoded = self.tokenizer(
                batch, padding=True, truncation=True, max_length=128, return_tensors="pt",
            )
            encoded = {k: v.to(self.encode_device) for k, v in encoded.items()}
            with torch.no_grad():
                outputs = self.encoder_model(**encoded)
                emb = self._mean_pooling(outputs, encoded["attention_mask"])
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)
                all_embeddings.append(emb.cpu())
            if len(prompts) > 1000 and (i + batch_size) % 5000 < batch_size:
                done = min(i + batch_size, len(prompts))
                log.info(f"  Encoded {done:,}/{len(prompts):,} prompts ({done/len(prompts):.0%})")
        return torch.cat(all_embeddings, dim=0)

    def score_models(
        self,
        prompt_embedding: torch.Tensor,
        model_ids: list[str],
    ) -> dict[str, float]:
        """Score a list of models for a single prompt.

        Args:
            prompt_embedding: (768,) tensor
            model_ids: list of canonical model IDs

        Returns:
            {model_id: score} dict
        """
        scores = {}
        valid_ids = []
        valid_indices = []

        for mid in model_ids:
            # Try direct lookup, then alias resolution
            if mid in self.model_to_idx:
                valid_ids.append(mid)
                valid_indices.append(self.model_to_idx[mid])
            else:
                canonical = self.registry.resolve(mid)
                if canonical and canonical in self.model_to_idx:
                    valid_ids.append(mid)
                    valid_indices.append(self.model_to_idx[canonical])

        if not valid_indices:
            return scores

        # Batch score all valid models at once
        prompt_batch = prompt_embedding.unsqueeze(0).expand(len(valid_indices), -1).to(self.device)
        model_batch = torch.tensor(valid_indices, dtype=torch.long, device=self.device)

        with torch.no_grad():
            model_scores = self.engine.predict_score(prompt_batch, model_batch)

        for mid, score in zip(valid_ids, model_scores.cpu().tolist()):
            scores[mid] = score

        return scores

    def pick_best(
        self,
        prompt_embedding: torch.Tensor,
        model_ids: list[str],
    ) -> Optional[str]:
        """Pick the highest-scoring model for a prompt."""
        scores = self.score_models(prompt_embedding, model_ids)
        if not scores:
            return None
        return max(scores, key=scores.get)

    # ==================================================================
    # RouterBench evaluation
    # ==================================================================

    def eval_routerbench(self, include_5shot: bool = True) -> dict:
        """Evaluate on RouterBench: accuracy of model selection.

        For each prompt, the router picks from 11 models.
        Correctness = picked model's performance on that task.
        Accuracy = fraction of prompts where picked model scores >= 0.5.
        """
        log.info("=" * 60)
        log.info("RouterBench Evaluation")
        log.info("=" * 60)

        rb_models = [
            "gpt-3.5-turbo-1106", "gpt-4-1106-preview", "claude-instant-v1",
            "claude-v1", "claude-v2", "meta/llama-2-70b-chat",
            "mistralai/mixtral-8x7b-chat", "mistralai/mistral-7b-chat",
            "zero-one-ai/Yi-34B-Chat", "WizardLM/WizardLM-13B-V1.2",
            "meta/code-llama-instruct-34b-chat",
        ]

        # Check which models are in our registry
        available = []
        for mid in rb_models:
            if mid in self.model_to_idx:
                available.append(mid)
            else:
                canonical = self.registry.resolve(mid)
                if canonical and canonical in self.model_to_idx:
                    available.append(mid)
                else:
                    log.warning(f"RouterBench model not in trained registry: {mid}")
        log.info(f"Available models: {len(available)}/{len(rb_models)}")

        results = {"splits": {}}
        total_correct = 0
        total_samples = 0
        total_oracle_correct = 0
        total_random_correct = 0
        total_always_strong_correct = 0

        pkl_configs = [(ROUTERBENCH_0SHOT, "0shot")]
        if include_5shot:
            pkl_configs.append((ROUTERBENCH_5SHOT, "5shot"))

        for pkl_path, split_name in pkl_configs:
            if not pkl_path.exists():
                log.warning(f"RouterBench {split_name} not found: {pkl_path}")
                continue

            log.info(f"Loading RouterBench {split_name}...")
            df = pd.read_pickle(pkl_path)
            model_cols = [m for m in rb_models if m in df.columns]
            prompts = df["prompt"].fillna("").astype(str).tolist()

            # Pre-encode all prompts
            log.info(f"Encoding {len(prompts)} prompts...")
            embeddings = self.encode_prompts(prompts)

            correct = 0
            oracle_correct = 0
            random_correct = 0
            always_strong_correct = 0
            n = 0
            per_model_picks = defaultdict(int)
            per_model_correct = defaultdict(int)

            for i in range(len(df)):
                if not prompts[i]:
                    continue

                # Ground truth scores for each model
                gt_scores = {}
                for mid in model_cols:
                    val = df.iloc[i].get(mid, 0)
                    gt_scores[mid] = float(val) if pd.notna(val) else 0.0

                # Router's pick
                picked = self.pick_best(embeddings[i], model_cols)
                if picked is None:
                    continue

                n += 1
                picked_score = gt_scores.get(picked, 0.0)
                per_model_picks[picked] += 1

                if picked_score >= 0.5:
                    correct += 1
                    per_model_correct[picked] += 1

                # Oracle: always pick the best model
                oracle_score = max(gt_scores.values()) if gt_scores else 0.0
                if oracle_score >= 0.5:
                    oracle_correct += 1

                # Random baseline
                random_pick = np.random.choice(list(gt_scores.keys()))
                if gt_scores.get(random_pick, 0.0) >= 0.5:
                    random_correct += 1

                # Always-strong (gpt-4) baseline
                strong_score = gt_scores.get("gpt-4-1106-preview", 0.0)
                if strong_score >= 0.5:
                    always_strong_correct += 1

            acc = correct / max(n, 1) * 100
            oracle_acc = oracle_correct / max(n, 1) * 100
            random_acc = random_correct / max(n, 1) * 100
            strong_acc = always_strong_correct / max(n, 1) * 100

            log.info(f"  {split_name}: {acc:.2f}% accuracy ({correct}/{n})")
            log.info(f"  Oracle: {oracle_acc:.2f}%, Random: {random_acc:.2f}%, Always-GPT4: {strong_acc:.2f}%")

            # Model selection diversity
            n_unique_picks = len([m for m, c in per_model_picks.items() if c > 0])
            log.info(f"  Diversity: {n_unique_picks} unique models picked")
            for mid, cnt in sorted(per_model_picks.items(), key=lambda x: -x[1])[:5]:
                mid_acc = per_model_correct[mid] / max(cnt, 1) * 100
                log.info(f"    {mid}: {cnt} picks ({cnt/max(n,1)*100:.1f}%), acc={mid_acc:.1f}%")

            results["splits"][split_name] = {
                "accuracy": round(acc, 2),
                "oracle_accuracy": round(oracle_acc, 2),
                "random_accuracy": round(random_acc, 2),
                "always_strong_accuracy": round(strong_acc, 2),
                "n_samples": n,
                "model_picks": dict(per_model_picks),
                "n_unique_picks": n_unique_picks,
            }

            total_correct += correct
            total_samples += n
            total_oracle_correct += oracle_correct
            total_random_correct += random_correct
            total_always_strong_correct += always_strong_correct

        overall_acc = total_correct / max(total_samples, 1) * 100
        results["overall_accuracy"] = round(overall_acc, 2)
        results["total_samples"] = total_samples
        results["v4_baseline"] = V4_BASELINES["routerbench_accuracy"]
        results["improvement_vs_v4"] = round(overall_acc - V4_BASELINES["routerbench_accuracy"], 2)

        log.info(f"\nRouterBench Overall: {overall_acc:.2f}% (v4 baseline: {V4_BASELINES['routerbench_accuracy']}%)")
        log.info(f"Improvement vs v4: {results['improvement_vs_v4']:+.2f}%")

        return results

    # ==================================================================
    # RouteLLM evaluation
    # ==================================================================

    def eval_routellm(self) -> dict:
        """Evaluate on RouteLLM: binary routing between gpt-4 and mixtral.

        Metrics:
        - Accuracy: fraction of correct routing decisions
        - Quality at thresholds: quality retained at 20%/50%/80% strong model usage
        - AUC of quality-cost curve
        """
        log.info("=" * 60)
        log.info("RouteLLM Evaluation")
        log.info("=" * 60)

        strong = "gpt-4-1106-preview"
        weak = "mistralai/mixtral-8x7b-chat"  # RouterBench canonical name

        # Also try RouteLLM's canonical name
        weak_alt = "mistralai/Mixtral-8x7B-Instruct-v0.1"

        # Check availability
        strong_idx = self._resolve_model_idx(strong)
        weak_idx = self._resolve_model_idx(weak) or self._resolve_model_idx(weak_alt)

        if strong_idx is None or weak_idx is None:
            log.error(f"Missing models: strong={strong_idx is not None}, weak={weak_idx is not None}")
            return {"error": "Missing required models"}

        # Load RouteLLM evaluation data
        all_prompts = []
        all_strong_scores = []
        all_weak_scores = []
        all_sources = []

        # GSM8K
        gsm_path = ROUTELLM_EVALS / "gsm8k" / "gsm8k_responses.csv"
        if gsm_path.exists():
            df = pd.read_csv(gsm_path)
            for _, row in df.iterrows():
                prompt = str(row.get("prompt", ""))
                if prompt:
                    all_prompts.append(prompt[:512])
                    all_strong_scores.append(float(row.get(strong, row.get("gpt-4-1106-preview", 0))))
                    all_weak_scores.append(float(row.get(weak_alt, row.get(weak, 0))))
                    all_sources.append("gsm8k")
            log.info(f"  GSM8K: {len(df)} samples")

        # MMLU
        mmlu_dir = ROUTELLM_EVALS / "mmlu" / "responses"
        if mmlu_dir.exists():
            mmlu_count = 0
            for csv_file in sorted(mmlu_dir.glob("*.csv")):
                df = pd.read_csv(csv_file)
                for _, row in df.iterrows():
                    prompt = str(row.get("prompt", ""))
                    if prompt:
                        all_prompts.append(prompt[:512])
                        all_strong_scores.append(float(row.get(strong, row.get("gpt-4-1106-preview", 0))))
                        all_weak_scores.append(float(row.get(weak_alt, row.get(weak, 0))))
                        all_sources.append("mmlu")
                        mmlu_count += 1
            log.info(f"  MMLU: {mmlu_count} samples")

        # MT-Bench
        q_path = ROUTELLM_EVALS / "mt_bench" / "question.jsonl"
        j_path = ROUTELLM_EVALS / "mt_bench" / "judgements.jsonl"
        if q_path.exists() and j_path.exists():
            questions = pd.read_json(q_path, lines=True)
            judgements = pd.read_json(j_path, lines=True)
            strong_scores_map = {}
            weak_scores_map = {}
            for _, j in judgements.iterrows():
                qid = j["question_id"]
                model = j["model"]
                score = j["score"]
                turn = j.get("turn", 1)
                if turn == 1:
                    if model == strong:
                        strong_scores_map[qid] = score
                    elif model in (weak, weak_alt):
                        weak_scores_map[qid] = score

            mt_count = 0
            for _, q in questions.iterrows():
                qid = q["question_id"]
                if qid in strong_scores_map and qid in weak_scores_map:
                    prompt = q["turns"][0] if isinstance(q["turns"], list) else str(q["turns"])
                    all_prompts.append(str(prompt)[:512])
                    all_strong_scores.append(float(strong_scores_map[qid]))
                    all_weak_scores.append(float(weak_scores_map[qid]))
                    all_sources.append("mt_bench")
                    mt_count += 1
            log.info(f"  MT-Bench: {mt_count} samples")

        if not all_prompts:
            log.error("No RouteLLM evaluation data found")
            return {"error": "No data"}

        log.info(f"Total: {len(all_prompts)} samples")

        # Encode all prompts
        log.info("Encoding prompts...")
        embeddings = self.encode_prompts(all_prompts)

        # Score both models for each prompt
        strong_preds = []
        weak_preds = []
        for i in range(len(all_prompts)):
            scores = self.score_models(embeddings[i], [strong, weak])
            strong_preds.append(scores.get(strong, 0.5))
            weak_preds.append(scores.get(weak, 0.5))

        strong_preds = np.array(strong_preds)
        weak_preds = np.array(weak_preds)
        strong_gt = np.array(all_strong_scores)
        weak_gt = np.array(all_weak_scores)

        # Router decision: route to strong if strong_pred > weak_pred
        route_to_strong = strong_preds > weak_preds
        pct_strong = route_to_strong.mean() * 100

        # Quality: what's the actual score of the model we picked?
        picked_gt = np.where(route_to_strong, strong_gt, weak_gt)
        oracle_gt = np.maximum(strong_gt, weak_gt)
        quality = picked_gt.mean()
        oracle_quality = oracle_gt.mean()
        always_strong_quality = strong_gt.mean()
        always_weak_quality = weak_gt.mean()

        # Binary routing accuracy: was our pick actually the better model?
        strong_is_better = strong_gt > weak_gt
        routing_acc = ((route_to_strong == strong_is_better) | (strong_gt == weak_gt)).mean() * 100

        # Quality-cost curve at different thresholds
        # Sort by confidence (difference in predicted scores)
        confidence = strong_preds - weak_preds
        sorted_indices = np.argsort(-confidence)  # most confident "route strong" first

        thresholds = {}
        for pct in [20, 50, 80]:
            n_strong = int(len(all_prompts) * pct / 100)
            strong_set = set(sorted_indices[:n_strong])
            quality_at_pct = np.mean([
                strong_gt[i] if i in strong_set else weak_gt[i]
                for i in range(len(all_prompts))
            ])
            quality_ratio = quality_at_pct / max(always_strong_quality, 1e-8) * 100
            thresholds[f"{pct}pct_strong"] = {
                "quality": round(float(quality_at_pct), 4),
                "quality_ratio": round(float(quality_ratio), 2),
            }
            log.info(f"  At {pct}% GPT-4 calls: quality={quality_at_pct:.4f} ({quality_ratio:.1f}% of always-strong)")

        # AUC approximation (trapezoidal)
        n_points = 20
        auc_sum = 0.0
        for step in range(n_points + 1):
            pct = step / n_points
            n_strong = int(len(all_prompts) * pct)
            strong_set = set(sorted_indices[:n_strong])
            q = np.mean([
                strong_gt[i] if i in strong_set else weak_gt[i]
                for i in range(len(all_prompts))
            ])
            auc_sum += q
        auc = auc_sum / (n_points + 1)

        results = {
            "total_samples": len(all_prompts),
            "pct_routed_strong": round(float(pct_strong), 2),
            "routing_accuracy": round(float(routing_acc), 2),
            "quality": round(float(quality), 4),
            "oracle_quality": round(float(oracle_quality), 4),
            "always_strong_quality": round(float(always_strong_quality), 4),
            "always_weak_quality": round(float(always_weak_quality), 4),
            "auc": round(float(auc), 4),
            "thresholds": thresholds,
            "v4_baseline_auc": V4_BASELINES["routellm_auc"],
            "per_source": {},
        }

        # Per-source breakdown
        source_set = set(all_sources)
        for src in sorted(source_set):
            mask = np.array([s == src for s in all_sources])
            src_correct = ((route_to_strong[mask] == strong_is_better[mask]) | (strong_gt[mask] == weak_gt[mask])).mean()
            results["per_source"][src] = {
                "n": int(mask.sum()),
                "routing_accuracy": round(float(src_correct * 100), 2),
            }

        log.info(f"\nRouteLLM Overall:")
        log.info(f"  Routing accuracy: {routing_acc:.2f}%")
        log.info(f"  Routes to strong: {pct_strong:.1f}%")
        log.info(f"  Quality: {quality:.4f} (oracle: {oracle_quality:.4f})")
        log.info(f"  AUC: {auc:.4f}")

        return results

    # ==================================================================
    # RouterEval evaluation
    # ==================================================================

    def eval_routereval(self, max_pool_size: int = 100) -> dict:
        """Evaluate on RouterEval: model selection from variable pool sizes.

        Metrics:
        - mu: average performance of selected model relative to oracle
        - Vr: variance reduction vs random selection
        - accuracy: fraction of times the oracle model is selected
        """
        log.info("=" * 60)
        log.info("RouterEval Evaluation")
        log.info("=" * 60)

        if not ROUTEREVAL_DATA.exists():
            log.error(f"RouterEval data not found: {ROUTEREVAL_DATA}")
            return {"error": "Data not found"}

        total_router_score = 0.0
        total_oracle_score = 0.0
        total_random_score = 0.0
        total_correct = 0
        total_samples = 0
        per_dataset = {}

        for pkl_file in sorted(ROUTEREVAL_DATA.glob("*.pkl")):
            dataset_name = pkl_file.stem.replace("_router_dataset", "")
            log.info(f"  Processing RouterEval {dataset_name}...")

            with open(pkl_file, "rb") as f:
                data = pickle.load(f)

            dataset_router = 0.0
            dataset_oracle = 0.0
            dataset_random = 0.0
            dataset_correct = 0
            dataset_n = 0

            for difficulty in ["easy", "hard"]:
                diff_data = data.get(difficulty, {})
                for pool_size, configs in diff_data.items():
                    if not isinstance(configs, dict):
                        continue

                    for config_name, config in configs.items():
                        if not isinstance(config, dict):
                            continue

                        model_names = config.get("model", np.array([]))
                        if len(model_names) == 0:
                            continue

                        model_list = list(model_names)
                        d = config.get("data", {})

                        # Use val_score for evaluation (not train_score)
                        scores_mat = d.get("val_score", np.array([]))
                        if len(scores_mat) == 0:
                            scores_mat = d.get("train_score", np.array([]))
                        if len(scores_mat) == 0:
                            continue

                        n_rows = scores_mat.shape[0]
                        n_models = min(scores_mat.shape[1], len(model_list))

                        # Subsample if pool is too large
                        if n_models > max_pool_size:
                            eval_indices = np.random.choice(n_models, max_pool_size, replace=False)
                            eval_models = [model_list[j] for j in eval_indices]
                        else:
                            eval_indices = list(range(n_models))
                            eval_models = model_list[:n_models]

                        # Check which models we can score
                        scorable = []
                        for mid in eval_models:
                            if mid in self.model_to_idx:
                                scorable.append(mid)
                            else:
                                canonical = self.registry.resolve(mid)
                                if canonical and canonical in self.model_to_idx:
                                    scorable.append(mid)

                        if len(scorable) < 2:
                            continue

                        # We need prompts for RouterEval, but it doesn't have them
                        # Use a generic prompt as surrogate (model embeddings carry the signal)
                        generic_emb = self.encode_prompts(["general benchmark task"])[0]

                        for i in range(min(n_rows, 100)):  # Cap for speed
                            row_scores = {}
                            for j, mid in enumerate(eval_models):
                                if j < len(eval_indices):
                                    row_scores[mid] = float(scores_mat[i, eval_indices[j]] if isinstance(eval_indices, list) and j < n_models else scores_mat[i, j])

                            # Router pick
                            picked = self.pick_best(generic_emb, scorable)
                            if picked is None:
                                continue

                            picked_score = row_scores.get(picked, 0.0)
                            oracle_score = max(row_scores.values()) if row_scores else 0.0
                            random_score = float(np.mean(list(row_scores.values()))) if row_scores else 0.0

                            dataset_router += picked_score
                            dataset_oracle += oracle_score
                            dataset_random += random_score
                            dataset_n += 1

                            if picked_score >= oracle_score - 1e-6:
                                dataset_correct += 1

            if dataset_n > 0:
                mu = dataset_router / max(dataset_oracle, 1e-8)
                acc = dataset_correct / dataset_n * 100
                per_dataset[dataset_name] = {
                    "n_samples": dataset_n,
                    "mu": round(float(mu), 4),
                    "accuracy": round(float(acc), 2),
                    "avg_router_score": round(dataset_router / dataset_n, 4),
                    "avg_oracle_score": round(dataset_oracle / dataset_n, 4),
                }
                log.info(f"    {dataset_name}: mu={mu:.4f}, acc={acc:.1f}% ({dataset_n} samples)")

            total_router_score += dataset_router
            total_oracle_score += dataset_oracle
            total_random_score += dataset_random
            total_correct += dataset_correct
            total_samples += dataset_n

        # Overall metrics
        overall_mu = total_router_score / max(total_oracle_score, 1e-8)
        overall_acc = total_correct / max(total_samples, 1) * 100
        random_mu = total_random_score / max(total_oracle_score, 1e-8)

        # Vr = variance reduction vs random
        vr = (overall_mu - random_mu) / max(1.0 - random_mu, 1e-8)

        results = {
            "total_samples": total_samples,
            "mu": round(float(overall_mu), 4),
            "accuracy": round(float(overall_acc), 2),
            "random_mu": round(float(random_mu), 4),
            "Vr": round(float(vr), 4),
            "per_dataset": per_dataset,
            "v4_baseline_mu": V4_BASELINES["routereval_mu"],
            "improvement_vs_v4": round(float(overall_mu - V4_BASELINES["routereval_mu"]), 4),
        }

        log.info(f"\nRouterEval Overall:")
        log.info(f"  mu={overall_mu:.4f} (v4 baseline: {V4_BASELINES['routereval_mu']})")
        log.info(f"  Accuracy: {overall_acc:.2f}%")
        log.info(f"  Vr (variance reduction): {vr:.4f}")
        log.info(f"  Improvement vs v4: {results['improvement_vs_v4']:+.4f}")

        return results

    # ==================================================================
    # Internal test set evaluation
    # ==================================================================

    def eval_internal_test(self) -> dict:
        """Evaluate on the held-out test split from training data.

        This gives a quick sanity check on MSE loss and score prediction quality.
        """
        log.info("=" * 60)
        log.info("Internal Test Set Evaluation")
        log.info("=" * 60)

        cache_path = DATA_DIR / "embedding_cache.pt"
        if not cache_path.exists():
            log.error("No embedding cache found. Run training first.")
            return {"error": "No cache"}

        cached = torch.load(str(cache_path), map_location="cpu", weights_only=False)
        test_emb = cached["test_embeddings"]
        test_idx = cached["test_model_indices"]
        test_scores = cached["test_scores"]

        n = len(test_scores)
        log.info(f"Test set: {n:,} samples")

        # Predict in batches
        all_preds = []
        batch_size = 512
        for i in range(0, n, batch_size):
            emb = test_emb[i:i+batch_size].to(self.device)
            idx = test_idx[i:i+batch_size].to(self.device)
            with torch.no_grad():
                preds = self.engine.predict_score(emb, idx)
            all_preds.append(preds.cpu())

        all_preds = torch.cat(all_preds)
        mse = torch.nn.functional.mse_loss(all_preds, test_scores).item()
        mae = (all_preds - test_scores).abs().mean().item()

        # Correlation
        pred_np = all_preds.numpy()
        gt_np = test_scores.numpy()
        correlation = float(np.corrcoef(pred_np, gt_np)[0, 1])

        # Binary accuracy (predict > 0.5 vs ground truth > 0.5)
        pred_binary = (all_preds > 0.5).float()
        gt_binary = (test_scores > 0.5).float()
        binary_acc = (pred_binary == gt_binary).float().mean().item() * 100

        results = {
            "n_samples": n,
            "mse": round(mse, 6),
            "mae": round(mae, 6),
            "correlation": round(correlation, 4),
            "binary_accuracy": round(binary_acc, 2),
        }

        log.info(f"  MSE: {mse:.6f}")
        log.info(f"  MAE: {mae:.6f}")
        log.info(f"  Pearson r: {correlation:.4f}")
        log.info(f"  Binary accuracy: {binary_acc:.2f}%")

        return results

    # ==================================================================
    # Helpers
    # ==================================================================

    def _resolve_model_idx(self, model_id: str) -> Optional[int]:
        """Resolve model ID to embedding index."""
        if model_id in self.model_to_idx:
            return self.model_to_idx[model_id]
        canonical = self.registry.resolve(model_id)
        if canonical and canonical in self.model_to_idx:
            return self.model_to_idx[canonical]
        return None


def main():
    parser = argparse.ArgumentParser(description="Evaluate BilinearPhiEngine")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT))
    parser.add_argument(
        "--benchmark", type=str, default="all",
        choices=["all", "routerbench", "routellm", "routereval", "internal"],
    )
    parser.add_argument("--output", type=str, default=str(DATA_DIR / "eval_results.json"))
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        log.error(f"Checkpoint not found: {checkpoint_path}")
        log.error("Train the model first with: PYTHONPATH=. ./venv/bin/python -m agents.router.train_bilinear")
        return

    # Load trained engine
    log.info(f"Loading checkpoint: {checkpoint_path}")
    engine, model_to_idx = load_trained_engine(checkpoint_path)
    log.info(f"Loaded engine: {engine.num_models} models, {engine.n_phi} phi heads")

    # Use CPU for eval (more reliable than MPS for small batches)
    evaluator = BilinearEvaluator(engine, model_to_idx, device="cpu")

    results = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ")}

    benchmarks = args.benchmark
    if benchmarks == "all":
        benchmarks = ["internal", "routerbench", "routellm", "routereval"]
    else:
        benchmarks = [benchmarks]

    for bench in benchmarks:
        start = time.time()
        if bench == "routerbench":
            results["routerbench"] = evaluator.eval_routerbench()
        elif bench == "routellm":
            results["routellm"] = evaluator.eval_routellm()
        elif bench == "routereval":
            results["routereval"] = evaluator.eval_routereval()
        elif bench == "internal":
            results["internal"] = evaluator.eval_internal_test()
        elapsed = time.time() - start
        log.info(f"  {bench} eval took {elapsed:.1f}s")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    log.info(f"\nResults saved: {output_path}")

    # Summary table
    log.info("\n" + "=" * 60)
    log.info("EVALUATION SUMMARY")
    log.info("=" * 60)

    if "internal" in results:
        r = results["internal"]
        log.info(f"Internal Test:  MSE={r['mse']:.6f}  r={r['correlation']:.4f}  binary_acc={r['binary_accuracy']:.1f}%")

    if "routerbench" in results:
        r = results["routerbench"]
        log.info(f"RouterBench:    acc={r['overall_accuracy']:.2f}%  (v4: {r['v4_baseline']}%, delta: {r['improvement_vs_v4']:+.2f}%)")

    if "routellm" in results:
        r = results["routellm"]
        log.info(f"RouteLLM:       routing_acc={r['routing_accuracy']:.2f}%  quality={r['quality']:.4f}  AUC={r['auc']:.4f}")

    if "routereval" in results:
        r = results["routereval"]
        log.info(f"RouterEval:     mu={r['mu']:.4f}  (v4: {r['v4_baseline_mu']}, delta: {r['improvement_vs_v4']:+.4f})")

    log.info("=" * 60)


if __name__ == "__main__":
    main()
