"""Phi function evaluators for the S(M,T) router.

16 phi functions, each producing a 0-1 score for (endpoint, task) pairs.
- 8 data-grounded: query normalized_scores.db for empirical scores
- 8 runtime/stub: return prior-based estimates or runtime metrics

Imports PHI_DATA_RULES from data_ingester.phi_mapper to avoid duplication.
"""

import logging
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional

from agents.router.models import Task, Endpoint, PhiScore, EndpointType
from agents.router.config import NORMALIZED_DB_PATH

logger = logging.getLogger(__name__)

# Import phi data mapping rules from the data ingester
try:
    from agents.data_ingester.phi_mapper import PHI_DATA_RULES
except ImportError:
    logger.warning("Could not import PHI_DATA_RULES, using empty rules")
    PHI_DATA_RULES = {}

# Phi function names for reference
PHI_NAMES = {
    "phi_1": "constraint_filter",
    "phi_2": "domain_classifier",
    "phi_3": "complexity_estimator",
    "phi_4": "performance_predictor",
    "phi_5": "context_fitness",
    "phi_6": "historical_success",
    "phi_7": "uncertainty_score",
    "phi_8": "cascade_position",
    "phi_9": "load_availability",
    "phi_10": "cross_node_capability",
    "phi_11": "calibration_quality",
    "phi_12": "output_structure_compliance",
    "phi_13": "reasoning_depth",
    "phi_14": "cost_efficiency_ratio",
    "phi_15": "context_window_utilization",
    "phi_16": "instruction_adherence",
}

# Default prior scores for phi functions with no data
DEFAULT_PRIORS = {
    "phi_1": 0.5,
    "phi_2": 0.5,
    "phi_3": 0.5,
    "phi_4": 0.5,
    "phi_5": 0.5,
    "phi_6": 0.5,
    "phi_7": 0.5,  # uncertainty: 0.5 = maximum entropy (unknown)
    "phi_8": 0.5,
    "phi_9": 1.0,  # availability: assume available
    "phi_10": 0.3, # cross-node: conservative
    "phi_11": 0.5, # calibration: unknown
    "phi_12": 0.5,
    "phi_13": 0.5,
    "phi_14": 0.5,
    "phi_15": 0.5,
    "phi_16": 0.5,
}

# Model ID aliases: maps external/benchmark model IDs to DB model IDs.
# Each key can map to multiple candidate IDs (tried in order).
MODEL_ID_ALIASES: Dict[str, List[str]] = {
    # RouterBench models
    "mistralai/mixtral-8x7b-chat": ["mixtral-8x7b", "Mixtral 8x7B", "Mixtral-8x7B-Instruct-v0.1", "mistralai/mixtral-8x7b-instruct"],
    "mixtral-8x7b-chat": ["mixtral-8x7b", "Mixtral 8x7B", "Mixtral-8x7B-Instruct-v0.1"],
    "meta/llama-2-70b-chat": ["llama-2-70b-chat-hf", "Llama 2-70B", "meta-llama/llama-3.3-70b-instruct"],
    "llama-2-70b-chat": ["llama-2-70b-chat-hf", "Llama 2-70B"],
    "mistralai/mistral-7b-chat": ["Mistral 7B", "mistralai/mistral-7b-instruct-v0.1", "Mistral-7B-Instruct-v0.2"],
    "mistral-7b-chat": ["Mistral 7B", "mistralai/mistral-7b-instruct-v0.1"],
    "claude-instant-v1": ["claude", "claude-3-haiku-20240307", "anthropic/claude-3-haiku"],
    "claude-v1": ["claude", "anthropic/claude-3.5-sonnet"],
    "claude-v2": ["claude-2", "claude-2.1", "Claude 2"],
    "WizardLM/WizardLM-13B-V1.2": ["WizardLM 13B v1.2", "wizardlm-13b"],
    "WizardLM-13B-V1.2": ["WizardLM 13B v1.2", "wizardlm-13b"],
    "meta/code-llama-instruct-34b-chat": ["Code Llama-34B", "codellama-34b"],
    "code-llama-instruct-34b-chat": ["Code Llama-34B", "codellama-34b"],
    # RouteLLM models (strong=gpt-4, weak=mixtral)
    "gpt-4": ["gpt-4", "gpt-4-0613", "gpt4", "gpt4_0613", "GPT-4 (Jun 2023)"],
    "mistralai/Mixtral-8x7B-Instruct-v0.1": ["mixtral-8x7b", "Mixtral 8x7B", "Mixtral-8x7B-Instruct-v0.1", "mistralai/mixtral-8x7b-instruct"],
    "Mixtral-8x7B-Instruct-v0.1": ["mixtral-8x7b", "Mixtral 8x7B", "mistralai/mixtral-8x7b-instruct"],
    "mixtral-8x7b-instruct-v0.1": ["mixtral-8x7b", "Mixtral 8x7B", "mistralai/mixtral-8x7b-instruct"],
    # GPT-4 variants (alpaca_eval uses underscore format)
    "gpt-4-1106-preview": ["gpt-4-1106-preview", "gpt4_1106_preview", "GPT-4 (Jun 2023)"],
    "gpt-4-0613": ["gpt-4-0613", "gpt4_0613", "GPT-4 (Jun 2023)"],
    "gpt-4-0125-preview": ["gpt-4-0125-preview", "gpt4_0125_preview"],
    # GPT-3.5 variants (already exact match in alpaca_eval as gpt-3.5-turbo-1106)
    "gpt-3.5-turbo-1106": ["gpt-3.5-turbo-1106", "gpt35_turbo_instruct", "GPT-3.5 Turbo"],
    "gpt-3.5-turbo": ["gpt-3.5-turbo-1106", "gpt-3.5-turbo-0301", "gpt35_turbo_instruct", "GPT-3.5 Turbo"],
}


class PhiEvaluatorEngine:
    """Evaluates all 16 phi functions for an (endpoint, task) pair."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or NORMALIZED_DB_PATH
        self._conn: Optional[sqlite3.Connection] = None
        self._model_score_cache: Dict[str, Dict[str, float]] = {}

    def _get_conn(self) -> Optional[sqlite3.Connection]:
        """Lazy connection to normalized_scores DB."""
        if self._conn is None and self.db_path.exists():
            try:
                self._conn = sqlite3.connect(str(self.db_path))
                logger.info(f"Connected to {self.db_path}")
            except Exception as e:
                logger.warning(f"Could not connect to DB: {e}")
        return self._conn

    def evaluate_all(self, task: Task, endpoint: Endpoint) -> List[PhiScore]:
        """Evaluate all 16 phi functions and return scores."""
        scores = []
        for i in range(1, 17):
            phi_id = f"phi_{i}"
            method = getattr(self, f"_eval_{phi_id}", None)
            if method:
                score = method(task, endpoint)
            else:
                score = PhiScore(
                    phi_id=phi_id,
                    name=PHI_NAMES.get(phi_id, phi_id),
                    score=DEFAULT_PRIORS.get(phi_id, 0.5),
                    confidence=0.0,
                    source="default",
                )
            scores.append(score)
        return scores

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    # ------------------------------------------------------------------
    # Data-grounded phi functions (query normalized_scores.db)
    # ------------------------------------------------------------------

    def _eval_phi_1(self, task: Task, endpoint: Endpoint) -> PhiScore:
        """constraint_filter: Binary fit based on hard constraints (context window)."""
        # For LLMs, check context window ratio
        if endpoint.endpoint_type == EndpointType.LLM and task.context_length > 0:
            ratio = min(endpoint.context_window / max(task.context_length, 1), 1.0)
            # Penalize if context usage > 80% of window
            score = 1.0 if ratio > 1.25 else (ratio / 1.25)
        else:
            score = 1.0  # non-LLM endpoints or no context requirement

        return PhiScore(
            phi_id="phi_1", name="constraint_filter",
            score=score, confidence=0.9, source="computed",
        )

    def _eval_phi_2(self, task: Task, endpoint: Endpoint) -> PhiScore:
        """domain_classifier: Task-domain classification strength."""
        score = self._query_model_phi_score(endpoint, "phi_2", task)
        return PhiScore(
            phi_id="phi_2", name="domain_classifier",
            score=score["score"], confidence=score["confidence"],
            source=score["source"],
        )

    def _eval_phi_3(self, task: Task, endpoint: Endpoint) -> PhiScore:
        """complexity_estimator: Surface-level task complexity signal."""
        # Blend DB score with task complexity
        db_score = self._query_model_phi_score(endpoint, "phi_3", task)
        # Weight complexity: harder tasks benefit from better models
        complexity_weight = task.complexity
        blended = db_score["score"] * (0.5 + 0.5 * complexity_weight)

        return PhiScore(
            phi_id="phi_3", name="complexity_estimator",
            score=min(blended, 1.0), confidence=db_score["confidence"],
            source=db_score["source"],
        )

    def _eval_phi_4(self, task: Task, endpoint: Endpoint) -> PhiScore:
        """performance_predictor: Expected quality per model per task type."""
        score = self._query_model_phi_score(endpoint, "phi_4", task)
        return PhiScore(
            phi_id="phi_4", name="performance_predictor",
            score=score["score"], confidence=score["confidence"],
            source=score["source"],
        )

    def _eval_phi_5(self, task: Task, endpoint: Endpoint) -> PhiScore:
        """context_fitness: Content relevance (limited data, return prior)."""
        return PhiScore(
            phi_id="phi_5", name="context_fitness",
            score=DEFAULT_PRIORS["phi_5"], confidence=0.1,
            source="prior",
        )

    def _eval_phi_6(self, task: Task, endpoint: Endpoint) -> PhiScore:
        """historical_success: Past success rate on similar tasks."""
        score = self._query_model_phi_score(endpoint, "phi_6", task)
        return PhiScore(
            phi_id="phi_6", name="historical_success",
            score=score["score"], confidence=score["confidence"],
            source=score["source"],
        )

    def _eval_phi_7(self, task: Task, endpoint: Endpoint) -> PhiScore:
        """uncertainty_score: Epistemic uncertainty about model capability."""
        # Inverse of data coverage: more data = lower uncertainty = higher score
        phi4_score = self._query_model_phi_score(endpoint, "phi_4", task)
        # High confidence in phi_4 means low uncertainty (good)
        uncertainty = 1.0 - phi4_score["confidence"]
        # Invert: lower uncertainty = higher phi_7 score
        score = 1.0 - uncertainty

        return PhiScore(
            phi_id="phi_7", name="uncertainty_score",
            score=score, confidence=0.5,
            source="derived",
        )

    def _eval_phi_8(self, task: Task, endpoint: Endpoint) -> PhiScore:
        """cascade_position: Where model sits in cost-quality cascade."""
        score = self._query_model_phi_score(endpoint, "phi_8", task)
        return PhiScore(
            phi_id="phi_8", name="cascade_position",
            score=score["score"], confidence=score["confidence"],
            source=score["source"],
        )

    def _eval_phi_9(self, task: Task, endpoint: Endpoint) -> PhiScore:
        """load_availability: Current availability (runtime metric)."""
        score = 1.0 if endpoint.available else 0.0
        return PhiScore(
            phi_id="phi_9", name="load_availability",
            score=score, confidence=1.0, source="runtime",
        )

    def _eval_phi_10(self, task: Task, endpoint: Endpoint) -> PhiScore:
        """cross_node_capability: Multi-endpoint ensemble potential."""
        # Non-LLM endpoints are complementary by nature
        if endpoint.endpoint_type != EndpointType.LLM:
            score = 0.7
        else:
            score = DEFAULT_PRIORS["phi_10"]

        return PhiScore(
            phi_id="phi_10", name="cross_node_capability",
            score=score, confidence=0.2, source="prior",
        )

    def _eval_phi_11(self, task: Task, endpoint: Endpoint) -> PhiScore:
        """calibration_quality: Model's self-knowledge (ECE-like)."""
        return PhiScore(
            phi_id="phi_11", name="calibration_quality",
            score=DEFAULT_PRIORS["phi_11"], confidence=0.1,
            source="prior",
        )

    def _eval_phi_12(self, task: Task, endpoint: Endpoint) -> PhiScore:
        """output_structure_compliance: Structured output quality gradient."""
        if task.requires_structured_output and endpoint.supports_structured_output:
            score = 0.9
        elif task.requires_structured_output:
            score = 0.3
        else:
            score = 0.7  # no requirement, neutral-positive

        return PhiScore(
            phi_id="phi_12", name="output_structure_compliance",
            score=score, confidence=0.4, source="computed",
        )

    def _eval_phi_13(self, task: Task, endpoint: Endpoint) -> PhiScore:
        """reasoning_depth: Multi-step reasoning capability."""
        score = self._query_model_phi_score(endpoint, "phi_13", task)
        # Boost for endpoints with thinking support on reasoning tasks
        if task.task_type.value in ("reasoning", "math", "coding") and endpoint.supports_thinking:
            boosted = min(score["score"] * 1.15, 1.0)
        else:
            boosted = score["score"]

        return PhiScore(
            phi_id="phi_13", name="reasoning_depth",
            score=boosted, confidence=score["confidence"],
            source=score["source"],
        )

    def _eval_phi_14(self, task: Task, endpoint: Endpoint) -> PhiScore:
        """cost_efficiency_ratio: Quality per dollar."""
        if endpoint.endpoint_type != EndpointType.LLM:
            # Non-LLM endpoints have no direct per-token cost
            return PhiScore(
                phi_id="phi_14", name="cost_efficiency_ratio",
                score=0.8, confidence=0.3, source="computed",
            )

        # Lower cost = higher efficiency score
        total_cost_per_m = endpoint.cost_per_million_input + endpoint.cost_per_million_output
        if total_cost_per_m <= 0:
            score = 1.0  # free
        elif total_cost_per_m < 1.0:
            score = 0.9
        elif total_cost_per_m < 5.0:
            score = 0.7
        elif total_cost_per_m < 20.0:
            score = 0.4
        else:
            score = 0.2

        return PhiScore(
            phi_id="phi_14", name="cost_efficiency_ratio",
            score=score, confidence=0.8, source="computed",
        )

    def _eval_phi_15(self, task: Task, endpoint: Endpoint) -> PhiScore:
        """context_window_utilization: Performance degradation with length."""
        if task.context_length <= 0 or endpoint.context_window <= 0:
            return PhiScore(
                phi_id="phi_15", name="context_window_utilization",
                score=0.8, confidence=0.3, source="computed",
            )

        utilization = task.context_length / endpoint.context_window
        # Degradation curve: performance drops as utilization increases
        if utilization < 0.25:
            score = 1.0
        elif utilization < 0.5:
            score = 0.9
        elif utilization < 0.75:
            score = 0.7
        elif utilization < 0.9:
            score = 0.5
        else:
            score = 0.3

        return PhiScore(
            phi_id="phi_15", name="context_window_utilization",
            score=score, confidence=0.5, source="computed",
        )

    def _eval_phi_16(self, task: Task, endpoint: Endpoint) -> PhiScore:
        """instruction_adherence: How well the model follows instructions."""
        score = self._query_model_phi_score(endpoint, "phi_16", task)
        return PhiScore(
            phi_id="phi_16", name="instruction_adherence",
            score=score["score"], confidence=score["confidence"],
            source=score["source"],
        )

    # ------------------------------------------------------------------
    # DB query helper
    # ------------------------------------------------------------------

    def _query_model_phi_score(
        self, endpoint: Endpoint, phi_id: str, task: Task
    ) -> Dict:
        """Query normalized_scores DB for a model's average score on a phi function.

        Returns dict with score (0-1), confidence (0-1), and source string.
        Falls back to prior if no data or non-LLM endpoint.
        """
        default = {
            "score": DEFAULT_PRIORS.get(phi_id, 0.5),
            "confidence": 0.0,
            "source": "prior",
        }

        # Only query DB for LLM endpoints
        if endpoint.endpoint_type != EndpointType.LLM:
            return default

        # Check cache
        cache_key = f"{endpoint.model_id}:{phi_id}"
        if cache_key in self._model_score_cache:
            return self._model_score_cache[cache_key]

        conn = self._get_conn()
        if conn is None:
            return default

        rules = PHI_DATA_RULES.get(phi_id)
        if not rules:
            return default

        # Build query from PHI_DATA_RULES using strict AND logic:
        # When BOTH task_types AND metrics are specified, require BOTH to match.
        # This prevents e.g. 357K routellm_quality records flooding phi_4
        # just because task_type='general' matches (routellm_quality is NOT
        # a correctness/accuracy/win_rate metric).
        task_type_clause = None
        task_type_params = []
        metric_clause = None
        metric_params = []
        source_clause = None
        source_params = []

        if rules.get("task_types"):
            placeholders = ",".join(["?"] * len(rules["task_types"]))
            task_type_clause = f"task_type IN ({placeholders})"
            task_type_params = list(rules["task_types"])

        metric_conds = []
        for m in rules.get("metrics", []):
            metric_conds.append("metric_name LIKE ?")
            metric_params.append(f"%{m}%")
        if metric_conds:
            metric_clause = f"({' OR '.join(metric_conds)})"

        if rules.get("sources"):
            placeholders = ",".join(["?"] * len(rules["sources"]))
            source_clause = f"source IN ({placeholders})"
            source_params = list(rules["sources"])

        # Combine: all non-None clauses are AND'd together
        clauses = []
        params = []
        if task_type_clause:
            clauses.append(task_type_clause)
            params.extend(task_type_params)
        if metric_clause:
            clauses.append(metric_clause)
            params.extend(metric_params)
        if source_clause:
            clauses.append(source_clause)
            params.extend(source_params)

        if not clauses:
            return default

        where = " AND ".join(clauses)

        # Build candidate model IDs: exact, strip prefix, aliases
        model_ids = [endpoint.model_id]
        if "/" in endpoint.model_id:
            model_ids.append(endpoint.model_id.split("/")[-1])
        # Add alias candidates
        for mid_try in [endpoint.model_id] + model_ids[1:]:
            if mid_try in MODEL_ID_ALIASES:
                model_ids.extend(MODEL_ID_ALIASES[mid_try])
        # Deduplicate while preserving order
        seen = set()
        unique_ids = []
        for m in model_ids:
            if m not in seen:
                seen.add(m)
                unique_ids.append(m)
        model_ids = unique_ids

        # Query ALL aliases at once, averaging per-source first to prevent
        # sources with many duplicate rows from dominating.
        # E.g. swebench has 28 rows at ~0.12, alpaca_eval has 6 rows at ~0.977.
        # A flat AVG would give 0.28. Per-source AVG gives (0.12 + 0.977)/2 = 0.55.
        mid_placeholders = ",".join(["?"] * len(model_ids))
        full_where = f"({where}) AND model_id IN ({mid_placeholders}) AND score >= 0 AND score <= 1.0"
        full_params = params + model_ids

        try:
            cursor = conn.execute(
                f"""SELECT AVG(src_avg), SUM(src_cnt) FROM (
                    SELECT source, AVG(score) as src_avg, COUNT(*) as src_cnt
                    FROM normalized_scores WHERE {full_where}
                    GROUP BY source
                )""",
                full_params,
            )
            row = cursor.fetchone()
            if row and row[0] is not None and row[1] > 0:
                avg_score = row[0]
                count = int(row[1])
                # Confidence scales with data volume: asymptotic to 1.0
                confidence = min(count / (count + 10), 0.95)
                result = {
                    "score": round(avg_score, 4),
                    "confidence": round(confidence, 3),
                    "source": "data",
                }
                self._model_score_cache[cache_key] = result
                return result
        except Exception as e:
            logger.debug(f"DB query failed for {model_ids}/{phi_id}: {e}")

        self._model_score_cache[cache_key] = default
        return default
