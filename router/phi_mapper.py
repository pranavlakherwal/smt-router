"""Which benchmark records ground each phi function.

Each phi function in S(M,T) needs grounding data. For each one this table names the task
types, the metric names and the sources whose records count toward it in the benchmark
database. phi_evaluators.py builds its database query from these rules.

The benchmark database itself is not shipped with this repository. Without it, the
data-grounded phi functions fall back to priors, except phi_4, which reads the measured
table in data/measured_capability.json.
"""

# Each phi has: task_types (match the task_type column), metrics (substring match on
# metric_name), sources (match the source column). All given conditions must hold.
PHI_DATA_RULES = {
    "phi_1": {
        "name": "constraint_filter",
        "description": "Binary fit/no-fit based on hard constraints (context window, format support)",
        "task_types": ["metadata"],
        "metrics": ["context_length"],
        "sources": ["openrouter"],
        "score_type": "raw",  # not normalized 0-1
    },
    "phi_2": {
        "name": "domain_classifier",
        "description": "Task-domain classification strength per model",
        "task_types": ["knowledge", "domain_medical", "domain_legal"],
        "metrics": ["mmlu", "medqa", "legalbench"],
        "sources": ["routereval", "epoch_ai", "open_llm_leaderboard"],
        "score_type": "normalized",
    },
    "phi_3": {
        "name": "complexity_estimator",
        "description": "Surface-level task complexity signal",
        "task_types": ["reasoning", "math", "knowledge"],
        "metrics": ["arc", "hellaswag", "winogrande"],
        "sources": ["routereval", "epoch_ai", "open_llm_leaderboard"],
        "score_type": "normalized",
    },
    "phi_4": {
        "name": "performance_predictor",
        "description": "Expected quality score per model per task type",
        "task_types": ["general", "reasoning", "math", "coding", "knowledge", "instruction_following"],
        "metrics": ["correctness", "accuracy", "win_rate", "resolved"],
        "sources": ["routereval", "routellm", "open_llm_leaderboard", "epoch_ai", "alpaca_eval", "swebench"],
        "score_type": "normalized",
    },
    "phi_5": {
        "name": "context_fitness",
        "description": "Content relevance between task and model training distribution",
        "task_types": [],  # limited data, gated datasets
        "metrics": [],
        "sources": ["chatbot_arena", "lmsys_chat_1m"],
        "score_type": "normalized",
        "note": "Gated datasets not yet downloaded. Low data coverage.",
    },
    "phi_6": {
        "name": "historical_success",
        "description": "Past success rate of model on similar tasks",
        "task_types": ["general"],
        "metrics": ["routellm_quality", "pairwise_win", "routereval"],
        "sources": ["routellm", "routereval"],
        "score_type": "normalized",
    },
    "phi_7": {
        "name": "uncertainty_score",
        "description": "Router's epistemic uncertainty about model capability",
        "task_types": [],
        "metrics": [],
        "sources": [],
        "score_type": "derived",
        "note": "Runtime metric. Computed from data sparsity, not from benchmarks directly.",
    },
    "phi_8": {
        "name": "cascade_position",
        "description": "Where model sits in cost-quality cascade",
        "task_types": ["general", "metadata"],
        "metrics": ["routellm_quality", "routereval", "price_per_token"],
        "sources": ["routellm", "routereval", "openrouter"],
        "score_type": "derived",
    },
    "phi_9": {
        "name": "load_availability",
        "description": "Current model availability and load",
        "task_types": [],
        "metrics": [],
        "sources": [],
        "score_type": "runtime",
        "note": "Pure runtime metric. No historical data needed.",
    },
    "phi_10": {
        "name": "cross_node_capability",
        "description": "Multi-model ensemble potential",
        "task_types": [],
        "metrics": [],
        "sources": [],
        "score_type": "derived",
        "note": "Derived from pairwise model comparisons at runtime.",
    },
    "phi_11": {
        "name": "calibration_quality",
        "description": "Model's self-knowledge (ECE-like)",
        "task_types": [],
        "metrics": [],
        "sources": [],
        "score_type": "derived",
        "note": "From research papers in auto-search DB. Limited direct data.",
    },
    "phi_12": {
        "name": "output_structure_compliance",
        "description": "Structured output / function calling quality gradient",
        "task_types": [],
        "metrics": [],
        "sources": [],
        "score_type": "normalized",
        "note": "Berkeley Function Calling Leaderboard not yet downloaded.",
    },
    "phi_13": {
        "name": "reasoning_depth",
        "description": "Multi-step reasoning capability",
        "task_types": ["math", "reasoning", "coding"],
        "metrics": ["gsm8k", "math", "arc", "swebench", "resolved"],
        "sources": ["open_llm_leaderboard", "routereval", "swebench"],
        "score_type": "normalized",
    },
    "phi_14": {
        "name": "cost_efficiency_ratio",
        "description": "Task-conditional quality per dollar",
        "task_types": ["metadata", "cost"],
        "metrics": ["price_per_token", "cost", "routellm_quality"],
        "sources": ["openrouter", "routellm"],
        "score_type": "derived",
    },
    "phi_15": {
        "name": "context_window_utilization",
        "description": "Performance degradation with context length",
        "task_types": ["long_context"],
        "metrics": ["ruler", "needle", "longbench", "context"],
        "sources": [],
        "score_type": "derived",
        "note": "Limited data. RULER degradation curves not in current datasets.",
    },
    "phi_16": {
        "name": "instruction_adherence",
        "description": "Does the model follow instructions precisely",
        "task_types": ["instruction_following"],
        "metrics": ["win_rate", "winrate", "ifeval", "mt_bench", "alpaca"],
        "sources": ["alpaca_eval"],
        "score_type": "normalized",
    },
}
