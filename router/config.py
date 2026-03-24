"""Router configuration: endpoint definitions, seed weight loader, pipeline settings.

Loads seed weights from data/seed_weights.json and defines the available
endpoints (LLMs, agents, scripts, MCP tools) the router can select from.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List

from router.models import Endpoint, EndpointType, TaskType

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SEED_WEIGHTS_PATH = DATA_DIR / "seed_weights.json"
LEARNED_WEIGHTS_PATH = DATA_DIR / "learned_weights.json"
NORMALIZED_DB_PATH = DATA_DIR / "auto_search.db"

# ---------------------------------------------------------------------------
# Pipeline settings
# ---------------------------------------------------------------------------

PIPELINE = {
    "top_k": 3,
    "min_gate_pass": True,  # require all gates to pass
    "use_entropy_regularization": True,
    "decision_log_path": str(DATA_DIR / "routing_decisions.jsonl"),
}

# ---------------------------------------------------------------------------
# Endpoint definitions
# ---------------------------------------------------------------------------

LLM_ENDPOINTS: List[Endpoint] = [
    Endpoint(
        id="claude-opus-4",
        name="Claude Opus 4",
        endpoint_type=EndpointType.LLM,
        provider="anthropic",
        model_id="claude-opus-4-20250514",
        context_window=200000,
        supports_thinking=True,
        supports_structured_output=True,
        supports_vision=True,
        cost_per_million_input=15.0,
        cost_per_million_output=75.0,
        tags=["reasoning", "coding", "general"],
    ),
    Endpoint(
        id="claude-sonnet-4",
        name="Claude Sonnet 4",
        endpoint_type=EndpointType.LLM,
        provider="anthropic",
        model_id="claude-sonnet-4-20250514",
        context_window=200000,
        supports_thinking=True,
        supports_structured_output=True,
        supports_vision=True,
        cost_per_million_input=3.0,
        cost_per_million_output=15.0,
        tags=["coding", "general", "instruction_following"],
    ),
    Endpoint(
        id="gpt-4o",
        name="GPT-4o",
        endpoint_type=EndpointType.LLM,
        provider="openai",
        model_id="gpt-4o",
        context_window=128000,
        supports_thinking=False,
        supports_structured_output=True,
        supports_vision=True,
        cost_per_million_input=2.50,
        cost_per_million_output=10.0,
        tags=["general", "knowledge", "vision"],
    ),
    Endpoint(
        id="deepseek-r1",
        name="DeepSeek R1",
        endpoint_type=EndpointType.LLM,
        provider="hf",
        model_id="deepseek-ai/DeepSeek-R1",
        context_window=128000,
        supports_thinking=True,
        supports_structured_output=False,
        cost_per_million_input=0.70,
        cost_per_million_output=2.50,
        tags=["reasoning", "math", "coding"],
    ),
    Endpoint(
        id="deepseek-v3",
        name="DeepSeek V3",
        endpoint_type=EndpointType.LLM,
        provider="hf",
        model_id="deepseek-ai/DeepSeek-V3",
        context_window=128000,
        supports_thinking=False,
        supports_structured_output=True,
        cost_per_million_input=0.28,
        cost_per_million_output=0.42,
        tags=["general", "coding", "cost"],
    ),
    Endpoint(
        id="qwen3-coder",
        name="Qwen3 Coder 480B",
        endpoint_type=EndpointType.LLM,
        provider="hf",
        model_id="Qwen/Qwen3-Coder-480B-A35B-Instruct",
        context_window=262144,
        supports_thinking=True,
        supports_structured_output=True,
        cost_per_million_input=0.30,
        cost_per_million_output=1.30,
        tags=["coding", "reasoning"],
    ),
    Endpoint(
        id="gemini-2.5-pro",
        name="Gemini 2.5 Pro",
        endpoint_type=EndpointType.LLM,
        provider="google",
        model_id="gemini-2.5-pro-preview-06-05",
        context_window=1048576,
        supports_thinking=True,
        supports_structured_output=True,
        supports_vision=True,
        cost_per_million_input=1.25,
        cost_per_million_output=10.0,
        tags=["long_context", "reasoning", "general"],
    ),
]

AGENT_ENDPOINTS: List[Endpoint] = [
    Endpoint(
        id="auto-search",
        name="Auto-Search Agent",
        endpoint_type=EndpointType.AGENT,
        model_id="agents.auto_search",
        tags=["research", "knowledge"],
        metadata={"entry": "python -m agents.auto_search.run"},
    ),
    Endpoint(
        id="ramanujan",
        name="Ramanujan Verification Engine",
        endpoint_type=EndpointType.AGENT,
        model_id="agents.ramanujan",
        tags=["math", "reasoning", "verification"],
        metadata={"entry": "python -m agents.ramanujan.run"},
    ),
    Endpoint(
        id="thinking",
        name="First Principles Thinking Agent",
        endpoint_type=EndpointType.AGENT,
        model_id="agents.thinking",
        tags=["reasoning", "strategy"],
        metadata={"entry": "python -m agents.thinking"},
    ),
]

SCRIPT_ENDPOINTS: List[Endpoint] = [
    Endpoint(
        id="session-capture",
        name="Session Capture Agent",
        endpoint_type=EndpointType.SCRIPT,
        model_id="scripts.session_capture_agent",
        tags=["memory", "metadata"],
        metadata={"entry": "python scripts/session_capture_agent.py"},
    ),
]

MCP_ENDPOINTS: List[Endpoint] = [
    Endpoint(
        id="mcp-notion",
        name="Notion MCP",
        endpoint_type=EndpointType.MCP_TOOL,
        model_id="notion",
        tags=["knowledge", "writing"],
        metadata={"tools": ["search", "fetch", "create-pages", "update-page"]},
    ),
    Endpoint(
        id="mcp-google-workspace",
        name="Google Workspace MCP",
        endpoint_type=EndpointType.MCP_TOOL,
        model_id="google-workspace",
        tags=["email", "calendar", "docs"],
        metadata={"tools": ["gmail", "drive", "calendar", "docs", "sheets"]},
    ),
]

ALL_ENDPOINTS = LLM_ENDPOINTS + AGENT_ENDPOINTS + SCRIPT_ENDPOINTS + MCP_ENDPOINTS

# ---------------------------------------------------------------------------
# Model ID aliases (map common names to our endpoint IDs)
# ---------------------------------------------------------------------------

MODEL_ALIASES: Dict[str, str] = {
    # Anthropic
    "claude": "claude-sonnet-4",
    "opus": "claude-opus-4",
    "sonnet": "claude-sonnet-4",
    # OpenAI
    "gpt4": "gpt-4o",
    "gpt-4": "gpt-4o",
    "chatgpt": "gpt-4o",
    # DeepSeek
    "deepseek": "deepseek-v3",
    "r1": "deepseek-r1",
    # Qwen
    "qwen": "qwen3-coder",
    # Google
    "gemini": "gemini-2.5-pro",
    # Agents
    "search": "auto-search",
    "verify": "ramanujan",
    "think": "thinking",
    # MCP
    "notion": "mcp-notion",
    "google": "mcp-google-workspace",
    "gmail": "mcp-google-workspace",
    "drive": "mcp-google-workspace",
}


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------

def load_seed_weights() -> Dict[str, Any]:
    """Load seed weights from data/seed_weights.json."""
    if not SEED_WEIGHTS_PATH.exists():
        logger.warning(f"Seed weights not found at {SEED_WEIGHTS_PATH}, using uniform")
        return _uniform_weights()

    with open(SEED_WEIGHTS_PATH) as f:
        data = json.load(f)

    logger.info(f"Loaded seed weights v={data.get('version', 'unknown')}")
    return data


def load_learned_weights() -> Dict[str, float]:
    """Load learned weights if they exist, otherwise return seed weights."""
    if LEARNED_WEIGHTS_PATH.exists():
        with open(LEARNED_WEIGHTS_PATH) as f:
            data = json.load(f)
        logger.info(f"Loaded learned weights (n={data.get('decision_count', 0)} decisions)")
        return data.get("weights", {})

    seed = load_seed_weights()
    return seed.get("weights", _uniform_weights()["weights"])


def save_learned_weights(weights: Dict[str, float], decision_count: int):
    """Persist learned weights to disk."""
    data = {
        "weights": weights,
        "decision_count": decision_count,
        "version": "learned",
    }
    with open(LEARNED_WEIGHTS_PATH, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved learned weights (n={decision_count})")


def _uniform_weights() -> Dict[str, Any]:
    """Fallback: uniform weights across 16 phi functions."""
    w = 1.0 / 16
    return {
        "version": "uniform-fallback",
        "weights": {f"phi_{i}": w for i in range(1, 17)},
        "beta": {"task_conditional": {t.value: 0.5 for t in TaskType}},
        "entropy_regularization": {
            "initial_lambda": 0.5,
            "decay_rate": 0.995,
            "min_lambda": 0.01,
        },
    }
