"""Pydantic models for the S(M,T) router.

Defines the core data structures: tasks, endpoints, scores, and decisions.
Endpoints generalize beyond LLMs to agents, scripts, and MCP tools.
"""

from enum import Enum
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class EndpointType(str, Enum):
    """Type of capability endpoint the router can select."""
    LLM = "llm"
    AGENT = "agent"
    SCRIPT = "script"
    MCP_TOOL = "mcp_tool"


class TaskType(str, Enum):
    """Task classification categories matching phi_mapper task_types."""
    GENERAL = "general"
    CODING = "coding"
    MATH = "math"
    REASONING = "reasoning"
    KNOWLEDGE = "knowledge"
    INSTRUCTION_FOLLOWING = "instruction_following"
    COST = "cost"
    METADATA = "metadata"
    SAFETY = "safety"
    LONG_CONTEXT = "long_context"


# ---------------------------------------------------------------------------
# Core models
# ---------------------------------------------------------------------------

class Task(BaseModel):
    """A user request decomposed into a routable task."""
    id: str = ""
    query: str = ""
    task_type: TaskType = TaskType.GENERAL
    complexity: float = 0.5  # 0-1 estimated complexity
    context_length: int = 0  # token count of input context
    requires_thinking: bool = False
    requires_structured_output: bool = False
    max_budget_usd: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Endpoint(BaseModel):
    """A capability endpoint (LLM, agent, script, or MCP tool)."""
    id: str
    name: str
    endpoint_type: EndpointType = EndpointType.LLM
    provider: str = ""  # openrouter, hf, local, etc.
    model_id: str = ""  # for LLMs: the API model identifier
    context_window: int = 128000
    supports_thinking: bool = False
    supports_structured_output: bool = False
    supports_vision: bool = False
    cost_per_million_input: float = 0.0
    cost_per_million_output: float = 0.0
    available: bool = True
    tags: List[str] = Field(default_factory=list)  # e.g., ["coding", "math"]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PhiScore(BaseModel):
    """Score from a single phi function evaluation."""
    phi_id: str  # phi_1 through phi_16
    name: str  # human-readable name
    score: float = 0.0  # typically 0-1
    confidence: float = 1.0  # how much data backs this score
    source: str = ""  # "data" or "prior" or "runtime"


class GateResult(BaseModel):
    """Result of a binary gate check."""
    gate_name: str
    passed: bool = True
    reason: str = ""


class ScoredEndpoint(BaseModel):
    """An endpoint with its full S(M,T) score breakdown."""
    endpoint: Endpoint
    gate_results: List[GateResult] = Field(default_factory=list)
    gates_passed: bool = True
    phi_scores: List[PhiScore] = Field(default_factory=list)
    weighted_phi_sum: float = 0.0  # Sum_i w_i * phi_i
    temperature_factor: float = 1.0  # e^(-beta * H_i)
    entropy_bonus: float = 0.0  # lambda * H(pi) contribution
    final_score: float = 0.0  # S(M,T) complete


class RoutingDecision(BaseModel):
    """The router's output: ranked endpoints with explanations."""
    task: Task
    scored_endpoints: List[ScoredEndpoint] = Field(default_factory=list)
    selected: Optional[ScoredEndpoint] = None
    top_k: List[ScoredEndpoint] = Field(default_factory=list)
    decision_number: int = 0  # for Robbins-Monro step tracking
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = Field(default_factory=dict)


class FeedbackSignal(BaseModel):
    """Post-execution feedback for weight learning."""
    decision_id: str = ""
    endpoint_id: str = ""
    task_type: TaskType = TaskType.GENERAL
    success: bool = True
    quality_score: float = 0.0  # 0-1 observed quality
    latency_seconds: float = 0.0
    cost_usd: float = 0.0
    user_rating: Optional[float] = None  # 1-5 if provided
    metadata: Dict[str, Any] = Field(default_factory=dict)
