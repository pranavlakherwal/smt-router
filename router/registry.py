"""Endpoint registry for the S(M,T) router.

Manages the set of available endpoints (LLMs, agents, scripts, MCP tools).
Loads from config, supports runtime registration/deregistration, and
provides lookup by ID or alias.
"""

import logging
from typing import Dict, List, Optional

from agents.router.models import Endpoint, EndpointType
from agents.router.config import (
    ALL_ENDPOINTS,
    MODEL_ALIASES,
    LLM_ENDPOINTS,
    AGENT_ENDPOINTS,
    SCRIPT_ENDPOINTS,
    MCP_ENDPOINTS,
)

logger = logging.getLogger(__name__)


class EndpointRegistry:
    """Registry of all available endpoints."""

    def __init__(self, load_defaults: bool = True):
        self._endpoints: Dict[str, Endpoint] = {}
        self._aliases: Dict[str, str] = dict(MODEL_ALIASES)

        if load_defaults:
            for ep in ALL_ENDPOINTS:
                self.register(ep)
            logger.info(
                f"Registry loaded: {len(self._endpoints)} endpoints "
                f"({self.count_by_type()})"
            )

    def register(self, endpoint: Endpoint):
        """Register an endpoint."""
        self._endpoints[endpoint.id] = endpoint

    def deregister(self, endpoint_id: str) -> bool:
        """Remove an endpoint from the registry."""
        if endpoint_id in self._endpoints:
            del self._endpoints[endpoint_id]
            return True
        return False

    def get(self, id_or_alias: str) -> Optional[Endpoint]:
        """Look up endpoint by ID or alias."""
        # Direct ID lookup
        if id_or_alias in self._endpoints:
            return self._endpoints[id_or_alias]
        # Alias lookup
        resolved = self._aliases.get(id_or_alias.lower())
        if resolved and resolved in self._endpoints:
            return self._endpoints[resolved]
        return None

    def get_all(self) -> List[Endpoint]:
        """Get all registered endpoints."""
        return list(self._endpoints.values())

    def get_by_type(self, endpoint_type: EndpointType) -> List[Endpoint]:
        """Get endpoints of a specific type."""
        return [
            ep for ep in self._endpoints.values()
            if ep.endpoint_type == endpoint_type
        ]

    def get_available(self) -> List[Endpoint]:
        """Get all available endpoints."""
        return [ep for ep in self._endpoints.values() if ep.available]

    def get_by_tag(self, tag: str) -> List[Endpoint]:
        """Get endpoints that have a specific tag."""
        return [ep for ep in self._endpoints.values() if tag in ep.tags]

    def set_availability(self, endpoint_id: str, available: bool):
        """Update an endpoint's availability."""
        ep = self._endpoints.get(endpoint_id)
        if ep:
            ep.available = available

    def add_alias(self, alias: str, endpoint_id: str):
        """Add a new alias mapping."""
        self._aliases[alias.lower()] = endpoint_id

    def count_by_type(self) -> Dict[str, int]:
        """Count endpoints by type."""
        counts = {}
        for ep in self._endpoints.values():
            t = ep.endpoint_type.value
            counts[t] = counts.get(t, 0) + 1
        return counts

    def summary(self) -> str:
        """Human-readable registry summary."""
        lines = [f"Endpoint Registry: {len(self._endpoints)} total"]
        for ep_type in EndpointType:
            eps = self.get_by_type(ep_type)
            if eps:
                lines.append(f"  {ep_type.value}: {len(eps)}")
                for ep in eps:
                    status = "OK" if ep.available else "DOWN"
                    lines.append(f"    - {ep.id} ({ep.name}) [{status}]")
        return "\n".join(lines)
