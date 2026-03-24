"""Canonical model identity resolution for S(M,T) bilinear phi training.

Maps any model name alias across all datasets to a single canonical model ID.
This is the foundation for unified training: without it, training data fragments
across aliases and model embeddings can't learn from all available signal.

Usage:
    registry = ModelRegistry()
    canonical = registry.resolve("gpt-4")          # -> "gpt-4-1106-preview"
    canonical = registry.resolve("mixtral-8x7b")    # -> "mistralai/mixtral-8x7b-chat"
    idx = registry.get_embedding_index("gpt-4-1106-preview")  # -> stable int
"""

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent / "data"
DEFAULT_REGISTRY_PATH = DATA_DIR / "model_registry.json"


class ModelRegistry:
    """Canonical model identity resolution.

    Maps any alias string to a canonical model ID. Builds O(1) alias lookup
    from three sources, merged in order:
      1. Seed registry JSON (data/model_registry.json)
      2. Dataset-specific extraction during ingestion
      3. Manual additions via add_alias() / register_model()
    """

    def __init__(self, registry_path: Optional[Path] = None):
        self.registry_path = Path(registry_path or DEFAULT_REGISTRY_PATH)
        self._data: dict = {"version": 1, "models": {}}
        self._alias_to_canonical: dict[str, str] = {}
        self._canonical_to_index: dict[str, int] = {}

        if self.registry_path.exists():
            with open(self.registry_path) as f:
                self._data = json.load(f)
            logger.info(f"Loaded {len(self._data['models'])} models from {self.registry_path}")
        else:
            logger.warning(f"Registry not found at {self.registry_path}, starting empty")

        self._build_alias_index()
        self._build_embedding_index()

    def _build_alias_index(self):
        """Build case-insensitive alias -> canonical_id lookup."""
        self._alias_to_canonical = {}
        for canonical_id, model in self._data["models"].items():
            # The canonical ID itself is always an alias
            normalized = canonical_id.strip().lower()
            self._alias_to_canonical[normalized] = canonical_id

            for alias in model.get("aliases", []):
                norm_alias = alias.strip().lower()
                if norm_alias in self._alias_to_canonical:
                    existing = self._alias_to_canonical[norm_alias]
                    if existing != canonical_id:
                        logger.warning(
                            f"Alias conflict: '{alias}' maps to both "
                            f"'{existing}' and '{canonical_id}'. Keeping '{existing}'."
                        )
                else:
                    self._alias_to_canonical[norm_alias] = canonical_id

        logger.info(f"Built alias index: {len(self._alias_to_canonical)} aliases -> "
                     f"{len(self._data['models'])} canonical models")

    def _build_embedding_index(self):
        """Build stable canonical_id -> integer index for embedding lookup.

        Uses sorted alphabetical order of canonical IDs so the mapping
        is deterministic and reproducible across runs.
        """
        sorted_ids = sorted(self._data["models"].keys())
        self._canonical_to_index = {cid: i for i, cid in enumerate(sorted_ids)}

    def resolve(self, raw_model_id: str) -> Optional[str]:
        """Resolve any alias to its canonical model ID.

        Args:
            raw_model_id: Any model name string from any dataset.

        Returns:
            Canonical model ID if found, None if unknown.
        """
        if not raw_model_id:
            return None
        normalized = raw_model_id.strip().lower()
        return self._alias_to_canonical.get(normalized)

    def register_model(
        self,
        canonical_id: str,
        provider: str,
        family: str,
        aliases: Optional[list[str]] = None,
        metadata: Optional[dict] = None,
        sources: Optional[list[str]] = None,
    ):
        """Add a new model to the registry.

        Args:
            canonical_id: The unique canonical identifier.
            provider: Model provider (e.g., "openai", "meta").
            family: Model family (e.g., "gpt-4", "llama-2").
            aliases: Known alternative names.
            metadata: Context window, costs, etc.
            sources: Datasets this model appears in.
        """
        if canonical_id in self._data["models"]:
            logger.warning(f"Model '{canonical_id}' already exists, updating")

        self._data["models"][canonical_id] = {
            "canonical_id": canonical_id,
            "provider": provider,
            "family": family,
            "aliases": aliases or [],
            "metadata": metadata or {},
            "sources": sources or [],
        }
        self._build_alias_index()
        self._build_embedding_index()

    def add_alias(self, canonical_id: str, new_alias: str) -> bool:
        """Add an alias to an existing canonical model.

        Args:
            canonical_id: Must already exist in registry.
            new_alias: The new alias string to add.

        Returns:
            True if added, False if canonical_id not found.
        """
        if canonical_id not in self._data["models"]:
            logger.error(f"Cannot add alias: '{canonical_id}' not in registry")
            return False

        aliases = self._data["models"][canonical_id].get("aliases", [])
        if new_alias not in aliases:
            aliases.append(new_alias)
            self._data["models"][canonical_id]["aliases"] = aliases
            self._build_alias_index()

        return True

    def add_source(self, canonical_id: str, source: str) -> bool:
        """Record that a model appears in a given dataset source.

        Args:
            canonical_id: Must already exist in registry.
            source: Dataset source name (e.g., "routerbench", "lmsys_arena").

        Returns:
            True if added, False if canonical_id not found.
        """
        if canonical_id not in self._data["models"]:
            return False

        sources = self._data["models"][canonical_id].get("sources", [])
        if source not in sources:
            sources.append(source)
            self._data["models"][canonical_id]["sources"] = sources

        return True

    def get_embedding_index(self, canonical_id: str) -> Optional[int]:
        """Get stable integer index for nn.Embedding lookup.

        Returns:
            Integer index (0-based, sorted alphabetical), or None if not found.
        """
        return self._canonical_to_index.get(canonical_id)

    @property
    def num_models(self) -> int:
        """Total number of canonical models in registry."""
        return len(self._data["models"])

    @property
    def num_aliases(self) -> int:
        """Total number of alias mappings."""
        return len(self._alias_to_canonical)

    def list_models(self, min_sources: int = 1) -> list[str]:
        """List canonical model IDs, optionally filtered by data coverage.

        Args:
            min_sources: Minimum number of dataset sources required.

        Returns:
            List of canonical model IDs meeting the threshold.
        """
        result = []
        for cid, model in self._data["models"].items():
            if len(model.get("sources", [])) >= min_sources:
                result.append(cid)
        return sorted(result)

    def list_families(self) -> dict[str, list[str]]:
        """Group canonical models by family.

        Returns:
            Dict of family -> list of canonical IDs.
        """
        families: dict[str, list[str]] = {}
        for cid, model in self._data["models"].items():
            family = model.get("family", "unknown")
            families.setdefault(family, []).append(cid)
        return families

    def get_model(self, canonical_id: str) -> Optional[dict]:
        """Get full model record by canonical ID."""
        return self._data["models"].get(canonical_id)

    def export_for_training(self) -> dict[str, int]:
        """Export {canonical_id: embedding_index} mapping for training.

        Returns:
            Dict mapping every canonical model ID to its stable integer index.
        """
        return dict(self._canonical_to_index)

    def find_unresolved(self, model_ids: list[str]) -> list[str]:
        """Find model IDs that don't resolve to any canonical model.

        Args:
            model_ids: Raw model ID strings from a dataset.

        Returns:
            List of IDs that couldn't be resolved.
        """
        return [mid for mid in model_ids if self.resolve(mid) is None]

    def save(self, path: Optional[Path] = None):
        """Persist the registry to disk.

        Args:
            path: Output path. Defaults to the path the registry was loaded from.
        """
        out_path = Path(path or self.registry_path)
        with open(out_path, "w") as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {self.num_models} models to {out_path}")

    def stats(self) -> dict:
        """Summary statistics about the registry."""
        providers = set()
        families = set()
        total_aliases = 0
        all_sources = set()

        for model in self._data["models"].values():
            providers.add(model.get("provider", "unknown"))
            families.add(model.get("family", "unknown"))
            total_aliases += len(model.get("aliases", []))
            all_sources.update(model.get("sources", []))

        return {
            "num_models": self.num_models,
            "num_aliases": self.num_aliases,
            "num_providers": len(providers),
            "providers": sorted(providers),
            "num_families": len(families),
            "families": sorted(families),
            "total_alias_entries": total_aliases,
            "data_sources": sorted(all_sources),
        }

    def __repr__(self) -> str:
        return f"ModelRegistry({self.num_models} models, {self.num_aliases} aliases)"
