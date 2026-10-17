"""Loader for prompts/system_prompts.yaml.

Keeps the YAML as the single source of truth: categories are fetched by id or
category name, and the shared fragments ({spatial_rule}, {grounded_contract},
...) are spliced into each system prompt at load time.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from pipeline.config_util import PROJECT_ROOT

DEFAULT_PROMPTS_PATH = PROJECT_ROOT / "prompts" / "system_prompts.yaml"


def _collapse(text: str) -> str:
    """YAML folded scalars keep newlines in odd places; normalise whitespace so
    prompts don't carry ragged formatting into the model context."""
    return " ".join(str(text).split())


@dataclass
class PromptEntry:
    id: str
    category: str
    purpose: str
    version: str
    system_prompt: str
    few_shot: list[dict[str, str]]
    templates: dict[str, str]
    known_failure_modes: list[str]
    raw: dict[str, Any]


class PromptLibrary:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else DEFAULT_PROMPTS_PATH
        with open(self.path, "r", encoding="utf-8") as f:
            self.raw = yaml.safe_load(f)
        self.meta = self.raw.get("meta", {})
        self.shared = {k: _collapse(v) for k, v in (self.raw.get("shared") or {}).items()}
        self._entries: dict[str, PromptEntry] = {}
        self._by_category: dict[str, PromptEntry] = {}
        for item in self.raw.get("categories", []):
            entry = self._build(item)
            self._entries[entry.id] = entry
            # First entry wins as the default for its category.
            self._by_category.setdefault(entry.category, entry)

    def _build(self, item: dict[str, Any]) -> PromptEntry:
        system_prompt = _collapse(item.get("system_prompt", ""))
        # Splice shared fragments. Unknown placeholders are left untouched
        # rather than raising, so adding a fragment to the YAML can't crash the
        # pipeline mid-session.
        for key, value in self.shared.items():
            system_prompt = system_prompt.replace("{" + key + "}", value)
        few_shot = [
            {"user": _collapse(ex.get("user", "")), "assistant": _collapse(ex.get("assistant", ""))}
            for ex in (item.get("few_shot") or [])
        ]
        return PromptEntry(
            id=item["id"],
            category=item["category"],
            purpose=_collapse(item.get("purpose", "")),
            version=item.get("version", "0.0.0"),
            system_prompt=system_prompt,
            few_shot=few_shot,
            templates=item.get("templates") or {},
            known_failure_modes=list(item.get("known_failure_modes") or []),
            raw=item,
        )

    def get(self, prompt_id: str) -> PromptEntry:
        if prompt_id not in self._entries:
            raise KeyError(f"No prompt with id {prompt_id!r}. Have: {list(self._entries)}")
        return self._entries[prompt_id]

    def by_category(self, category: str) -> PromptEntry:
        if category not in self._by_category:
            raise KeyError(f"No prompt for category {category!r}. Have: {list(self._by_category)}")
        return self._by_category[category]

    def ids(self) -> list[str]:
        return list(self._entries)

    def categories(self) -> list[str]:
        return list(self._by_category)
