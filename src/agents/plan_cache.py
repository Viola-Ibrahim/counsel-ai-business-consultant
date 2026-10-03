"""
plan_cache.py
--------------
Saves LLM PLANS (never data) so the same plan is not paid for twice:
  - the data-prep plan, keyed by the dataset schema (column names + types)
  - the analysis plan, keyed by the schema + the question + the goal

Cached plans are still re-validated by the code guards on every use
(apply_prep_plan checks every decision against the real data, plan_analysis blocks
restricted columns), so a cached plan can never be less safe than a fresh one.

A plan holds column names, roles and tool choices: no rows and no values.
"""

import hashlib
from pathlib import Path
from typing import Optional, Type, TypeVar

import pandas as pd
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def make_key(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:20]
    return f"{prefix}_{digest}"


def schema_signature(df: pd.DataFrame) -> str:
    """Same columns + same types = same signature (column order does not matter)."""
    return "|".join(sorted(f"{c}:{df[c].dtype}" for c in df.columns))


class PlanCache:
    def __init__(self, directory: Optional[str] = None):
        self.memory = {}
        self.directory = Path(directory) if directory else None
        if self.directory:
            self.directory.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    def get(self, key: str, model: Type[T]) -> Optional[T]:
        if key in self.memory:
            self.hits += 1
            return self.memory[key]
        if self.directory and (self.directory / f"{key}.json").exists():
            try:
                obj = model.model_validate_json((self.directory / f"{key}.json").read_text(encoding="utf-8"))
                self.memory[key] = obj
                self.hits += 1
                return obj
            except Exception:  # corrupted or outdated file: treat as a miss
                pass
        self.misses += 1
        return None

    def put(self, key: str, obj: BaseModel) -> None:
        self.memory[key] = obj
        if self.directory:
            (self.directory / f"{key}.json").write_text(obj.model_dump_json(), encoding="utf-8")
