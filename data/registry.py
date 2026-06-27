from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hollowcore_llm.config import TrainCurriculumConfig

_DEFAULT_PATH = Path(__file__).resolve().parent / "datasets.json"


@dataclass(frozen=True)
class DataSourceSpec:
    kind: str
    dataset_id: str
    split: str = "train"
    text_key: str = "text"
    chat_key: str = "chat"
    config: str | None = None

    @classmethod
    def from_dict(cls, name: str, raw: dict[str, Any]) -> DataSourceSpec:
        kind = raw.get("kind")
        if kind not in ("hf_text", "glaive"):
            raise ValueError(f"source {name!r}: unknown kind {kind!r} (use hf_text or glaive)")
        dataset_id = raw.get("dataset_id")
        if not dataset_id:
            raise ValueError(f"source {name!r}: dataset_id is required")
        return cls(
            kind=kind,
            dataset_id=str(dataset_id),
            split=str(raw.get("split", "train")),
            text_key=str(raw.get("text_key", "text")),
            chat_key=str(raw.get("chat_key", "chat")),
            config=(str(raw["config"]) if raw.get("config") else None),
        )


@dataclass(frozen=True)
class DatasetRegistry:
    sources: dict[str, DataSourceSpec]
    curriculum: TrainCurriculumConfig

    @classmethod
    def load(cls, path: str | Path | None = None) -> DatasetRegistry:
        resolved = Path(path or os.environ.get("HOLLOWCORE_DATASETS_JSON", _DEFAULT_PATH))
        with open(resolved, encoding="utf-8-sig") as f:
            data = json.load(f)
        return cls.from_dict(data, resolved)

    @classmethod
    def from_dict(cls, data: dict[str, Any], path: Path | None = None) -> DatasetRegistry:
        raw_sources = data.get("sources")
        if not isinstance(raw_sources, dict) or not raw_sources:
            raise ValueError("datasets config: sources must be a non-empty object")

        sources = {name: DataSourceSpec.from_dict(name, spec) for name, spec in raw_sources.items()}
        curriculum = _curriculum_from_dict(data.get("curriculum", {}), tuple(sources.keys()), path)
        _validate_curriculum(curriculum, sources)
        return cls(sources=sources, curriculum=curriculum)

    def category_names(self) -> tuple[str, ...]:
        return tuple(self.sources.keys())


def _curriculum_from_dict(
    raw: dict[str, Any],
    source_keys: tuple[str, ...],
    path: Path | None,
) -> TrainCurriculumConfig:
    initial = raw.get("initial_mix") or {"stable": 1.0}
    mixed = raw.get("mixed_proportions") or {k: 1.0 / len(source_keys) for k in source_keys}
    return TrainCurriculumConfig(
        stable_steps=int(raw.get("stable_steps", 200)),
        mix_start_step=int(raw.get("mix_start_step", 200)),
        categories=source_keys,
        initial_mix={k: float(v) for k, v in initial.items()},
        mixed_proportions={k: float(v) for k, v in mixed.items()},
        replay_floor=float(raw.get("replay_floor", 0.10)),
    )


def _validate_curriculum(curriculum: TrainCurriculumConfig, sources: dict[str, DataSourceSpec]) -> None:
    for key in curriculum.initial_mix:
        if key not in sources:
            raise ValueError(f"initial_mix key {key!r} not in sources")
    for key in curriculum.mixed_proportions:
        if key not in sources:
            raise ValueError(f"mixed_proportions key {key!r} not in sources")
    if not curriculum.initial_mix:
        raise ValueError("curriculum.initial_mix cannot be empty")
