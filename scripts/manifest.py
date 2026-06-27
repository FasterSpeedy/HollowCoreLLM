from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class CheckpointManifest:
    step: int
    seq_len: int
    loss: float
    seed: int
    profile: str
    dataset_mix: dict[str, float]
    view: str = "text_code"
    epoch: int = 0
    lineage_id: str = ""
    parent_lineage_id: str = ""
    branch_owner: str = ""
    checkpoint_path: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8", newline="\n") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> CheckpointManifest:
        with open(path, encoding="utf-8-sig") as f:
            data = json.load(f)
        return cls(**data)


def new_lineage_id() -> str:
    return str(uuid.uuid4())


def assert_resume_compatible(parent: CheckpointManifest, profile: str, allow_branch_takeover: bool = False) -> None:
    if parent.branch_owner and parent.branch_owner != profile and not allow_branch_takeover:
        raise RuntimeError(
            f"Branch conflict: checkpoint owned by profile={parent.branch_owner!r}, "
            f"resume requested by profile={profile!r}. "
            "Parallel train from same checkpoint creates divergent branches. "
            "Pick one branch or set --allow-branch-takeover after explicit merge plan."
        )
