from __future__ import annotations

import random
from dataclasses import dataclass

from hollowcore_llm.config import TrainCurriculumConfig


@dataclass
class CurriculumSampler:
    cfg: TrainCurriculumConfig
    rng: random.Random

    def mix_at_step(self, step: int) -> dict[str, float]:
        if step < self.cfg.mix_start_step:
            return dict(self.cfg.initial_mix)
        return self._with_replay(self.cfg.mixed_proportions)

    def _with_replay(self, target: dict[str, float]) -> dict[str, float]:
        out = dict(target)
        total = sum(out.values())
        if total <= 0:
            return {"stable": 1.0}
        out = {k: v / total for k, v in out.items()}
        floor = self.cfg.replay_floor
        for cat in self.cfg.categories:
            if cat not in out:
                out[cat] = 0.0
        for cat in self.cfg.categories:
            if cat != "stable" and out.get(cat, 0) > 0:
                out["stable"] = max(out.get("stable", 0), floor)
        total = sum(out.values())
        return {k: v / total for k, v in out.items()}

    def pick_category(self, step: int) -> str:
        mix = self.mix_at_step(step)
        roll = self.rng.random()
        acc = 0.0
        for cat, weight in mix.items():
            acc += weight
            if roll <= acc:
                return cat
        return self.cfg.categories[-1]
