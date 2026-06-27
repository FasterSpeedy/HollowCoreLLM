import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import HollowCoreConfig


class FullAttnResAccumulator(nn.Module):
    def __init__(self, cfg: HollowCoreConfig, num_steps: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_steps = num_steps
        self.queries = nn.Parameter(torch.randn(num_steps, cfg.hidden_size) * cfg.initializer_std)

    def combine(self, step_idx: int, values: list[torch.Tensor], h: torch.Tensor) -> torch.Tensor:
        if not values:
            return h
        q = self.queries[step_idx].to(dtype=torch.float32, device=h.device)
        scores = torch.stack([(v.float() * q).sum(dim=-1) for v in values], dim=-1)
        weights = F.softmax(scores, dim=-1)
        return sum(weights[..., i : i + 1].to(values[i].dtype) * values[i] for i in range(len(values)))

    def reset(self) -> list[torch.Tensor]:
        return []
