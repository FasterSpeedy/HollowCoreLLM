import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import HollowCoreConfig
from .sigreg import sigreg_loss


class JEPAHead(nn.Module):
    def __init__(self, cfg: HollowCoreConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.to_thought = nn.Linear(cfg.hidden_size, cfg.thought_dim, bias=False)
        self.cross_view_pred = nn.Sequential(
            nn.Linear(cfg.thought_dim, cfg.thought_dim, bias=False),
            nn.GELU(),
            nn.Linear(cfg.thought_dim, cfg.thought_dim, bias=False),
        )
        self.chunk_pred = nn.Sequential(
            nn.Linear(cfg.thought_dim, cfg.thought_dim, bias=False),
            nn.GELU(),
            nn.Linear(cfg.thought_dim, cfg.thought_dim, bias=False),
        )

    def _cos_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (1.0 - F.cosine_similarity(pred.float(), target.float(), dim=-1)).mean()

    def to_thought_vec(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.to_thought(hidden)

    def cross_view_loss(
        self,
        hidden_a: torch.Tensor,
        hidden_b: torch.Tensor | None = None,
        thought_b: torch.Tensor | None = None,
    ) -> torch.Tensor:
        thought_a = self.to_thought(hidden_a)
        if thought_b is None:
            if hidden_b is None:
                return hidden_a.new_zeros(())
            thought_b = self.to_thought(hidden_b)
        thought_b = thought_b.detach()
        if thought_a.size(1) < 1 or thought_b.size(1) < 1:
            return hidden_a.new_zeros(())
        pred = self.cross_view_pred(thought_a.mean(dim=1))
        target = thought_b.mean(dim=1)
        return self._cos_loss(pred, target)  #4

    def chunk_loss(self, hidden: torch.Tensor) -> torch.Tensor:
        thought = self.to_thought(hidden)
        zero = hidden.new_zeros(())
        chunk = self.cfg.jepa_chunk_size
        usable = (thought.size(1) // chunk) * chunk
        if usable < chunk * 2:
            return zero
        chunks = thought[:, :usable].view(thought.size(0), -1, chunk, thought.size(-1)).mean(dim=2)
        return self._cos_loss(self.chunk_pred(chunks[:, :-1]), chunks[:, 1:].detach())  #5

    def sigreg_on_views(
        self,
        hidden_a: torch.Tensor,
        hidden_b: torch.Tensor | None = None,
        thought_b: torch.Tensor | None = None,
    ) -> torch.Tensor:
        thought_a = self.to_thought(hidden_a).mean(dim=1)
        if thought_b is None:
            if hidden_b is None:
                return hidden_a.new_zeros(())
            thought_b = self.to_thought(hidden_b).mean(dim=1)
        else:
            thought_b = thought_b.mean(dim=1)
        return sigreg_loss(torch.cat([thought_a, thought_b], dim=0))

    def losses(
        self,
        hidden: torch.Tensor,
        hidden_b: torch.Tensor | None = None,
        thought_b: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cross = hidden.new_zeros(())
        if hidden_b is not None or thought_b is not None:
            cross = self.cross_view_loss(hidden, hidden_b, thought_b=thought_b)
        chunk = self.chunk_loss(hidden)
        sig = hidden.new_zeros(())
        if hidden_b is not None or thought_b is not None:
            sig = self.sigreg_on_views(hidden, hidden_b, thought_b=thought_b)
        return cross, chunk, sig
