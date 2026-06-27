import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import HollowCoreConfig


class ToolPolicyHead(nn.Module):
    def __init__(self, cfg: HollowCoreConfig) -> None:
        super().__init__()
        self.decision = nn.Linear(cfg.hidden_size, 2, bias=True)
        self.tool = nn.Linear(cfg.hidden_size, cfg.num_tools, bias=True)

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.decision(hidden), self.tool(hidden)

    def loss(
        self,
        hidden: torch.Tensor,
        action_mask: torch.Tensor | None,
        tool_decision_labels: torch.Tensor | None,
        tool_id_labels: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        zero = hidden.new_zeros(())
        if action_mask is None or tool_decision_labels is None:
            return zero, zero

        decision_logits, tool_logits = self(hidden)
        mask = action_mask.bool()
        if not mask.any():
            return zero, zero

        decision_loss = F.cross_entropy(decision_logits[mask].float(), tool_decision_labels[mask].long())  #6
        if tool_id_labels is None:
            return decision_loss, zero

        call_mask = mask & (tool_decision_labels == 1) & (tool_id_labels >= 0)
        if not call_mask.any():
            return decision_loss, zero

        tool_loss = F.cross_entropy(tool_logits[call_mask].float(), tool_id_labels[call_mask].long())  #7
        return decision_loss, tool_loss
