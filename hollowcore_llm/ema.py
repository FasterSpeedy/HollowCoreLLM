import copy

import torch
import torch.nn as nn


class EMATargetEncoder(nn.Module):
    def __init__(self, model: nn.Module, decay: float = 0.996) -> None:
        super().__init__()
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        for param in self.shadow.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for s_param, param in zip(self.shadow.parameters(), model.parameters(), strict=True):
            s_param.data.mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    @torch.no_grad()
    def forward_view(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.shadow.forward_view(input_ids)

    @torch.no_grad()
    def forward_view_jepa(
        self, input_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.forward_view(input_ids)
        thought = self.shadow.jepa.to_thought(hidden)
        return hidden, thought

    def state_dict_for_checkpoint(self) -> dict[str, torch.Tensor]:
        return self.shadow.state_dict()

    def load_state_dict_from_checkpoint(self, state: dict[str, torch.Tensor]) -> None:
        self.shadow.load_state_dict(state)

    def forward(self, *args, **kwargs):
        return self.shadow(*args, **kwargs)
