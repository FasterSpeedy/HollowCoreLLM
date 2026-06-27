from pathlib import Path
from typing import Any

import torch

from .ema import EMATargetEncoder
from .model import HollowCoreLLM


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def build_optimizer(
    model: HollowCoreLLM,
    lr: float = 3e-4,
    weight_decay: float = 0.1,
    adam_8bit: bool = False,
) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith("bias") or "norm" in name or "A_log" in name or "dt_bias" in name:
            no_decay.append(param)
        else:
            decay.append(param)
    param_groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    if adam_8bit:
        try:
            import bitsandbytes as bnb

            return bnb.optim.AdamW8bit(param_groups, lr=lr, betas=(0.9, 0.95))
        except ImportError:
            print("WARN: bitsandbytes not installed — falling back to AdamW fp32")
    return torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.95))


def train_step(
    model: HollowCoreLLM,
    batch: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    ema: EMATargetEncoder | None = None,
    grad_clip: float = 1.0,
) -> dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)

    hidden_b = None
    thought_b = None
    if ema is not None and "input_ids_b" in batch:
        hidden_b, thought_b = ema.forward_view_jepa(batch["input_ids_b"])

    if model.cfg.chunk_train_size > 0:
        metrics = model._run_chunked_train(
            batch["input_ids"],
            batch["labels"],
            action_mask=batch.get("action_mask"),
            tool_decision_labels=batch.get("tool_decision_labels"),
            tool_id_labels=batch.get("tool_id_labels"),
            hidden_b=hidden_b,
            thought_b=thought_b,
        )
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if ema is not None:
            ema.update(model)
        return metrics

    out = model(
        **{k: v for k, v in batch.items() if k != "input_ids_b"},
        hidden_b=hidden_b,
        thought_b=thought_b,
    )
    if out.loss is None:
        raise RuntimeError("batch must include labels for training")
    out.loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    if ema is not None:
        ema.update(model)
    return out.metrics


def save_checkpoint(
    path: str | Path,
    model: HollowCoreLLM,
    optimizer: torch.optim.Optimizer | None = None,
    ema: EMATargetEncoder | None = None,
) -> None:
    payload: dict[str, Any] = {"model": model.state_dict(), "config": model.cfg}
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if ema is not None:
        payload["ema"] = ema.state_dict_for_checkpoint()
    torch.save(payload, Path(path))


def load_checkpoint(
    path: str | Path,
    model: HollowCoreLLM,
    optimizer: torch.optim.Optimizer | None = None,
    ema: EMATargetEncoder | None = None,
) -> None:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    if ema is not None and "ema" in payload:
        ema.load_state_dict_from_checkpoint(payload["ema"])


DATASET_BATCH_KEYS = (
    "input_ids",
    "labels",
    "action_mask",
    "tool_decision_labels",
    "tool_id_labels",
    "input_ids_b",
)
