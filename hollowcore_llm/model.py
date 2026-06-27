from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .attn_res import FullAttnResAccumulator
from .config import HollowCoreConfig
from .jepa import JEPAHead
from .layers import DeltaFlashMixer, RMSNorm, Top1MoE
from .tools import ToolPolicyHead


@dataclass
class HollowCoreOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None
    metrics: dict[str, float]
    tool_decision_logits: torch.Tensor
    tool_id_logits: torch.Tensor
    hidden: torch.Tensor | None = None


class HollowBlock(nn.Module):
    def __init__(self, cfg: HollowCoreConfig, layer_idx: int) -> None:
        super().__init__()
        self.norm_mixer = RMSNorm(cfg.hidden_size)
        self.mixer = DeltaFlashMixer(cfg, layer_idx)
        self.norm_moe = RMSNorm(cfg.hidden_size)
        self.moe = Top1MoE(cfg)

    def forward_mixer(
        self,
        x: torch.Tensor,
        gdn2_state=None,
        kv_halo=None,
        pos_offset: int = 0,
    ) -> tuple[torch.Tensor, Any, Any]:
        return self.mixer(self.norm_mixer(x), gdn2_state, kv_halo, pos_offset)

    def forward_moe(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.moe(self.norm_moe(x))


class HollowCoreLLM(nn.Module):
    def __init__(self, cfg: HollowCoreConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or HollowCoreConfig()
        self.cfg.validate()
        self.embed = nn.Embedding(self.cfg.vocab_size, self.cfg.hidden_size)
        self.layers = nn.ModuleList(HollowBlock(self.cfg, i) for i in range(self.cfg.num_layers))
        self.attn_res = FullAttnResAccumulator(self.cfg, num_steps=self.cfg.num_layers * 2)
        self.norm = RMSNorm(self.cfg.hidden_size)
        self.lm_head = nn.Linear(self.cfg.hidden_size, self.cfg.vocab_size, bias=False)
        self.jepa = JEPAHead(self.cfg)
        self.tool_head = ToolPolicyHead(self.cfg)
        if self.cfg.jepa_bridge:
            self.jepa_bridge_proj = nn.Linear(self.cfg.thought_dim, self.cfg.hidden_size, bias=False)
            self.bridge_gate = nn.Parameter(torch.ones(()))
        self.apply(self._init_weights)
        if self.cfg.jepa_bridge:
            nn.init.zeros_(self.jepa_bridge_proj.weight)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.initializer_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.initializer_std)

    def _run_layer_pair(
        self,
        layer: HollowBlock,
        h: torch.Tensor,
        values: list[torch.Tensor],
        step: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        if self.cfg.grad_checkpoint and self.training:

            def _mixer_out(x: torch.Tensor) -> torch.Tensor:
                v, _, _ = layer.forward_mixer(x)
                return v

            v_mixer = checkpoint(_mixer_out, h, use_reentrant=False)
        else:
            v_mixer, _, _ = layer.forward_mixer(h)

        values.append(v_mixer)
        h = self.attn_res.combine(step, values, h)
        step += 1

        if self.cfg.grad_checkpoint and self.training:
            v_moe, aux, z = checkpoint(layer.forward_moe, h, use_reentrant=False)
        else:
            v_moe, aux, z = layer.forward_moe(h)

        values.append(v_moe)
        h = self.attn_res.combine(step, values, h)
        step += 1
        return h, aux, z, step

    def _chunk_loss_metrics(
        self,
        hn: torch.Tensor,
        chunk_labels: torch.Tensor,
        aux_total: torch.Tensor,
        z_total: torch.Tensor,
        action_mask: torch.Tensor | None,
        tool_decision_labels: torch.Tensor | None,
        tool_id_labels: torch.Tensor | None,
        hidden_b: torch.Tensor | None,
        thought_b: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        cfg = self.cfg
        logits = self.lm_head(hn)
        ce = F.cross_entropy(
            logits[:, :-1].float().reshape(-1, cfg.vocab_size),
            chunk_labels[:, 1:].reshape(-1),
            ignore_index=-100,
        )
        cross_jepa, chunk_jepa, sigreg = self.jepa.losses(hn, hidden_b, thought_b=thought_b)
        tool_decision_loss, tool_id_loss = self.tool_head.loss(
            hn, action_mask, tool_decision_labels, tool_id_labels
        )
        total = (
            cfg.ce_weight * ce
            + cfg.cross_view_jepa_weight * cross_jepa
            + cfg.chunk_jepa_weight * chunk_jepa
            + cfg.sigreg_weight * sigreg
            + cfg.router_aux_weight * aux_total
            + cfg.router_z_weight * z_total
            + cfg.tool_decision_weight * tool_decision_loss
            + cfg.tool_id_weight * tool_id_loss
        )
        metrics = {
            "loss": float(total.detach().cpu()),
            "ce": float(ce.detach().cpu()),
            "cross_view_jepa": float(cross_jepa.detach().cpu()),
            "chunk_jepa": float(chunk_jepa.detach().cpu()),
            "sigreg": float(sigreg.detach().cpu()),
            "router_aux": float(aux_total.detach().cpu()),
            "router_z": float(z_total.detach().cpu()),
            "tool_decision": float(tool_decision_loss.detach().cpu()),
            "tool_id": float(tool_id_loss.detach().cpu()),
        }
        return total, metrics

    def _run_chunked_train(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        action_mask: torch.Tensor | None = None,
        tool_decision_labels: torch.Tensor | None = None,
        tool_id_labels: torch.Tensor | None = None,
        hidden_b: torch.Tensor | None = None,
        thought_b: torch.Tensor | None = None,
    ) -> dict[str, float]:
        cfg = self.cfg
        C = cfg.chunk_train_size
        gdn2_state: list[Any] = [None] * cfg.num_layers
        kv_halo: list[Any] = [None] * cfg.num_layers
        prev_thought = None
        metrics_acc: dict[str, float] = {}
        n_chunks = 0

        for start in range(0, input_ids.size(1), C):
            chunk = input_ids[:, start : start + C]
            chunk_labels = labels[:, start : start + C]
            h = self.embed(chunk)
            values: list[torch.Tensor] = []
            if cfg.jepa_bridge and prev_thought is not None:
                b = self.jepa_bridge_proj(prev_thought).unsqueeze(1).expand(h.size(0), h.size(1), -1)
                values = [self.bridge_gate * b]
            step = 0
            aux_total = h.new_zeros(())
            z_total = h.new_zeros(())
            for li, layer in enumerate(self.layers):
                v_mixer, gs, kh = layer.forward_mixer(h, gdn2_state[li], kv_halo[li], pos_offset=start)
                gdn2_state[li] = gs.detach()
                kv_halo[li] = kh
                values.append(v_mixer)
                h = self.attn_res.combine(step, values, h)
                step += 1
                v_moe, aux, z = layer.forward_moe(h)
                aux_total = aux_total + aux
                z_total = z_total + z
                values.append(v_moe)
                h = self.attn_res.combine(step, values, h)
                step += 1
            hn = self.norm(h)
            am = action_mask[:, start : start + C] if action_mask is not None else None
            td = tool_decision_labels[:, start : start + C] if tool_decision_labels is not None else None
            ti = tool_id_labels[:, start : start + C] if tool_id_labels is not None else None
            chunk_loss, chunk_metrics = self._chunk_loss_metrics(
                hn, chunk_labels, aux_total, z_total, am, td, ti, hidden_b, thought_b
            )
            chunk_loss.backward()
            for key, val in chunk_metrics.items():
                metrics_acc[key] = metrics_acc.get(key, 0.0) + val
            n_chunks += 1
            prev_thought = self.jepa.to_thought(hn).mean(dim=1).detach()

        if n_chunks == 0:
            return {"loss": 0.0}
        return {k: v / n_chunks for k, v in metrics_acc.items()}

    def _run_backbone(
        self,
        input_ids: torch.Tensor,
        compute_loss: bool,
        labels: torch.Tensor | None,
        action_mask: torch.Tensor | None,
        tool_decision_labels: torch.Tensor | None,
        tool_id_labels: torch.Tensor | None,
        hidden_b: torch.Tensor | None,
        thought_b: torch.Tensor | None,
    ) -> HollowCoreOutput:
        if input_ids.size(1) > self.cfg.max_seq_len:
            raise ValueError(f"seq_len {input_ids.size(1)} exceeds {self.cfg.max_seq_len}")

        h = self.embed(input_ids)
        values: list[torch.Tensor] = []
        aux_total = h.new_zeros(())
        z_total = h.new_zeros(())
        step = 0

        for layer in self.layers:
            h, aux, z, step = self._run_layer_pair(layer, h, values, step)
            aux_total = aux_total + aux
            z_total = z_total + z

        h = self.norm(h)
        logits = self.lm_head(h)
        tool_decision_logits, tool_id_logits = self.tool_head(h)

        metrics: dict[str, float] = {}
        total = None

        if compute_loss and labels is not None:
            total, metrics = self._chunk_loss_metrics(
                h,
                labels,
                aux_total,
                z_total,
                action_mask,
                tool_decision_labels,
                tool_id_labels,
                hidden_b,
                thought_b,
            )

        return HollowCoreOutput(logits, total, metrics, tool_decision_logits, tool_id_logits, h)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        action_mask: torch.Tensor | None = None,
        tool_decision_labels: torch.Tensor | None = None,
        tool_id_labels: torch.Tensor | None = None,
        hidden_b: torch.Tensor | None = None,
        thought_b: torch.Tensor | None = None,
    ) -> HollowCoreOutput:
        return self._run_backbone(
            input_ids,
            labels is not None,
            labels,
            action_mask,
            tool_decision_labels,
            tool_id_labels,
            hidden_b,
            thought_b,
        )

    @torch.no_grad()
    def forward_view(self, input_ids: torch.Tensor) -> torch.Tensor:
        was_training = self.training
        self.eval()
        out = self._run_backbone(input_ids, False, None, None, None, None, None, None)
        if was_training:
            self.train()
        assert out.hidden is not None
        return out.hidden

    def parameter_report(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        expert = sum(p.numel() for p in self.layers[0].moe.experts[0].parameters()) * self.cfg.num_layers
        return {"total": total, "one_expert_across_layers": expert}
