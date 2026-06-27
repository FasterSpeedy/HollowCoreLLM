from dataclasses import dataclass, field


@dataclass(frozen=True)
class HollowCoreConfig:
    vocab_size: int = 320
    byte_vocab_size: int = 256
    pad_token_id: int = 256
    bos_token_id: int = 257
    eos_token_id: int = 258
    action_token_id: int = 259
    tool_call_token_id: int = 260
    tool_result_token_id: int = 261
    tool_end_token_id: int = 262
    thought_token_id: int = 263

    max_seq_len: int = 1_048_576
    hidden_size: int = 4096
    num_layers: int = 32
    num_heads: int = 32
    head_dim: int = 128
    local_window: int = 2048
    rope_base: float = 10_000.0
    rope_scale: float = 512.0

    num_experts: int = 3
    moe_top_k: int = 1
    expert_intermediate_size: int = 4096
    router_aux_weight: float = 0.01
    router_z_weight: float = 0.001

    thought_dim: int = 2048
    jepa_chunk_size: int = 2048
    ce_weight: float = 0.30
    token_jepa_weight: float = 0.35
    chunk_jepa_weight: float = 0.35
    cross_view_jepa_weight: float = 0.25
    sigreg_weight: float = 0.01
    ema_decay: float = 0.996
    jepa_view_types: tuple[str, ...] = ("text_code", "paraphrase", "context_tool")

    num_tools: int = 64
    tool_decision_weight: float = 0.05
    tool_id_weight: float = 0.05

    dropout: float = 0.0
    initializer_std: float = 0.02
    grad_checkpoint: bool = False

    chunk_train_size: int = 16384
    jepa_bridge: bool = True

    def validate(self) -> None:
        assert self.hidden_size == self.num_heads * self.head_dim
        assert self.moe_top_k == 1
        assert self.vocab_size > self.thought_token_id
        assert self.chunk_train_size == 0 or self.chunk_train_size >= self.local_window


@dataclass(frozen=True)
class TrainCurriculumConfig:
    stable_steps: int = 200
    mix_start_step: int = 200
    categories: tuple[str, ...] = ("stable", "web", "code", "tool_call")
    initial_mix: dict[str, float] = field(
        default_factory=lambda: {"stable": 1.0, "web": 0.0, "code": 0.0, "tool_call": 0.0}
    )
    mixed_proportions: dict[str, float] = field(
        default_factory=lambda: {
            "stable": 0.15,
            "web": 0.15,
            "code": 0.15,
            "tool_call": 0.55,
        }
    )
    replay_floor: float = 0.10
