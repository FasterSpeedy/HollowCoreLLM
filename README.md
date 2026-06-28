# HollowCoreLLM

**Experimental research prototype. Not a trained foundation model.**

HollowCoreLLM is a from-scratch byte-level language model architecture combining local FlashAttention, global Gated DeltaNet-2 recurrent memory, top-1 MoE, Attention Residual mixing, EMA-JEPA latent prediction, SIGReg regularization, and explicit tool-use heads.

The current goal is not to claim benchmark performance. The goal is to test whether the architecture can execute and train stably at long context under constrained hardware, then document the results honestly.

## Current Local Evidence Snapshot

This run is a local execution proof on laptop hardware. It tests the real training stack with EMA cross-view JEPA enabled.

| Item | Value |
|---|---|
| Run date | 2026-06-28 |
| OS | WSL2 Linux 6.18.33.2 microsoft standard, glibc 2.35 |
| Python | CPython 3.10.12 |
| Hardware | Acer Nitro 5 laptop |
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU, 4GB VRAM |
| CPU | 6 physical cores, 12 logical threads |
| Model size | 102,043,891 total parameters |
| Estimated active parameters | ~40,702,195 |
| Context length | 500,000 tokens |
| Chunk size | 2,048 |
| Training mode | chunked long-context training |
| JEPA mode | EMA cross-view JEPA + chunk JEPA + SIGReg |
| Dataset mix | English, code, philosophy, tool-use, instruction, code-docs, HollowCore self-knowledge |
| W&B run path | HollowCore-/HollowCoreLLM/7i0d4924 |
| Metrics dashboard | https://wandb.ai/HollowCore-/HollowCoreLLM/runs/7i0d4924 |

Important W&B note: the W&B run metadata shows `wandb_bridge.py` as the command because metrics are mirrored into W&B by a separate bridge process. The actual training process is `spawn_train_100m_500k_7mix_self_ema.py` running locally in WSL.

Latest watcher snapshot:

```text
step=10
elapsed=0.2691 h
speed=94.29 steps/h
tokens/sec=13095.25
tokens_seen_est=5,000,000
source=english
loss=1.3510
ce=3.8299
cross_view_jepa=0.2125
chunk_jepa=0.1210
sigreg=0.6683
router_aux=9.2967
router_z=6.8874
tool_decision=0.0000
tool_id=0.0000
peak_vram_step=2.394 GB
gpu_memory=2909/4096 MB
gpu_util=44%
gpu_mem_util=37%
temp=82 C
power=47.97 W
tok/J=272.988
Wh/Mtok=1.017544
alerts=none
```

## What This Shows

The current code can execute a 100M-class, 500K-context, EMA-JEPA training run on a 4GB laptop GPU without immediate OOM.

The run records the signals needed to evaluate execution and early stability: CE loss, cross-view JEPA, chunk JEPA, SIGReg, router losses, tool losses, VRAM, GPU utilization, power, throughput, energy per token, samples, checkpoints, and alerts.

## What This Does Not Show Yet

This is not a trained chat model.
This is not a benchmark result.
This is not a final scaling claim.
This does not yet prove model quality or superiority over transformer baselines.

The current evidence supports execution feasibility and early training stability only. Scaling curves, downstream evaluations, ablations, and baseline comparisons still need to be run.

## Architecture

- **Byte-level vocabulary**: 256 raw UTF-8 bytes plus special tokens.
- **Hybrid mixer**:
  - Local FlashAttention path for short-range precision.
  - Gated DeltaNet-2 recurrent path for long-range memory.
- **Top-1 MoE** feed-forward layers.
- **Attention Residuals**: learned weighted mixing over prior sublayer outputs.
- **JEPA heads**: cross-view and chunk-level latent prediction against an EMA target encoder.
- **SIGReg**: variance/covariance regularization to reduce collapse risk.
- **Tool-policy head**: learns no-tool/tool-call decisions and tool IDs.
- **JEPA bridge**: previous chunk thought enters the next chunk as a slow learned memory signal.

## Long-Context Training

The sequence is processed in chunks. GDN-2 recurrent state and local-attention KV halo are carried across chunk boundaries, keeping activation memory bounded while allowing very long input contexts.

The current local proof run uses 500K context with 2K chunks. Larger chunk sizes and larger models are separate experiments.

## Data

The current experimental mix uses prepared byte-level token shards from:

- English text
- Code
- Philosophy text / QA
- Tool-use traces
- Instruction data
- Code documentation / explanation data
- HollowCore self-knowledge data

The data pipeline is experimental and may change.

## Limitations

- Research prototype, not production training software.
- No trained model weights are published yet.
- Byte-level tokenization increases sequence length.
- Long-context gradients are truncated at chunk boundaries by default.
- Current evidence is from early local runs, not controlled benchmark comparisons.
- APIs and configs may change quickly.

Third-party library licenses: see `THIRD_PARTY_NOTICES.md`.
