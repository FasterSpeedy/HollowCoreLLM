**HollowCoreLLM is an experimental research prototype, not a trained foundation model.
The current goal is to validate whether the architecture can train stably at small scale before attempting larger long-context runs.**

# HollowCoreLLM

[![Weights & Biases — live run](https://img.shields.io/badge/W%26B-live%20training%20run-FFBE00?logo=weightsandbiases&logoColor=black)](https://wandb.ai/HollowCore-/HollowCoreLLM/runs/7xiper3h)

> ###  Live training run — real numbers, not a claim
> A **100M-param / ~40M-active**, byte-level, **500K-context** model is training **right now on a single 4 GB RTX 3050 laptop**, resumed from checkpoint step 41.
> **Live dashboard → [wandb.ai/HollowCore-/HollowCoreLLM](https://wandb.ai/HollowCore-/HollowCoreLLM/runs/7xiper3h)** — loss, throughput (tokens/s), VRAM and energy (tokens/kWh) stream straight off the machine.
> Came here from Reddit? Open the dashboard. The loss curve, step log and hardware metrics are all there.

I don't have a PhD or a math background. I'm a self-taught builder who learns by connecting dots. I built HollowCoreLLM because I saw a way to combine byte-level tokenization, DeltaNet-2, and JEPA heads that standard academia isn't looking at. I'm using AI to write the heavy math, but the architecture is mine.

Experimental, from-scratch **byte-level language model** that explores two ideas in a single training loop:

1. **JEPA latent prediction** - the model predicts latent "thought" vectors of future context (cross-view and chunk-level), not just next tokens.
2. **Tool-use policy** - an explicit head learns *when* and *which* tool to call, co-trained with language modeling.

The global path is **linear attention (DeltaNet-2)**, so context can be trained in **chunks up to ~1M tokens** on a single GPU. ( current run: 500K / 2048-chunk )

> **Status: experimental / pre-training.** This repository is a working training stack, not a finished model.
> No *downstream benchmark* scores yet — but training is **live and fully instrumented**; see the dashboard above for real loss / throughput / energy curves.

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
| Chunk size | 4096 |
| Training mode | chunked long-context training |
| JEPA mode | EMA cross-view JEPA + chunk JEPA + SIGReg |
| Dataset mix | English, code, philosophy, tool-use, instruction, code-docs, HollowCore self-knowledge |



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

Chunk 4096 fits in ~2.3 GB peak VRAM on the 4 GB RTX 3050 via gradient checkpointing of the per-chunk mixer/MoE. (  2048 ≈ 2.44 GB ) 

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
