**HollowCoreLLM is an experimental research prototype, not a trained foundation model.
The current goal is to validate whether the architecture can train stably at small scale before attempting larger long-context runs.**

# HollowCoreLLM

I don't have a PhD or a math background. I'm a self-taught builder who learns by connecting dots. I built HollowCoreLLM because I saw a way to combine byte-level tokenization, DeltaNet-2, and JEPA heads that standard academia isn't looking at. I'm using AI to write the heavy math, but the architecture is mine. I'm looking for MLSys veterans to tell me where this will break.

Experimental, from-scratch **byte-level language model** that explores two ideas in a single training loop:

1. **JEPA latent prediction** - the model predicts latent "thought" vectors of future context (cross-view and chunk-level), not just next tokens.
2. **Tool-use policy** - an explicit head learns *when* and *which* tool to call, co-trained with language modeling.

The global path is **linear attention (DeltaNet-2)**, so context can be trained in **chunks up to ~1M tokens** on a single GPU.

> **Status: experimental / pre-training.** This repository is a working training stack, not a finished model.
> **There are no benchmark results, scaling curves, or performance numbers here** - the current goal is a stack that either trains stably or fails in documented ways.

## Architecture

- **Byte-level vocabulary** - 256 raw UTF-8 bytes + 8 special tokens (BOS / EOS / PAD / tool markers / thought).
- **Hybrid mixer (per block):**
  - *Local path* - FlashAttention-2 with a fixed causal window (short-range precision).
  - *Global path* - Gated DeltaNet-2 (`fla.ops.gdn2`) recurrent linear attention (long-horizon state, linear memory).
- **Top-1 Mixture-of-Experts** feed-forward layers.
- **Full Attention Residuals** - each block learns a weighted mix over all prior sublayer outputs instead of a fixed `x += f(x)`.
- **JEPA heads** - cross-view and chunk-level latent prediction against an EMA target encoder, with a variance/covariance regularizer (SIGReg) to limit representation collapse.
- **Tool-policy head** - masked `no_tool` / `tool_call` decision + tool-id classification at marked positions.

## Long-context training

- **Chunked training** - the sequence is processed in chunks (default 16K). The GDN-2 recurrent state and a local-attention KV halo are carried across chunk boundaries, so activation memory stays bounded at long context.
- **JEPA bridge** - the pooled "thought" of the previous chunk enters the next chunk as an extra Attention-Residual value: a learned, slow cross-chunk memory alongside the fast recurrent state.
- **Token-packing** - text documents are packed end-to-end (EOS-separated) to fill each sequence with real tokens instead of padding.
- Loss and backward run **per chunk** (truncated BPTT across chunk boundaries by default).

## Data

Sources are declared in [`data/datasets.json`](data/datasets.json) and **streamed** from the Hugging Face Hub (no full download). Two source kinds:

- `hf_text` - any text column, trained as next-token LM and token-packed. Optional `config` for datasets that need a subset name.
- `glaive` - function-calling traces that produce the tool-policy labels.

A curriculum mixes the categories over training (`data/registry.py`, `data/curriculum_sampler.py`).


## Limitations

- Research code - **no trained model or evaluation numbers are published.**
- Byte-level tokenization makes sequences long.
- Long-context gradients are truncated at chunk boundaries by default.
- APIs and configuration may change without notice.

Third-party library licenses: see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
