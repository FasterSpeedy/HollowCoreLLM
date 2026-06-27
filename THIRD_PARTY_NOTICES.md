# Third-party notices

HollowCoreLLM depends on external libraries. Each retains its own license.
When you distribute code that bundles or builds on this project, retain the
notices below alongside the MIT license in `LICENSE`.

## PyTorch

- Package: `torch`
- License: BSD-style (see PyTorch LICENSE)
- https://github.com/pytorch/pytorch

## flash-linear-attention (GDN-2 / DeltaNet-2 kernels)

- Package: `flash-linear-attention`
- Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
- License: MIT License
- https://github.com/fla-org/flash-linear-attention

HollowCoreLLM calls `fla.ops.gdn2.chunk_gdn2` for the global recurrent path.

## flash-attn (local FlashAttention-2 path)

- Package: `flash-attn`
- Copyright (c) Dao-AILab contributors
- License: BSD 3-Clause "New" or "Revised" License
- https://github.com/Dao-AILab/flash-attention

## Hugging Face libraries

- Packages: `datasets`, `huggingface_hub`
- License: Apache License 2.0
- https://github.com/huggingface/datasets
- https://github.com/huggingface/huggingface_hub

## bitsandbytes (optional 8-bit optimizer)

- Package: `bitsandbytes`
- License: MIT License
- https://github.com/bitsandbytes-foundation/bitsandbytes
