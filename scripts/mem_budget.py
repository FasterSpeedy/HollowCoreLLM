"""Memorijski bill prije 1M runa. ponytail: stdlib + argparse only."""
from __future__ import annotations

import argparse

BYTES_BF16 = 2
BYTES_FP32 = 4
H200_GB = 141


def gb(n_bytes: float) -> float:
    return n_bytes / (1024**3)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--params-billions", type=float, required=True)
    p.add_argument("--seq", type=int, default=1_048_576)
    p.add_argument("--chunk", type=int, default=16_384)
    p.add_argument("--hidden", type=int, default=4096)
    p.add_argument("--batch", type=int, default=1)
    args = p.parse_args()

    n = args.params_billions * 1e9
    weights = gb(n * BYTES_BF16)
    adam_fp32 = gb(n * 2 * BYTES_FP32)
    adam_8bit = gb(n * 2 * 1)
    ema_gpu = gb(n * BYTES_BF16)

    L_max = 64
    attn_stack_tb = args.batch * args.seq * L_max * args.hidden * BYTES_FP32 / (1024**4)

    print(f"params={args.params_billions}B  seq={args.seq}  chunk={args.chunk}")
    print(f"weights_bf16_GB={weights:.1f}")
    print(f"adam_fp32_GB={adam_fp32:.1f}  adam_8bit_est_GB={adam_8bit:.1f}")
    print(f"ema_gpu_deepcopy_GB={ema_gpu:.1f}")
    print(f"attn_res_stack_TB_no_chunk={attn_stack_tb:.2f}  (reference only)")
    print(f"H200_GPU_GB={H200_GB}  fixed_overhead={weights + adam_8bit:.1f}GB (weights+8bit, no EMA)")
    ok = weights + adam_8bit < H200_GB * 0.85
    print(f"fits_h200_fixed={'YES' if ok else 'NO — use 8bit+no_ema or B200'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
