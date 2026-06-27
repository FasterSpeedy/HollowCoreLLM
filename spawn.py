import argparse
from pathlib import Path

import torch

from hollowcore_llm import HollowCoreConfig, HollowCoreLLM
from hollowcore_llm.deps import assert_runtime_deps
from hollowcore_llm.trainer import build_optimizer, save_checkpoint


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Init checkpoint on Modal GPU")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bf16", choices=("bf16", "fp16", "fp32"))
    p.add_argument("--save-init", default=None)
    p.add_argument("--optimizer", action="store_true")
    return p.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required — run spawn via Modal (modal run modal_app.py --spawn-only)")

    assert_runtime_deps()

    cfg = HollowCoreConfig()
    model = HollowCoreLLM(cfg).to(device=args.device, dtype=dtype_from_name(args.dtype))
    report = model.parameter_report()
    print(f"total_params={report['total']:,}")
    print(f"one_expert_across_layers={report['one_expert_across_layers']:,}")
    print(f"max_seq_len={cfg.max_seq_len:,}")
    print(f"vocab_size={cfg.vocab_size}")

    optimizer = build_optimizer(model) if args.optimizer else None

    if args.save_init:
        save_checkpoint(Path(args.save_init), model, optimizer)
        print(f"saved={args.save_init}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
