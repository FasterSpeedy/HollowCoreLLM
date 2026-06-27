from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from data.byte_tokenizer import ByteTokenizer
from data.curriculum_sampler import CurriculumSampler
from data.registry import DatasetRegistry
from data.streams import MixedBatchIterator
from hollowcore_llm import HollowCoreConfig, HollowCoreLLM
from hollowcore_llm.deps import assert_runtime_deps
from hollowcore_llm.ema import EMATargetEncoder
from hollowcore_llm.trainer import build_optimizer, load_checkpoint, move_batch, save_checkpoint, train_step
from manifest import CheckpointManifest, new_lineage_id


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HollowCore train loop — Modal GPU only")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bf16", choices=("bf16", "fp16", "fp32"))
    p.add_argument("--seq-len", type=int, default=1_048_576)
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--checkpoint-every", type=int, default=100)
    p.add_argument("--resume", default=None, help="path to .pt or 'auto' from checkpoints/latest_manifest.json")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--profile",
        default=os.environ.get("MODAL_PROFILE") or os.environ.get("TRAIN_PROFILE", "modal-a"),
    )
    p.add_argument("--ckpt-dir", type=Path, default=Path("checkpoints"))
    p.add_argument("--push-hf", action="store_true")
    p.add_argument("--hf-repo", default=os.environ.get("HF_CHECKPOINT_REPO"))
    p.add_argument("--chunk-train-size", type=int, default=16384)
    p.add_argument("--no-jepa-bridge", action="store_true")
    p.add_argument("--no-ema", action="store_true")
    p.add_argument("--no-adam-8bit", action="store_true")
    return p.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def resolve_resume(path: str | None, ckpt_dir: Path) -> tuple[Path | None, CheckpointManifest | None]:
    if path is None:
        return None, None
    if path == "auto":
        manifest_path = ckpt_dir / "latest_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError("No latest_manifest.json for --resume auto")
        manifest = CheckpointManifest.load(manifest_path)
        ckpt_name = Path(manifest.checkpoint_path).name if manifest.checkpoint_path else f"step-{manifest.step:06d}.pt"
        ckpt = ckpt_dir / ckpt_name
        return ckpt, manifest
    return Path(path), None


def push_checkpoint(ckpt: Path, manifest: CheckpointManifest, hf_repo: str) -> None:
    manifest_path = ckpt.with_suffix(".manifest.json")
    manifest.save(manifest_path)
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "push_hf.py"), str(ckpt), "--manifest", str(manifest_path), "--repo-id", hf_repo],
        check=True,
        env=os.environ.copy(),
    )


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required — run via Modal (modal run modal_app.py)")

    assert_runtime_deps()

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    dtype = dtype_from_name(args.dtype)

    cfg = HollowCoreConfig(
        grad_checkpoint=True,
        chunk_train_size=args.chunk_train_size,
        jepa_bridge=not args.no_jepa_bridge,
    )
    model = HollowCoreLLM(cfg).to(device=device, dtype=dtype)
    optimizer = build_optimizer(model, adam_8bit=not args.no_adam_8bit)
    ema = None if args.no_ema else EMATargetEncoder(model, decay=cfg.ema_decay)

    start_step = 0
    lineage_id = new_lineage_id()
    parent_lineage = ""
    resume_manifest: CheckpointManifest | None = None

    ckpt_path, resume_manifest = resolve_resume(args.resume, args.ckpt_dir)
    if ckpt_path and ckpt_path.exists():
        load_checkpoint(ckpt_path, model, optimizer, ema=ema)
        if resume_manifest:
            start_step = resume_manifest.step
            lineage_id = resume_manifest.lineage_id or lineage_id
            parent_lineage = resume_manifest.parent_lineage_id

    registry = DatasetRegistry.load()
    sampler = CurriculumSampler(registry.curriculum, rng)
    tok = ByteTokenizer()
    iterator = MixedBatchIterator(tok, args.seq_len, sampler, registry=registry, seed=args.seed)

    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    last_metrics: dict[str, float] = {}

    for step in range(start_step, args.max_steps):
        iterator.step = step
        batch, view, mix = iterator.next_batch()
        batch = move_batch(batch, device)
        last_metrics = train_step(model, batch, optimizer, ema=ema)

        if (step + 1) % args.checkpoint_every == 0 or step + 1 == args.max_steps:
            ckpt_name = f"step-{step + 1:06d}.pt"
            ckpt_file = args.ckpt_dir / ckpt_name
            save_checkpoint(ckpt_file, model, optimizer, ema=ema)
            manifest = CheckpointManifest(
                step=step + 1,
                seq_len=args.seq_len,
                loss=last_metrics.get("loss", 0.0),
                seed=args.seed,
                profile=args.profile,
                dataset_mix=mix,
                view=view,
                lineage_id=lineage_id,
                parent_lineage_id=parent_lineage,
                branch_owner=args.profile,
                checkpoint_path=f"checkpoints/{ckpt_name}",
            )
            manifest.save(args.ckpt_dir / f"step-{step + 1:06d}.manifest.json")
            manifest.save(args.ckpt_dir / "latest_manifest.json")
            print(f"step={step + 1} loss={manifest.loss:.4f} mix={json.dumps(mix)}")
            if args.push_hf and args.hf_repo:
                push_checkpoint(ckpt_file, manifest, args.hf_repo)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
