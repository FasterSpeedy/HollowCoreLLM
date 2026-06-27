from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parent))
from manifest import CheckpointManifest, assert_resume_compatible


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-id", default=os.environ.get("HF_CHECKPOINT_REPO"))
    p.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    p.add_argument(
        "--profile",
        default=os.environ.get("MODAL_PROFILE") or os.environ.get("TRAIN_PROFILE", "modal-a"),
    )
    p.add_argument("--out-dir", type=Path, default=Path("checkpoints"))
    p.add_argument("--allow-branch-takeover", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.repo_id:
        raise SystemExit("Set HF_CHECKPOINT_REPO or pass --repo-id")
    if not args.token:
        raise SystemExit("Set HF_TOKEN or pass --token")

    manifest_path = hf_hub_download(
        repo_id=args.repo_id,
        filename="manifests/latest.json",
        repo_type="model",
        token=args.token,
    )
    manifest = CheckpointManifest.load(manifest_path)
    assert_resume_compatible(manifest, args.profile, args.allow_branch_takeover)

    ckpt_remote = manifest.checkpoint_path or f"checkpoints/step-{manifest.step:06d}.pt"
    ckpt_path = hf_hub_download(
        repo_id=args.repo_id,
        filename=ckpt_remote,
        repo_type="model",
        token=args.token,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    local_ckpt = args.out_dir / Path(ckpt_remote).name
    local_manifest = args.out_dir / "latest_manifest.json"
    local_ckpt.write_bytes(Path(ckpt_path).read_bytes())
    local_manifest.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")
    print(f"resume_checkpoint={local_ckpt}")
    print(f"resume_step={manifest.step}")
    print(f"lineage_id={manifest.lineage_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
