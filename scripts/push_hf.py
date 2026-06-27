from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from manifest import CheckpointManifest


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--repo-id", default=os.environ.get("HF_CHECKPOINT_REPO"))
    p.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    return p.parse_args()


def _remote_path(manifest: CheckpointManifest) -> str:
    if manifest.checkpoint_path and manifest.checkpoint_path.startswith("checkpoints/"):
        return manifest.checkpoint_path
    return f"checkpoints/step-{manifest.step:06d}.pt"


def main() -> int:
    args = parse_args()
    if not args.repo_id:
        raise SystemExit("Set HF_CHECKPOINT_REPO or pass --repo-id")
    if not args.token:
        raise SystemExit("Set HF_TOKEN or pass --token")

    manifest = CheckpointManifest.load(args.manifest)
    remote = _remote_path(manifest)
    manifest.checkpoint_path = remote
    manifest.save(args.manifest)

    api = HfApi(token=args.token)
    api.create_repo(args.repo_id, repo_type="model", exist_ok=True, private=True)
    api.upload_file(
        path_or_fileobj=str(args.checkpoint),
        path_in_repo=remote,
        repo_id=args.repo_id,
        repo_type="model",
        commit_message=f"step {manifest.step} profile={manifest.profile}",
    )
    api.upload_file(
        path_or_fileobj=json.dumps(manifest.to_dict(), indent=2).encode("utf-8"),
        path_in_repo=f"manifests/step-{manifest.step:06d}.json",
        repo_id=args.repo_id,
        repo_type="model",
    )
    api.upload_file(
        path_or_fileobj=json.dumps(manifest.to_dict(), indent=2).encode("utf-8"),
        path_in_repo="manifests/latest.json",
        repo_id=args.repo_id,
        repo_type="model",
    )
    print(f"uploaded={remote}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
