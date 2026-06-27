"""Modal entry — spawn, train, HF checkpoint sync. Sve na Modal GPU."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import modal

APP_NAME = "hollowcore-train"
ROOT = Path(__file__).resolve().parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2.0",
        "datasets>=2.19.0",
        "huggingface_hub>=0.23.0",
        "bitsandbytes",
    )
    .pip_install("flash-attn", extra_options="--no-build-isolation")
    .pip_install("flash-linear-attention[cuda]")
    .add_local_dir(str(ROOT / "hollowcore_llm"), remote_path="/root/hollowcore_llm")
    .add_local_dir(str(ROOT / "data"), remote_path="/root/data")
    .add_local_dir(str(ROOT / "scripts"), remote_path="/root/scripts")
    .add_local_file(str(ROOT / "train_modal.py"), remote_path="/root/train_modal.py")
    .add_local_file(str(ROOT / "spawn.py"), remote_path="/root/spawn.py")
    .add_local_file(str(ROOT / "scripts" / "check_state_grad.py"), remote_path="/root/scripts/check_state_grad.py")
)

app = modal.App(APP_NAME)
vol = modal.Volume.from_name("hollowcore-checkpoints", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface")


def _run(cmd: list[str], env: dict[str, str] | None = None, check: bool = True) -> None:
    merged = os.environ.copy()
    if env:
        merged.update(env)
    subprocess.run(cmd, check=check, env=merged)


@app.function(
    image=image,
    gpu="H200",
    timeout=60 * 30,
    secrets=[hf_secret],
    volumes={"/checkpoints": vol},
)
def spawn_init_on_modal(profile: str = "modal-a", push_hf: bool = True) -> str:
    """Init checkpoint na GPU + opcionalni push na HF Hub."""
    os.chdir("/root")
    sys.path.insert(0, "/root")
    init_path = "/checkpoints/step-000000.pt"
    _run(
        [
            sys.executable,
            "/root/spawn.py",
            "--device",
            "cuda",
            "--dtype",
            "bf16",
            "--save-init",
            init_path,
        ]
    )
    if not push_hf or not os.environ.get("HF_CHECKPOINT_REPO"):
        vol.commit()
        return init_path

    sys.path.insert(0, "/root/scripts")
    from manifest import CheckpointManifest, new_lineage_id
    manifest = CheckpointManifest(
        step=0,
        seq_len=512,
        loss=0.0,
        seed=42,
        profile=profile,
        dataset_mix={"stable": 1.0},
        view="text_code",
        lineage_id=new_lineage_id(),
        branch_owner=profile,
        checkpoint_path="checkpoints/step-000000.pt",
    )
    manifest_path = "/checkpoints/step-000000.manifest.json"
    manifest.save(manifest_path)
    manifest.save("/checkpoints/latest_manifest.json")
    _run(
        [
            sys.executable,
            "/root/scripts/push_hf.py",
            init_path,
            "--manifest",
            manifest_path,
        ]
    )
    vol.commit()
    return init_path


@app.function(
    image=image,
    gpu="H200",
    memory=131072,
    timeout=60 * 60 * 24,
    secrets=[hf_secret],
    volumes={"/checkpoints": vol},
)
def train_on_modal(
    max_steps: int = 100,
    seq_len: int = 1_048_576,
    checkpoint_every: int = 100,
    resume: str | None = None,
    profile: str = "modal-a",
    seed: int = 42,
    push_hf: bool = True,
    chunk_train_size: int = 16384,
    jepa_bridge: bool = True,
    no_ema: bool = True,
    adam_8bit: bool = True,
) -> str:
    os.chdir("/root")
    sys.path.insert(0, "/root")

    if resume == "auto" and os.environ.get("HF_CHECKPOINT_REPO"):
        _run(
            [
                sys.executable,
                "/root/scripts/resume_hf.py",
                "--profile",
                profile,
                "--out-dir",
                "/checkpoints",
            ],
            check=False,
        )
        resume = "auto"

    cmd = [
        sys.executable,
        "/root/train_modal.py",
        "--device",
        "cuda",
        "--max-steps",
        str(max_steps),
        "--seq-len",
        str(seq_len),
        "--checkpoint-every",
        str(checkpoint_every),
        "--seed",
        str(seed),
        "--profile",
        profile,
        "--ckpt-dir",
        "/checkpoints",
    ]
    if resume:
        cmd.extend(["--resume", resume])
    if push_hf:
        cmd.append("--push-hf")
    cmd.extend(["--chunk-train-size", str(chunk_train_size)])
    if no_ema:
        cmd.append("--no-ema")
    if not adam_8bit:
        cmd.append("--no-adam-8bit")
    if not jepa_bridge:
        cmd.append("--no-jepa-bridge")

    _run(cmd)
    vol.commit()
    return "done"


@app.function(
    image=image,
    gpu="H200",
    timeout=60 * 10,
)
def check_state_grad() -> str:
    """FORK: provjeri tece li grad kroz chunk_gdn2 initial_state."""
    os.chdir("/root")
    sys.path.insert(0, "/root")
    import importlib.util
    spec = importlib.util.spec_from_file_location("check_state_grad", "/root/scripts/check_state_grad.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod._run_check()
    return str(mod.RESULT)


@app.local_entrypoint()
def main(
    max_steps: int = 100,
    seq_len: int = 1_048_576,
    resume: str | None = None,
    profile: str = "modal-a",
    spawn_only: bool = False,
    chunk_train_size: int = 16384,
    jepa_bridge: bool = True,
    no_ema: bool = True,
    adam_8bit: bool = True,
):
    if spawn_only:
        path = spawn_init_on_modal.remote(profile=profile)
        print(f"spawn_init={path}")
        return
    train_on_modal.remote(
        max_steps=max_steps,
        seq_len=seq_len,
        resume=resume,
        profile=profile,
        chunk_train_size=chunk_train_size,
        jepa_bridge=jepa_bridge,
        no_ema=no_ema,
        adam_8bit=adam_8bit,
    )
