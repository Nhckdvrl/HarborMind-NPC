#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Check local or Docker training environments.")
    parser.add_argument(
        "--profile",
        choices=["base", "sft", "serve", "rl"],
        default="base",
        help="Environment profile to validate.",
    )
    args = parser.parse_args()

    ok = True
    print(f"profile: {args.profile}")
    print(f"python: {sys.executable}")
    print(f"python_version: {sys.version.split()[0]}")
    ok &= require_python(args.profile)

    if args.profile in {"sft", "serve", "rl"}:
        ok &= check_package("game_npc_llm")
    if args.profile == "sft":
        ok &= check_package("llamafactory")
        ok &= check_package("datasets")
        ok &= check_path(ROOT / "data" / "dataset_info.json")
    if args.profile == "serve":
        ok &= check_package("sglang")
        ok &= check_package("yaml")
        ok &= check_command("ninja")
        ok &= check_path(ROOT / "configs" / "sglang" / "models.yml")
    if args.profile == "rl":
        ok &= check_package("verl")
        ok &= check_package("vllm")
        ok &= check_package("ray")
        ok &= check_package("datasets")
        ok &= check_package("pyarrow")
        ok &= check_path(ROOT / "data" / "rl" / "verl" / "grpo_smoke_100.parquet")
        ok &= check_path(ROOT / "outputs" / "sft" / "qwen3_5_27b_npc_merged")
        ok &= check_torch_cuda()

    raise SystemExit(0 if ok else 1)


def require_python(profile: str) -> bool:
    version = sys.version_info
    if profile == "rl" and version.major == 3 and version.minor == 12:
        print("python_compat: OK (3.12 for verl)")
        return True
    if profile != "rl" and version.major == 3 and version.minor == 11:
        print("python_compat: OK (3.11)")
        return True
    if profile == "rl":
        print("python_compat: FAIL (recommend Python 3.12 for verl)")
    else:
        print("python_compat: FAIL (recommend Python 3.11 for this repo)")
    return False


def check_package(name: str) -> bool:
    found = importlib.util.find_spec(name) is not None
    print(f"package:{name}: {'OK' if found else 'MISSING'}")
    return found


def check_command(name: str) -> bool:
    found = shutil.which(name) is not None
    print(f"command:{name}: {'OK' if found else 'MISSING'}")
    return found


def check_path(path: Path) -> bool:
    found = path.exists()
    print(f"path:{path}: {'OK' if found else 'MISSING'}")
    return found


def check_env_var(name: str, prefer: str | None = None, optional: bool = False) -> bool:
    value = os.getenv(name)
    if value:
        print(f"env:{name}: OK ({value})")
        return True
    if prefer:
        print(f"env:{name}: {'WARN' if optional else 'MISSING'} (recommended: {prefer})")
    else:
        print(f"env:{name}: {'WARN' if optional else 'MISSING'}")
    return optional


def check_torch_cuda() -> bool:
    if importlib.util.find_spec("torch") is None:
        print("torch_cuda: MISSING (torch not installed)")
        return False
    import torch

    available = torch.cuda.is_available()
    count = torch.cuda.device_count() if available else 0
    print(f"torch_cuda: {'OK' if available else 'FAIL'} ({count} visible GPU(s))")
    return available


if __name__ == "__main__":
    main()
