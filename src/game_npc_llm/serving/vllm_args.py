from __future__ import annotations

import argparse
import shlex
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a vLLM OpenAI server command.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--profile", required=True)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    profile = config[args.profile]
    cmd = [
        "python",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        profile["model"],
        "--served-model-name",
        profile["served_model_name"],
        "--host",
        "0.0.0.0",
        "--port",
        str(profile.get("port", 8000)),
        "--tensor-parallel-size",
        str(profile.get("tensor_parallel_size", 1)),
        "--max-model-len",
        str(profile.get("max_model_len", 8192)),
        "--trust-remote-code",
    ]
    if profile.get("enable_lora"):
        cmd.append("--enable-lora")
        for module in profile.get("lora_modules", []):
            cmd.extend(["--lora-modules", f"{module['name']}={module['path']}"])
    print(" ".join(shlex.quote(part) for part in cmd))


if __name__ == "__main__":
    main()
