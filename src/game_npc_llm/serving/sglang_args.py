from __future__ import annotations

import argparse
import shlex
from pathlib import Path
from typing import Any

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Render an SGLang OpenAI-compatible server command.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--profile", required=True)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    print(render_command(config[args.profile]))


def render_command(profile: dict[str, Any]) -> str:
    cmd = [
        "python",
        "-m",
        "sglang.launch_server",
        "--model-path",
        profile["model"],
        "--served-model-name",
        profile["served_model_name"],
        "--host",
        str(profile.get("host", "0.0.0.0")),
        "--port",
        str(profile.get("port", 30000)),
        "--tensor-parallel-size",
        str(profile.get("tensor_parallel_size", 1)),
        "--context-length",
        str(profile.get("context_length", profile.get("max_model_len", 8192))),
    ]

    value_flags = {
        "mem_fraction_static": "--mem-fraction-static",
        "dtype": "--dtype",
        "quantization": "--quantization",
        "kv_cache_dtype": "--kv-cache-dtype",
        "chat_template": "--chat-template",
    }
    for key, flag in value_flags.items():
        if profile.get(key) is not None:
            cmd.extend([flag, str(profile[key])])

    for flag in profile.get("boolean_flags", []):
        cmd.append(str(flag))
    for flag, value in profile.get("extra_args", {}).items():
        cmd.append(str(flag))
        if value is not None:
            cmd.append(str(value))

    return " ".join(shlex.quote(part) for part in cmd)


if __name__ == "__main__":
    main()
