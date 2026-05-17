#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

from safetensors import safe_open
from safetensors.torch import save_file


TEXT_PREFIX = "model.language_model.language_model.language_model."
VISUAL_PREFIX = "model.language_model.visual."


def rename_key(key: str) -> str:
    if key.startswith(TEXT_PREFIX):
        return "model.language_model." + key[len(TEXT_PREFIX) :]
    if key.startswith(VISUAL_PREFIX):
        return "model.visual." + key[len(VISUAL_PREFIX) :]
    return key


def load_index(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def tensor_nbytes(tensor: Any) -> int:
    if hasattr(tensor, "nbytes"):
        return int(tensor.nbytes)
    return int(tensor.numel() * tensor.element_size())


def copy_non_weight_files(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for path in src.iterdir():
        if path.name.endswith(".safetensors") or path.name == "model.safetensors.index.json":
            continue
        if path.is_file():
            shutil.copy2(path, dst / path.name)


def rewrite_merged_shards(merged_dir: Path, output_dir: Path) -> tuple[dict[str, str], int]:
    index = load_index(merged_dir / "model.safetensors.index.json")
    by_shard: dict[str, list[str]] = defaultdict(list)
    for key, shard in index["weight_map"].items():
        by_shard[shard].append(key)

    weight_map: dict[str, str] = {}
    total_size = 0

    for shard in sorted(by_shard):
        tensors = {}
        with safe_open(merged_dir / shard, framework="pt", device="cpu") as handle:
            for old_key in by_shard[shard]:
                new_key = rename_key(old_key)
                if new_key in tensors:
                    raise ValueError(f"Duplicate key after rename: {new_key}")
                tensor = handle.get_tensor(old_key)
                tensors[new_key] = tensor
                weight_map[new_key] = shard
                total_size += tensor_nbytes(tensor)
        save_file(tensors, output_dir / shard)

    return weight_map, total_size


def add_missing_base_tensors(
    *,
    base_dir: Path,
    weight_map: dict[str, str],
    output_dir: Path,
    shard_name: str,
) -> int:
    base_index = load_index(base_dir / "model.safetensors.index.json")
    missing_keys = sorted(set(base_index["weight_map"]) - set(weight_map))
    if not missing_keys:
        return 0

    unexpected = [key for key in missing_keys if not key.startswith("mtp.")]
    if unexpected:
        joined = "\n".join(unexpected[:50])
        raise ValueError(f"Unexpected missing non-MTP keys after rename:\n{joined}")

    tensors = {}
    for key in missing_keys:
        base_shard = base_index["weight_map"][key]
        with safe_open(base_dir / base_shard, framework="pt", device="cpu") as handle:
            tensor = handle.get_tensor(key)
            tensors[key] = tensor
            weight_map[key] = shard_name

    save_file(tensors, output_dir / shard_name)
    return sum(tensor_nbytes(tensor) for tensor in tensors.values())


def write_index(output_dir: Path, weight_map: dict[str, str], total_size: int) -> None:
    index = {
        "metadata": {"total_size": total_size},
        "weight_map": dict(sorted(weight_map.items())),
    }
    (output_dir / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def validate_against_base(base_dir: Path, output_dir: Path) -> None:
    base_keys = set(load_index(base_dir / "model.safetensors.index.json")["weight_map"])
    output_keys = set(load_index(output_dir / "model.safetensors.index.json")["weight_map"])
    missing = sorted(base_keys - output_keys)
    extra = sorted(output_keys - base_keys)
    if missing or extra:
        raise ValueError(
            "Repaired index does not match base key set.\n"
            f"Missing ({len(missing)}): {missing[:20]}\n"
            f"Extra ({len(extra)}): {extra[:20]}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Repair LLaMA-Factory Qwen3.5 merged safetensors so keys match the original Qwen3.5 checkpoint."
    )
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--merged-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"Output already exists: {args.output_dir}. Pass --overwrite to replace it.")
        shutil.rmtree(args.output_dir)

    args.output_dir.mkdir(parents=True)
    copy_non_weight_files(args.merged_dir, args.output_dir)

    weight_map, total_size = rewrite_merged_shards(args.merged_dir, args.output_dir)
    total_size += add_missing_base_tensors(
        base_dir=args.base_dir,
        weight_map=weight_map,
        output_dir=args.output_dir,
        shard_name="model-00013-of-00013.safetensors",
    )
    write_index(args.output_dir, weight_map, total_size)
    validate_against_base(args.base_dir, args.output_dir)

    print(f"Wrote repaired merged model: {args.output_dir}")
    print(f"Tensors: {len(weight_map)}")
    print(f"Total size: {total_size}")


if __name__ == "__main__":
    main()
