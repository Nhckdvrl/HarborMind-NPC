#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a lightweight dataset/demo report.")
    parser.add_argument("--manifest", type=Path, default=Path("data/product_manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/product_report.md"))
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_report(manifest), encoding="utf-8")
    print(args.output)


def render_report(manifest: dict) -> str:
    counts = manifest.get("counts", {})
    lines = [
        "# GameNPC-RL Product Report",
        "",
        "## Data Counts",
        "",
    ]
    for key, value in sorted(counts.items()):
        lines.append(f"- `{key}`: {value}")
    lines.extend(
        [
            "",
            "## Evaluation Axes",
            "",
            "- JSON validity",
            "- Action validity",
            "- Role adherence",
            "- Quest progression",
            "- Memory write / recall",
            "- System leakage rate",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
