#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys


def main() -> None:
    checks = {
        "pydantic": True,
        "fastapi": True,
        "uvicorn": True,
        "datasets": False,
        "llamafactory": False,
        "deepeval": False,
        "trl": False,
    }
    ok = True
    for package, required in checks.items():
        installed = importlib.util.find_spec(package) is not None
        status = "OK" if installed else ("OPTIONAL-MISSING" if not required else "MISSING")
        print(f"{package}: {status}")
        ok &= installed or not required
    print(f"python: {sys.version.split()[0]}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
