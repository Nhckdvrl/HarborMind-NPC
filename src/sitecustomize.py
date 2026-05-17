from __future__ import annotations

import os


def _patch_sglang_http_server() -> None:
    try:
        import sglang.srt.entrypoints.http_server as http_server
        from sglang.srt.entrypoints.engine import Engine
    except Exception:
        return
    if not hasattr(http_server, "_launch_subprocesses") and hasattr(Engine, "_launch_subprocesses"):
        http_server._launch_subprocesses = Engine._launch_subprocesses


def _patch_qwen35_sglang_transformers_backend() -> None:
    try:
        import transformers
    except Exception:
        return
    model_cls = getattr(transformers, "Qwen3_5ForConditionalGeneration", None)
    if model_cls is not None:
        model_cls.is_backend_compatible = classmethod(lambda cls: True)


if os.getenv("ENABLE_SGLANG_VERL_PATCH") == "1":
    _patch_sglang_http_server()
    _patch_qwen35_sglang_transformers_backend()
