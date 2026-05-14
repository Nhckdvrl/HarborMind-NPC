from __future__ import annotations

SFT_REQUIRED_KEYS = {"id", "source", "split", "messages", "metadata"}
GRPO_REQUIRED_KEYS = {
    "id",
    "source",
    "split",
    "prompt",
    "persona",
    "setting",
    "goal",
    "allowed_entities",
    "metadata",
}
EVAL_REQUIRED_KEYS = {
    "id",
    "source",
    "split",
    "category",
    "prompt",
    "reference",
    "persona",
    "setting",
    "goal",
    "allowed_entities",
    "checks",
    "metadata",
}


def validate_record(record: dict, schema: str) -> list[str]:
    if schema == "sft":
        return _validate_sft(record)
    if schema == "grpo":
        return _validate_grpo(record)
    if schema == "eval":
        return _validate_eval(record)
    raise ValueError(f"Unknown schema: {schema}")


def _missing(record: dict, required: set[str]) -> list[str]:
    return [f"missing key: {key}" for key in sorted(required - set(record))]


def _validate_sft(record: dict) -> list[str]:
    errors = _missing(record, SFT_REQUIRED_KEYS)
    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        errors.append("messages must contain at least user and assistant turns")
    else:
        for idx, message in enumerate(messages):
            if message.get("role") not in {"system", "user", "assistant"}:
                errors.append(f"messages[{idx}].role is invalid")
            if not isinstance(message.get("content"), str) or not message["content"].strip():
                errors.append(f"messages[{idx}].content must be non-empty")
    return errors


def _validate_grpo(record: dict) -> list[str]:
    errors = _missing(record, GRPO_REQUIRED_KEYS)
    for key in ("prompt", "persona", "setting", "goal"):
        if not isinstance(record.get(key), str) or not record.get(key, "").strip():
            errors.append(f"{key} must be a non-empty string")
    if not isinstance(record.get("allowed_entities"), list):
        errors.append("allowed_entities must be a list")
    return errors


def _validate_eval(record: dict) -> list[str]:
    errors = _missing(record, EVAL_REQUIRED_KEYS)
    for key in ("prompt", "persona", "setting", "reference"):
        if not isinstance(record.get(key), str):
            errors.append(f"{key} must be a string")
    if not isinstance(record.get("checks"), dict):
        errors.append("checks must be an object")
    return errors
