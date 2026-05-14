from __future__ import annotations

import json
import os
from typing import Any

from openai import OpenAI


JUDGE_PROMPT = """Score the NPC response from 0.0 to 1.0.
Criteria: in-character behavior, quest progress, grounded world facts, no system leakage.
Return only JSON: {"score": number, "reason": string}.
"""


def judge_response(record: dict[str, Any], response: str) -> tuple[float, str]:
    base_url = os.getenv("JUDGE_BASE_URL")
    model = os.getenv("JUDGE_MODEL")
    if not base_url or not model:
        return 0.0, "LLM judge disabled"
    client = OpenAI(base_url=base_url, api_key=os.getenv("JUDGE_API_KEY", "EMPTY"))
    result = client.chat.completions.create(
        model=model,
        temperature=0.0,
        messages=[
            {"role": "system", "content": JUDGE_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "persona": record.get("persona"),
                        "setting": record.get("setting"),
                        "goal": record.get("goal"),
                        "response": response,
                    },
                    ensure_ascii=False,
                ),
            },
        ],
    )
    content = result.choices[0].message.content or "{}"
    try:
        payload = json.loads(content)
        return float(payload.get("score", 0.0)), str(payload.get("reason", ""))
    except Exception:
        return 0.0, f"Invalid judge JSON: {content[:120]}"
