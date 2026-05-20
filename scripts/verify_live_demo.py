#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from urllib.request import Request, urlopen


def request_json(url: str, payload: dict | None = None) -> dict:
    if payload is None:
        with urlopen(url, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> None:
    base = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080"
    health = request_json(f"{base}/health")
    world = request_json(f"{base}/world")
    assert health["status"] == "ok", health
    assert len(world["npcs"]) == 5, world["npcs"].keys()
    assert len(world["locations"]) == 3, world["locations"].keys()
    payload = {
        "session_id": "verify",
        "npc_id": "mika",
        "player_input": "The tide engine is overheating. I found the broken valve near the pier.",
    }
    result = request_json(f"{base}/chat", payload)
    response = result["response"]
    assert response["dialogue"], result
    assert response["action"] in world["npcs"]["mika"]["allowed_actions"], result
    print(json.dumps({"health": health, "response": response, "events": result["events"]}, indent=2))


if __name__ == "__main__":
    main()
