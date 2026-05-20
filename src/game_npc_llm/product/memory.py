from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol


class MemoryStore(Protocol):
    def add(self, session_id: str, npc_id: str, text: str) -> None: ...

    def search(self, session_id: str, npc_id: str, query: str, k: int = 4) -> list[str]: ...


@dataclass
class InMemoryMemoryStore:
    memories: dict[tuple[str, str], list[str]] = field(default_factory=dict)

    def add(self, session_id: str, npc_id: str, text: str) -> None:
        clean = text.strip()
        if not clean:
            return
        key = (session_id, npc_id)
        self.memories.setdefault(key, []).append(clean)
        self.memories[key] = self.memories[key][-200:]

    def search(self, session_id: str, npc_id: str, query: str, k: int = 4) -> list[str]:
        key = (session_id, npc_id)
        candidates = self.memories.get(key, [])
        query_terms = set(_tokens(query))
        if not query_terms:
            return candidates[-k:]
        scored = []
        for idx, memory in enumerate(candidates):
            overlap = len(query_terms & set(_tokens(memory)))
            recency = idx / max(1, len(candidates))
            scored.append((overlap, recency, memory))
        hits = [memory for overlap, _, memory in sorted(scored, reverse=True)[:k] if overlap > 0]
        return hits or candidates[-k:]


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in re.findall(r"[a-zA-Z0-9_]{3,}", text)]
