from __future__ import annotations

import hashlib
import math
import os
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


def _stable_bucket(token: str, dimensions: int) -> int:
    digest = hashlib.md5(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % dimensions


@dataclass
class HashEmbeddingMemoryStore:
    """Small dependency-free embedding store for local demos.

    It uses hashed bag-of-words vectors, which is not semantic like a real embedding
    model, but exercises the same retrieval interface and ranking path.
    """

    dimensions: int = 128
    memories: dict[tuple[str, str], list[tuple[str, list[float]]]] = field(default_factory=dict)

    def add(self, session_id: str, npc_id: str, text: str) -> None:
        clean = text.strip()
        if not clean:
            return
        key = (session_id, npc_id)
        self.memories.setdefault(key, []).append((clean, self._embed(clean)))
        self.memories[key] = self.memories[key][-200:]

    def search(self, session_id: str, npc_id: str, query: str, k: int = 4) -> list[str]:
        key = (session_id, npc_id)
        candidates = self.memories.get(key, [])
        if not candidates:
            return []
        query_vec = self._embed(query)
        scored = []
        for idx, (memory, vector) in enumerate(candidates):
            similarity = _cosine(query_vec, vector)
            recency = idx / max(1, len(candidates))
            scored.append((similarity, recency, memory))
        hits = [memory for similarity, _, memory in sorted(scored, reverse=True)[:k] if similarity > 0]
        return hits or [memory for memory, _ in candidates[-k:]]

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in _tokens(text):
            vector[_stable_bucket(token, self.dimensions)] += 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


class SentenceTransformerMemoryStore(HashEmbeddingMemoryStore):
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        super().__init__(dimensions=384)
        self.model = SentenceTransformer(model_name)

    def _embed(self, text: str) -> list[float]:
        vector = self.model.encode(text, normalize_embeddings=True)
        return [float(value) for value in vector]


def _cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def create_memory_store() -> MemoryStore:
    backend = os.getenv("NPC_MEMORY_BACKEND", "keyword").lower()
    if backend in {"hash", "embedding", "embeddings"}:
        return HashEmbeddingMemoryStore()
    if backend in {"sentence-transformers", "sentence_transformers", "st"}:
        model = os.getenv("NPC_MEMORY_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        try:
            return SentenceTransformerMemoryStore(model)
        except Exception:
            return InMemoryMemoryStore()
    return InMemoryMemoryStore()
