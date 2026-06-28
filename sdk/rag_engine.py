"""Retrieval-Augmented Generation engine.

A thin abstraction over a vector store with two backends behind one interface:

* **InMemoryBackend** (default) — lexical token-cosine search. Zero external
  services, deterministic, perfect for tests and offline demos.
* **ChromaBackend** — real ChromaDB (HTTP or persistent) when ``CHROMA_HOST``
  or a persist dir is configured.

Agents depend only on :class:`RAGEngine`, never on the backend, so the store
can be upgraded without touching agent code.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional

from sdk.config import Settings, get_settings
from sdk.hallucination_guard import RetrievedContext


@dataclass
class Document:
    text: str
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0


@dataclass
class RetrievalResult:
    documents: list[Document] = field(default_factory=list)

    @property
    def texts(self) -> list[str]:
        return [d.text for d in self.documents]

    @property
    def sources(self) -> list[str]:
        return [d.source for d in self.documents if d.source]

    def as_context(self) -> RetrievedContext:
        return RetrievedContext(documents=self.texts, sources=self.sources)


# --------------------------------------------------------------------------
# Lexical backend
# --------------------------------------------------------------------------
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _vec(text: str) -> Counter:
    return Counter(_TOKEN_RE.findall(text.lower()))


def _cosine(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    num = sum(a[t] * b[t] for t in common)
    da = math.sqrt(sum(v * v for v in a.values()))
    db = math.sqrt(sum(v * v for v in b.values()))
    return num / (da * db) if da and db else 0.0


class InMemoryBackend:
    def __init__(self) -> None:
        self._collections: dict[str, list[tuple[Document, Counter]]] = {}

    def index(self, docs: list[Document], collection: str) -> None:
        store = self._collections.setdefault(collection, [])
        store.extend((d, _vec(d.text)) for d in docs)

    def retrieve(self, query: str, collection: str, k: int) -> list[Document]:
        store = self._collections.get(collection, [])
        qv = _vec(query)
        scored = []
        for doc, dv in store:
            s = _cosine(qv, dv)
            if s > 0:
                scored.append(Document(doc.text, doc.source, dict(doc.metadata), score=s))
        scored.sort(key=lambda d: d.score, reverse=True)
        return scored[:k]

    def count(self, collection: str) -> int:
        return len(self._collections.get(collection, []))


class ChromaBackend:  # pragma: no cover - exercised only with live Chroma
    def __init__(self, settings: Settings) -> None:
        import chromadb

        if settings.chroma_host:
            self._client = chromadb.HttpClient(
                host=settings.chroma_host, port=settings.chroma_port
            )
        else:
            self._client = chromadb.PersistentClient(path=settings.chroma_persist_dir)

    def _col(self, name: str):
        return self._client.get_or_create_collection(name)

    def index(self, docs: list[Document], collection: str) -> None:
        col = self._col(collection)
        col.add(
            ids=[f"{collection}-{i}" for i in range(col.count(), col.count() + len(docs))],
            documents=[d.text for d in docs],
            metadatas=[{**d.metadata, "source": d.source} for d in docs],
        )

    def retrieve(self, query: str, collection: str, k: int) -> list[Document]:
        col = self._col(collection)
        res = col.query(query_texts=[query], n_results=k)
        docs: list[Document] = []
        for text, meta, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            meta = meta or {}
            docs.append(
                Document(
                    text=text,
                    source=meta.get("source", ""),
                    metadata=meta,
                    score=1.0 - dist,
                )
            )
        return docs

    def count(self, collection: str) -> int:
        return self._col(collection).count()


# --------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------
class RAGEngine:
    def __init__(self, settings: Optional[Settings] = None, backend=None) -> None:
        self.settings = settings or get_settings()
        self.backend = backend or self._build_backend()

    def _build_backend(self):
        if self.settings.chroma_host:
            try:
                return ChromaBackend(self.settings)
            except Exception:  # pragma: no cover
                pass
        return InMemoryBackend()

    def index_documents(self, docs: list[Document], collection_name: str) -> int:
        self.backend.index(docs, collection_name)
        return self.backend.count(collection_name)

    def retrieve(self, query: str, collection_name: str, k: int = 5) -> list[Document]:
        return self.backend.retrieve(query, collection_name, k)

    def retrieve_with_sources(
        self, query: str, collection_name: str, k: int = 5
    ) -> RetrievalResult:
        return RetrievalResult(documents=self.retrieve(query, collection_name, k))
