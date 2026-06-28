"""Index the knowledge base (standards, manuals, procedures) into the RAG store.

Run directly: ``python -m data.seed_chromadb``. Uses ChromaDB when
``CHROMA_HOST`` is configured, otherwise the in-memory lexical backend.
"""
from __future__ import annotations

from data.bootstrap import seed_rag
from sdk.config import get_settings
from sdk.rag_engine import RAGEngine


def main() -> None:
    settings = get_settings()
    rag = RAGEngine(settings)
    backend = type(rag.backend).__name__
    print(f"Indexing knowledge base [{backend}]...")
    counts = seed_rag(rag)
    for collection, n in counts.items():
        print(f"  {collection}: {n} docs")
    # Quick retrieval sanity check.
    hits = rag.retrieve("bearing vibration high temperature", "manuals", k=2)
    print("\nSample query 'bearing vibration high temperature':")
    for h in hits:
        print(f"  [{h.score:.2f}] {h.source}: {h.text[:80]}...")
    print("Done.")


if __name__ == "__main__":
    main()
