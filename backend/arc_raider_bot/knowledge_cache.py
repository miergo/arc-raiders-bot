"""Local RAG knowledge cache backed by ChromaDB + Ollama embeddings."""

from __future__ import annotations

import json
import uuid
import logging

import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

from arc_raider_bot.config import CHROMA_DATA_DIR, EMBEDDING_MODEL, OLLAMA_BASE_URL, CACHE_SIMILARITY_THRESHOLD
from arc_raider_bot.models import CachedQA

log = logging.getLogger(__name__)


class KnowledgeCache:
    """Semantic cache that stores and retrieves past Q&A pairs."""

    def __init__(
        self,
        data_dir: str | None = None,
        embedding_model: str | None = None,
        ollama_url: str | None = None,
    ):
        data_dir = data_dir or CHROMA_DATA_DIR
        embedding_model = embedding_model or EMBEDDING_MODEL
        ollama_url = ollama_url or OLLAMA_BASE_URL

        embed_url = ollama_url.replace("/v1", "/api/embeddings")

        self._ef = OllamaEmbeddingFunction(model_name=embedding_model, url=embed_url)
        self._client = chromadb.PersistentClient(path=data_dir)
        self._collection = self._client.get_or_create_collection(
            name="arc_raiders_qa",
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def count(self) -> int:
        return self._collection.count()

    def store(self, question: str, answer: str, sources: list[str]) -> None:
        """Embed and persist a Q&A pair."""
        doc_id = str(uuid.uuid4())
        self._collection.add(
            ids=[doc_id],
            documents=[question],
            metadatas=[{"answer": answer, "sources": json.dumps(sources)}],
        )
        log.debug("Stored Q&A (id=%s, total=%d)", doc_id, self.count)

    def search(
        self,
        question: str,
        n_results: int = 3,
        threshold: float | None = None,
    ) -> list[CachedQA]:
        """Find past Q&As semantically similar to *question*.

        Returns only results whose cosine similarity >= threshold.
        """
        threshold = threshold or CACHE_SIMILARITY_THRESHOLD

        if self.count == 0:
            return []

        results = self._collection.query(
            query_texts=[question],
            n_results=min(n_results, self.count),
            include=["metadatas", "documents", "distances"],
        )

        hits: list[CachedQA] = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            similarity = 1.0 - dist
            if similarity >= threshold:
                hits.append(CachedQA(
                    question=doc,
                    answer=meta["answer"],
                    sources=json.loads(meta.get("sources", "[]")),
                    similarity=similarity,
                ))

        hits.sort(key=lambda h: h.similarity, reverse=True)
        return hits
