"""
Hybrid retrieval: BM25 + dense embeddings merged via RRF,
then cross-encoder reranking with precedence logic and conflict detection.
"""

import os
import pickle
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
from rank_bm25 import BM25Okapi

from src.config import (
    CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME, EMBEDDING_MODEL,
    RERANKER_MODEL, BM25_TOP_K, DENSE_TOP_K, RRF_K,
    RERANK_TOP_N, RERANK_CONFIDENCE_THRESHOLD,
)
from src.models import RetrievalResult, RetrievalResponse, ChunkMetadata
from src.ingestion import Chunk


class Retriever:
    """Hybrid BM25 + dense retrieval with RRF fusion and cross-encoder reranking."""

    def __init__(
        self,
        chunks: list[Chunk],
        collection: chromadb.Collection,
        bm25: BM25Okapi,
        embedding_model: Optional[SentenceTransformer] = None,
        reranker: Optional[CrossEncoder] = None,
    ):
        self.chunks = chunks
        self.chunk_map = {c.chunk_id: c for c in chunks}
        self.collection = collection
        self.bm25 = bm25
        self.embedder = embedding_model or SentenceTransformer(EMBEDDING_MODEL)
        self.reranker = reranker or CrossEncoder(RERANKER_MODEL)

    def bm25_search(self, query: str, k: int = BM25_TOP_K) -> list[tuple[str, float]]:
        """BM25 search. Returns list of (chunk_id, score)."""
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:k]
        results = []
        for idx in top_indices:
            if idx < len(self.chunks) and scores[idx] > 0:
                results.append((self.chunks[idx].chunk_id, float(scores[idx])))
        return results

    def dense_search(self, query: str, k: int = DENSE_TOP_K) -> list[tuple[str, float]]:
        """ChromaDB dense embedding search. Returns list of (chunk_id, score)."""
        query_embedding = self.embedder.encode([query]).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=min(k, self.collection.count()),
            include=["distances"],
        )
        pairs = []
        if results["ids"] and results["ids"][0]:
            for chunk_id, distance in zip(results["ids"][0], results["distances"][0]):
                # ChromaDB cosine distance: lower = more similar. Convert to similarity.
                similarity = 1.0 - distance
                pairs.append((chunk_id, similarity))
        return pairs

    def rrf_fuse(
        self,
        bm25_results: list[tuple[str, float]],
        dense_results: list[tuple[str, float]],
        k: int = RRF_K,
    ) -> list[tuple[str, float]]:
        """
        Reciprocal Rank Fusion.
        Score = sum(1 / (k + rank)) across result lists.
        """
        scores: dict[str, float] = {}

        for rank, (chunk_id, _) in enumerate(bm25_results):
            scores[chunk_id] = scores.get(chunk_id, 0) + 1.0 / (k + rank + 1)

        for rank, (chunk_id, _) in enumerate(dense_results):
            scores[chunk_id] = scores.get(chunk_id, 0) + 1.0 / (k + rank + 1)

        # Sort by RRF score descending
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked

    def cross_encoder_rerank(
        self, query: str, candidates: list[tuple[str, float]], top_n: int = RERANK_TOP_N
    ) -> list[tuple[str, float]]:
        """Rerank candidates using cross-encoder."""
        if not candidates:
            return []

        # Prepare pairs
        chunk_ids = [cid for cid, _ in candidates[:top_n * 3]]  # Take more candidates for reranking
        texts = []
        valid_ids = []
        for cid in chunk_ids:
            chunk = self.chunk_map.get(cid)
            if chunk:
                texts.append(chunk.text)
                valid_ids.append(cid)

        if not texts:
            return []

        # Score with cross-encoder
        pairs = [(query, text) for text in texts]
        scores = self.reranker.predict(pairs)

        # Combine and sort
        scored = list(zip(valid_ids, [float(s) for s in scores]))
        scored.sort(key=lambda x: x[1], reverse=True)

        return scored[:top_n]

    def _apply_precedence(self, results: list[tuple[str, float]]) -> tuple[list[tuple[str, float]], bool, str | None]:
        """
        Apply precedence logic:
        - Prefer active over superseded when both match the same topic
        - If two active docs conflict, flag it
        Returns: (filtered_results, has_conflict, conflict_description)
        """
        if not results:
            return results, False, None

        # Group by status
        active_results = []
        superseded_results = []

        for chunk_id, score in results:
            chunk = self.chunk_map.get(chunk_id)
            if not chunk:
                continue
            if chunk.metadata.status == "superseded":
                superseded_results.append((chunk_id, score))
            else:
                active_results.append((chunk_id, score))

        # Check for conflicts between active docs on the same topic
        has_conflict = False
        conflict_description = None

        if len(active_results) >= 2:
            active_files = set()
            for cid, _ in active_results:
                chunk = self.chunk_map.get(cid)
                if chunk:
                    active_files.add(chunk.metadata.source_file)

            # Check if different active source files have potentially conflicting content
            # This is a heuristic: if multiple active policy docs are retrieved with high scores,
            # flag a potential conflict
            policy_files = set()
            for cid, score in active_results:
                chunk = self.chunk_map.get(cid)
                if chunk and chunk.metadata.doc_type == "policy" and score > 0:
                    policy_files.add(chunk.metadata.source_file)

            if len(policy_files) >= 2:
                # Check if they cover overlapping topics (e.g., two return policies)
                topics = {}
                for cid, score in active_results:
                    chunk = self.chunk_map.get(cid)
                    if chunk and chunk.metadata.doc_type == "policy":
                        base_topic = chunk.metadata.source_file.split("-")[0]  # e.g., 'return' from 'return-policy.md'
                        if base_topic not in topics:
                            topics[base_topic] = []
                        topics[base_topic].append(chunk.metadata.source_file)

                for topic, files in topics.items():
                    unique_files = set(files)
                    if len(unique_files) >= 2:
                        has_conflict = True
                        conflict_description = (
                            f"Multiple active policy documents found for '{topic}': "
                            f"{', '.join(sorted(unique_files))}. "
                            f"These may contain different information — please review both sources."
                        )
                        break

        # Build final list: active first, then superseded (marked)
        final = active_results.copy()
        if superseded_results and not active_results:
            # Only superseded results found — include them but note status
            final = superseded_results
        elif superseded_results:
            # Include superseded only if explicitly asked about old/previous policy
            final.extend(superseded_results)

        return final, has_conflict, conflict_description

    def retrieve(self, query: str) -> RetrievalResponse:
        """
        Full retrieval pipeline: BM25 + dense → RRF → rerank → precedence.
        """
        # Step 1: BM25 search
        bm25_results = self.bm25_search(query)

        # Step 2: Dense search
        dense_results = self.dense_search(query)

        # Step 3: RRF fusion
        fused = self.rrf_fuse(bm25_results, dense_results)

        # Step 4: Cross-encoder reranking
        reranked = self.cross_encoder_rerank(query, fused)

        # Step 5: Precedence logic
        final, has_conflict, conflict_desc = self._apply_precedence(reranked)

        # Step 6: Build response
        retrieval_results = []
        for chunk_id, score in final:
            chunk = self.chunk_map.get(chunk_id)
            if chunk:
                retrieval_results.append(RetrievalResult(
                    text=chunk.text,
                    score=score,
                    metadata=chunk.metadata,
                    chunk_id=chunk_id,
                ))

        # Check confidence
        low_confidence = False
        if not retrieval_results:
            low_confidence = True
        elif retrieval_results[0].score < RERANK_CONFIDENCE_THRESHOLD:
            low_confidence = True

        return RetrievalResponse(
            chunks=retrieval_results,
            has_conflict=has_conflict,
            conflict_description=conflict_desc,
            low_confidence=low_confidence,
        )


# Module-level singleton for the retriever
_retriever: Retriever | None = None


def init_retriever(chunks: list[Chunk], collection: chromadb.Collection, bm25: BM25Okapi) -> Retriever:
    """Initialize the global retriever instance."""
    global _retriever
    _retriever = Retriever(chunks, collection, bm25)
    return _retriever


def get_retriever() -> Retriever:
    """Get the global retriever. Must be initialized first."""
    if _retriever is None:
        raise RuntimeError("Retriever not initialized. Call init_retriever first.")
    return _retriever
