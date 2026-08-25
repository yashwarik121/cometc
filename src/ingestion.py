"""
Knowledge base ingestion pipeline.
Parses markdown files with YAML frontmatter, chunks by heading,
and indexes into ChromaDB + BM25.
"""

import os
import re
import pickle
from pathlib import Path
from typing import Optional

import frontmatter
import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

from src.config import (
    KNOWLEDGE_BASE_DIR, CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME,
    EMBEDDING_MODEL,
)
from src.models import ChunkMetadata


# ---------- Chunk data structure ----------

class Chunk:
    """A single chunk of text from a knowledge base document."""

    def __init__(self, chunk_id: str, text: str, metadata: ChunkMetadata):
        self.chunk_id = chunk_id
        self.text = text
        self.metadata = metadata

    def __repr__(self):
        return f"Chunk({self.chunk_id}, {self.metadata.source_file}#{self.metadata.heading})"


# ---------- Markdown parsing ----------

def parse_frontmatter(filepath: Path) -> dict:
    """Parse YAML frontmatter from a markdown file."""
    post = frontmatter.load(str(filepath))
    return {
        "title": post.get("title", filepath.stem),
        "status": post.get("status", "active"),
        "doc_type": post.get("doc_type", "unknown"),
        "effective_date": post.get("effective_date", None),
        "superseded_by": post.get("superseded_by", None),
        "content": post.content,
    }


def chunk_by_heading(content: str, source_file: str, fm: dict) -> list[Chunk]:
    """
    Split markdown content by ## headings.
    Each chunk includes the # title as prefix for context.
    """
    chunks = []

    # Extract the top-level heading
    title_match = re.match(r'^#\s+(.+)$', content, re.MULTILINE)
    doc_title = title_match.group(1).strip() if title_match else fm["title"]

    # Split on ## headings
    sections = re.split(r'^(##\s+.+)$', content, flags=re.MULTILINE)

    # If no ## headings, treat the whole doc as one chunk
    if len(sections) <= 1:
        chunk_id = f"{source_file}::full"
        meta = ChunkMetadata(
            source_file=source_file,
            heading=doc_title,
            title=fm["title"],
            status=fm["status"],
            doc_type=fm["doc_type"],
            effective_date=fm.get("effective_date"),
            superseded_by=fm.get("superseded_by"),
        )
        text = content.strip()
        if text:
            chunks.append(Chunk(chunk_id, text, meta))
        return chunks

    # Process the part before any ## heading
    preamble = sections[0].strip()
    if preamble and not preamble.startswith("#"):
        chunk_id = f"{source_file}::preamble"
        meta = ChunkMetadata(
            source_file=source_file,
            heading=doc_title,
            title=fm["title"],
            status=fm["status"],
            doc_type=fm["doc_type"],
            effective_date=fm.get("effective_date"),
            superseded_by=fm.get("superseded_by"),
        )
        chunks.append(Chunk(chunk_id, preamble, meta))

    # Process ## heading + content pairs
    for i in range(1, len(sections), 2):
        heading = sections[i].strip().lstrip("#").strip()
        body = sections[i + 1].strip() if i + 1 < len(sections) else ""

        # Combine heading with body for a meaningful chunk
        text = f"{doc_title} > {heading}\n{body}"
        chunk_id = f"{source_file}::{heading.lower().replace(' ', '_')}"

        meta = ChunkMetadata(
            source_file=source_file,
            heading=heading,
            title=fm["title"],
            status=fm["status"],
            doc_type=fm["doc_type"],
            effective_date=fm.get("effective_date"),
            superseded_by=fm.get("superseded_by"),
        )
        if text.strip():
            chunks.append(Chunk(chunk_id, text, meta))

    return chunks


# ---------- Indexing ----------

def load_and_chunk_all(kb_dir: Optional[Path] = None) -> list[Chunk]:
    """Load all markdown files and chunk them."""
    kb_dir = kb_dir or KNOWLEDGE_BASE_DIR
    all_chunks = []

    for filepath in sorted(kb_dir.glob("*.md")):
        fm = parse_frontmatter(filepath)
        source_file = filepath.name
        chunks = chunk_by_heading(fm["content"], source_file, fm)
        all_chunks.extend(chunks)

    return all_chunks


def build_chroma_index(chunks: list[Chunk], persist_dir: Optional[str] = None) -> chromadb.Collection:
    """Build/rebuild the ChromaDB collection from chunks."""
    persist_dir = persist_dir or CHROMA_PERSIST_DIR
    client = chromadb.PersistentClient(path=persist_dir)

    # Delete existing collection if present
    try:
        client.delete_collection(CHROMA_COLLECTION_NAME)
    except Exception:
        pass

    # Create embedding function
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Create collection
    collection = client.create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # Prepare batch data
    ids = [c.chunk_id for c in chunks]
    documents = [c.text for c in chunks]
    metadatas = [
        {k: v for k, v in c.metadata.model_dump().items() if v is not None}
        for c in chunks
    ]

    # Generate embeddings
    embeddings = model.encode(documents, show_progress_bar=True).tolist()

    # Add to collection
    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    return collection


def build_bm25_index(chunks: list[Chunk], persist_path: Optional[str] = None) -> BM25Okapi:
    """Build a BM25 index over the chunk texts."""
    tokenized = [c.text.lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized)

    # Save to disk
    if persist_path:
        with open(persist_path, "wb") as f:
            pickle.dump({"bm25": bm25, "chunk_ids": [c.chunk_id for c in chunks]}, f)

    return bm25


def run_ingestion(kb_dir: Optional[Path] = None, persist_dir: Optional[str] = None) -> tuple[list[Chunk], chromadb.Collection, BM25Okapi]:
    """Full ingestion pipeline: parse, chunk, index."""
    kb_dir = kb_dir or KNOWLEDGE_BASE_DIR
    persist_dir = persist_dir or CHROMA_PERSIST_DIR

    print(f"Loading knowledge base from {kb_dir}...")
    chunks = load_and_chunk_all(kb_dir)
    print(f"  Created {len(chunks)} chunks from {len(list(kb_dir.glob('*.md')))} files")

    print("Building ChromaDB index...")
    collection = build_chroma_index(chunks, persist_dir)
    print(f"  ChromaDB collection '{CHROMA_COLLECTION_NAME}' built with {collection.count()} documents")

    bm25_path = os.path.join(persist_dir, "bm25_index.pkl")
    print("Building BM25 index...")
    bm25 = build_bm25_index(chunks, bm25_path)
    print(f"  BM25 index built and saved to {bm25_path}")

    return chunks, collection, bm25


if __name__ == "__main__":
    run_ingestion()
