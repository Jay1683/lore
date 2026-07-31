"""
RAG building blocks: chunk text, embed + store it in Chroma, retrieve
relevant chunks for a query.

Design choices worth understanding:

- One Chroma *collection* per uploaded document, named by doc_id. This
  keeps documents isolated from each other -- querying doc A's collection
  never surfaces chunks from doc B.
- We use an in-memory (ephemeral) Chroma client, not a persistent one on
  disk. Documents are session-scoped: when the user uploads a new file or
  restarts the app, there's no reason to keep old embeddings around. If we
  later want chat history / documents to survive app restarts, that's a
  deliberate upgrade to a PersistentClient, not an accident.
- Chunking is hand-rolled (simple sliding window over characters) rather
  than pulling in a text-splitting library. It's a genuinely simple
  operation and worth understanding directly rather than treating as a
  black box.
"""

import hashlib

import chromadb
from chromadb.utils import embedding_functions

CHUNK_SIZE = 1000  # characters per chunk
CHUNK_OVERLAP = 150  # characters shared between consecutive chunks
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # small, fast, good-enough general model

# Lazy-loaded for the same reason as tiktoken in documents.py: this model
# downloads from Hugging Face on first use, then caches locally. We don't
# want importing this module to require network access.
_embedding_fn_cache = None
_client_cache = None


def _get_embedding_fn():
    global _embedding_fn_cache
    if _embedding_fn_cache is None:
        _embedding_fn_cache = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
    return _embedding_fn_cache


def _get_client():
    global _client_cache
    if _client_cache is None:
        _client_cache = chromadb.Client()  # ephemeral, in-memory
    return _client_cache


def chunk_text(
    text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> list[str]:
    """
    Simple sliding-window chunker over raw characters. Not sentence- or
    paragraph-aware -- it will happily cut mid-sentence. That's an
    acceptable tradeoff for a first version: the overlap (150 chars) means
    a sentence cut off at the end of one chunk still appears in full at
    the start of the next one, so retrieval rarely loses the relevant
    context entirely. This is the parameter to revisit first if retrieval
    quality is ever disappointing.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= text_len:
            break
        start = end - overlap

    return chunks


def _safe_collection_name(doc_id: str) -> str:
    """
    Chroma collection names are restricted to [a-zA-Z0-9._-], 3-512 chars.
    Real filenames routinely violate this (spaces, commas, parentheses,
    unicode, etc.), so rather than trying to sanitize-and-hope, we hash
    doc_id into a fixed-length hex string. This is both always valid
    (hex digits are a strict subset of the allowed charset) and collision-
    -safe (two different filenames that happened to sanitize to the same
    string would otherwise silently share a collection).
    """
    return "doc_" + hashlib.sha256(doc_id.encode("utf-8")).hexdigest()[:32]


def build_collection(doc_id: str, text: str):
    """
    Chunks the document and stores the embeddings in a fresh Chroma
    collection derived from doc_id. If a collection for that doc_id
    already exists (e.g. the same file re-uploaded), it's deleted and
    rebuilt rather than accumulating duplicate chunks.
    """
    client = _get_client()
    collection_name = _safe_collection_name(doc_id)

    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass  # collection didn't exist yet -- fine

    collection = client.create_collection(
        name=collection_name, embedding_function=_get_embedding_fn()
    )

    chunks = chunk_text(text)
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    collection.add(documents=chunks, ids=ids)

    return collection


def retrieve_chunks(collection, query: str, k: int = 4) -> list[str]:
    """Returns the top-k most relevant chunks for the query."""
    results = collection.query(query_texts=[query], n_results=k)
    return results["documents"][0]
