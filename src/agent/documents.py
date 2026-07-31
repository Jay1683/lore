"""
Document extraction utilities.

Deliberately kept as plain functions with no graph/LLM dependencies -- this
module only knows how to turn file bytes into text and count tokens. It
gets called once, at upload time, from app.py. The graph never re-parses
the document on every turn.
"""

import io

import fitz  # pymupdf
import tiktoken
from docx import Document as DocxDocument

# cl100k_base is the encoding used by GPT-4/3.5. Groq's llama models use a
# different tokenizer internally, so this is an *approximation*, not an
# exact count. That's fine for our purpose -- we only need a rough token
# count to decide "small enough to stuff raw" vs "needs RAG", not an exact
# billing-grade count.
#
# Loaded lazily (on first actual use) rather than at import time: tiktoken
# fetches this encoding's merge file over the network on first use ever
# (then caches it to disk permanently). Loading it eagerly at import time
# would mean simply importing this module requires network access, even
# for code paths that never call count_tokens().
_encoding_cache = None


def _get_encoding():
    global _encoding_cache
    if _encoding_cache is None:
        _encoding_cache = tiktoken.get_encoding("cl100k_base")
    return _encoding_cache


def extract_text(file_bytes: bytes, filename: str) -> str:
    """
    Dispatches to the right extractor based on file extension.
    Raises ValueError for unsupported types so callers fail loudly
    instead of silently returning empty text.
    """
    lower_name = filename.lower()

    if lower_name.endswith(".pdf"):
        return _extract_pdf(file_bytes)
    elif lower_name.endswith(".docx"):
        return _extract_docx(file_bytes)
    else:
        raise ValueError(f"Unsupported file type: {filename}")


def _extract_pdf(file_bytes: bytes) -> str:
    text_parts = []
    with fitz.open(stream=file_bytes, filetype="pdf") as pdf:
        for page in pdf:
            text_parts.append(page.get_text())
    return "\n".join(text_parts)


def _extract_docx(file_bytes: bytes) -> str:
    doc = DocxDocument(io.BytesIO(file_bytes))
    paragraphs = [p.text for p in doc.paragraphs]
    return "\n".join(paragraphs)


def count_tokens(text: str) -> int:
    """Approximate token count -- see note on _get_encoding above."""
    return len(_get_encoding().encode(text))
