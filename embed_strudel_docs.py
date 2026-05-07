"""Embed Strudel documentation into a FAISS index.

Improvements over the previous naive 500-char windowed chunker:
  * Semantic chunking by markdown-style headings and blank lines.
  * Each chunk has a soft min/max length and respects sentence boundaries.
  * Chunks include a small overlap with previous text to preserve context
    around heading transitions.
  * Metadata records both the source page and the local heading (when found).
"""

import os
import re

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# === CONFIG ===
DOC_DIR = "Data"
MIN_CHUNK_CHARS = 200       # smaller chunks get merged forward
MAX_CHUNK_CHARS = 900       # larger blocks get split at sentence boundaries
OVERLAP_CHARS = 80          # carry-over between adjacent chunks
BATCH_SIZE = 16
MODEL_NAME = "all-MiniLM-L6-v2"
DEVICE = "cpu"

# === Heading detection ===
# Treat lines that look like section titles as split points. Strudel pages
# scraped to text don't have markdown markers, so we use heuristic cues:
#   * lines that are short (< 60 chars) AND title-cased / single word
#   * lines starting with "URL:" mark a new page section.
_HEADING_RE = re.compile(
    r"^(?:URL:.*|[A-Z][A-Za-z0-9 _\-]{0,58}|#+\s.+)$"
)
_SENT_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def _looks_like_heading(line: str) -> bool:
    s = line.strip()
    if not s or len(s) > 80:
        return False
    if s.startswith("URL:") or s.startswith("#"):
        return True
    if len(s.split()) <= 6 and s[:1].isupper() and not s.endswith("."):
        return True
    return False


def split_into_blocks(text: str) -> list[tuple[str, str]]:
    """Split a doc into (heading, body) blocks using heuristic headings.

    Returns a list of (heading_label, body_text). The first block may have
    an empty heading.
    """
    lines = text.splitlines()
    blocks: list[tuple[str, list[str]]] = [("", [])]
    for line in lines:
        if _looks_like_heading(line):
            # start a new block
            blocks.append((line.strip(), []))
        else:
            blocks[-1][1].append(line)
    return [(h, "\n".join(b).strip()) for h, b in blocks if "\n".join(b).strip()]


def split_long_block(body: str, max_chars: int) -> list[str]:
    """Split a long block at sentence boundaries to honor max_chars."""
    if len(body) <= max_chars:
        return [body]
    parts: list[str] = []
    sentences = _SENT_END.split(body)
    cur: list[str] = []
    cur_len = 0
    for sent in sentences:
        if cur_len + len(sent) + 1 > max_chars and cur:
            parts.append(" ".join(cur).strip())
            cur = [sent]
            cur_len = len(sent)
        else:
            cur.append(sent)
            cur_len += len(sent) + 1
    if cur:
        parts.append(" ".join(cur).strip())
    return parts


def chunk_doc(text: str, source: str) -> list[tuple[str, dict]]:
    """Return [(chunk_text, metadata), ...] for a single document."""
    out: list[tuple[str, dict]] = []
    blocks = split_into_blocks(text)
    pending = ""
    pending_heading = ""
    for heading, body in blocks:
        for piece in split_long_block(body, MAX_CHUNK_CHARS):
            text_piece = piece.strip()
            if not text_piece:
                continue
            if pending and len(pending) < MIN_CHUNK_CHARS:
                # merge with pending small chunk
                merged_heading = pending_heading or heading
                merged = (pending + "\n\n" + text_piece).strip()
                if len(merged) <= MAX_CHUNK_CHARS:
                    pending = merged
                    pending_heading = merged_heading
                    continue
                # otherwise flush pending and start fresh
                out.append((pending, {"source": source, "heading": pending_heading}))
                pending = text_piece
                pending_heading = heading
            else:
                if pending:
                    out.append((pending, {"source": source, "heading": pending_heading}))
                pending = text_piece
                pending_heading = heading

    if pending:
        out.append((pending, {"source": source, "heading": pending_heading}))

    # add overlap by prepending the tail of previous chunk
    if OVERLAP_CHARS > 0 and len(out) > 1:
        with_overlap: list[tuple[str, dict]] = [out[0]]
        for i in range(1, len(out)):
            prev_text = out[i - 1][0]
            cur_text, meta = out[i]
            tail = prev_text[-OVERLAP_CHARS:]
            with_overlap.append((tail + " ... " + cur_text, meta))
        out = with_overlap

    return out


def main():
    print(f"Loading model '{MODEL_NAME}' on {DEVICE}...")
    model = SentenceTransformer(MODEL_NAME, device=DEVICE)

    if not os.path.isdir(DOC_DIR):
        raise SystemExit(
            f"Doc directory '{DOC_DIR}' not found. Run scraping.py first."
        )

    print(f"Reading and chunking docs in '{DOC_DIR}' (semantic chunker)...")
    all_chunks: list[str] = []
    all_meta: list[str] = []  # kept as flat strings for backwards compatibility

    for filename in sorted(os.listdir(DOC_DIR)):
        if not filename.endswith(".txt"):
            continue
        path = os.path.join(DOC_DIR, filename)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        for chunk_text_, meta in chunk_doc(content, filename):
            all_chunks.append(chunk_text_)
            head = meta.get("heading", "")
            all_meta.append(f"{filename}::{head}" if head else filename)

    print(f"Total chunks: {len(all_chunks)}")
    if not all_chunks:
        raise SystemExit("No chunks produced — check Data/ contents.")

    print("Generating embeddings (CPU)...")
    embeddings: list = []
    for i in tqdm(range(0, len(all_chunks), BATCH_SIZE)):
        batch = all_chunks[i : i + BATCH_SIZE]
        embeddings.extend(model.encode(batch))

    embs = np.array(embeddings, dtype="float32")
    print(f"Embeddings shape: {embs.shape}")

    dim = embs.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embs)

    faiss.write_index(index, "faiss_index.bin")
    np.save("doc_chunks.npy", np.array(all_chunks, dtype=object))
    np.save("doc_metadata.npy", np.array(all_meta, dtype=object))

    print("Saved: faiss_index.bin, doc_chunks.npy, doc_metadata.npy")


if __name__ == "__main__":
    main()
