#!/usr/bin/env python3
"""
build_embeddings.py
-------------------
PDF  ➜  cleaned text chunks  ➜  BatteryBERT vectors

Usage
-----
python build_embeddings.py --pdf_root data/raw_papers \
                           --model batterydata/batterybert-cased \
                           --chunk_size 350 --overlap 0.20
"""
import argparse, json, os, re, sys, math, glob, pathlib, hashlib
from typing import List, Tuple
import numpy as np
from sentence_transformers import models, SentenceTransformer

# --- third-party deps ---------------------------------------------------------
try:
    import fitz  # PyMuPDF – fast PDF text extraction
except ImportError:
    print("✘  Please `pip install pymupdf`", file=sys.stderr); sys.exit(1)

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("✘  Please `pip install sentence-transformers`", file=sys.stderr); sys.exit(1)

# -----------------------------------------------------------------------------


def pdf_to_pages(pdf_path: str) -> List[str]:
    """Extract page texts from a PDF with PyMuPDF."""
    doc = fitz.open(pdf_path)
    pages = [page.get_text("text") for page in doc]
    doc.close()
    return pages


_SENT_SPLIT_RE = re.compile(r"(?<=[\.\?\!])\s+")


def clean_text(text: str) -> str:
    """Very light cleaning: drop multiple spaces and non-printables."""
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_into_chunks(text: str,
                      chunk_size: int = 350,
                      overlap: float = 0.2) -> List[str]:
    """
    Very simple token-counted splitter.
    *chunk_size* is measured in (rough) whitespace-separated tokens.
    """
    tokens = text.split()
    if not tokens:
        return []

    step = int(chunk_size * (1 - overlap))
    chunks = []
    for start in range(0, len(tokens), step):
        chunk_tokens = tokens[start:start + chunk_size]
        if len(chunk_tokens) < 0.5 * chunk_size:
            break
        chunks.append(" ".join(chunk_tokens))
    return chunks


def make_doc_id(pdf_path: str) -> str:
    """Generate a stable doc_id from the filename (strip extension)."""
    return pathlib.Path(pdf_path).stem


def write_chunks_jsonl(doc_id: str, chunks: List[Tuple[str, int, int]], out_dir: str):
    """
    chunks: List[(text, page_idx, pos_idx)]
    Writes one JSONL per document.
    """
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{doc_id}.jsonl")
    with open(out_path, "w", encoding="utf-8") as fh:
        for text, page, pos in chunks:
            rec = {
                "chunk_id": f"{doc_id}_p{page}_c{pos}",
                "doc_id": doc_id,
                "text": text,
                "page": page,
                "position": pos
            }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def encode_chunks(model, chunk_texts: List[str], batch_size: int = 64):
    embeddings = model.encode(
        chunk_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True
    )
    return embeddings


def save_embeddings(doc_id: str, embeddings: np.ndarray, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f"{doc_id}.npy"), embeddings)


def process_pdf(pdf_path: str,
                chunk_size: int,
                overlap: float,
                model,
                chunks_dir: str,
                embeds_dir: str):
    doc_id = make_doc_id(pdf_path)
    print(f"→ {doc_id}")

    # 1) extract & clean
    pages = pdf_to_pages(pdf_path)
    all_chunks: List[Tuple[str, int, int]] = []
    for p_idx, page in enumerate(pages):
        page = clean_text(page)
        for pos_idx, chunk in enumerate(split_into_chunks(page, chunk_size, overlap)):
            all_chunks.append((chunk, p_idx, pos_idx))

    if not all_chunks:
        print(f"   (skip – no text)")
        return

    # 2) write JSONL
    write_chunks_jsonl(doc_id, all_chunks, chunks_dir)

    # 3) encode
    chunk_texts = [t for t, _, _ in all_chunks]
    emb = encode_chunks(model, chunk_texts)
    save_embeddings(doc_id, emb, embeds_dir)


def iter_pdfs(pdf_root: str):
    """yield all *.pdf under pdf_root (recursively)"""
    for p in glob.iglob(os.path.join(pdf_root, "**", "*.pdf"), recursive=True):
        yield p


def main():
    import sys
    from sentence_transformers import models, SentenceTransformer

    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf_root", required=True, help="Root folder of PDFs")
    ap.add_argument("--model", required=True,
                    help="Path or HF ID for BERT model (e.g., batterydata/batterybert-cased)")
    ap.add_argument("--chunk_size", type=int, default=350)
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--chunks_dir", default="data/processed/chunks")
    ap.add_argument("--embeds_dir", default="data/processed/embeddings")
    args = ap.parse_args()

    # Build a SentenceTransformer model with pooling
    bert_model_path = args.model  # e.g. "models/batterybert-cased" or "batterydata/batterybert-cased"

    try:
        word_embedding_model = models.Transformer(bert_model_path, max_seq_length=512)
        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=True,
            pooling_mode_cls_token=False,
            pooling_mode_max_tokens=False,
        )
        batterybert_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    except Exception as e:
        print(f"✘ Failed to load or wrap model '{bert_model_path}': {e}", file=sys.stderr)
        sys.exit(1)

    # Process each PDF file
    for pdf in iter_pdfs(args.pdf_root):
        try:
            process_pdf(
                pdf_path=pdf,
                chunk_size=args.chunk_size,
                overlap=args.overlap,
                model=batterybert_model,
                chunks_dir=args.chunks_dir,
                embeds_dir=args.embeds_dir
            )
        except Exception as e:
            print(f"!! Failed on {pdf}: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
