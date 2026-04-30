"""
ingest.py
---------
ONE-TIME SCRIPT — run this before your chatbot goes live.

WHAT IT DOES (step by step):
  1. Opens a PDF from your /docs folder
  2. Extracts all the text
  3. Splits text into overlapping chunks (so context isn't lost at boundaries)
  4. Embeds each chunk into a 1536-dim vector
  5. Upserts (insert or update) into Pinecone

You only re-run this when your docs change.
"""

import os
import fitz  # PyMuPDF — parses PDF files
from app.utils.vector_store import embed_text, index


# ── PDF TEXT EXTRACTION ───────────────────────────────────────────────────────
def extract_pdf_text(file_path: str) -> str:
    """
    Open a PDF and concatenate all page text into one string.
    fitz (PyMuPDF) handles multi-column, tables, headers, etc. better than pdfplumber for simple text.
    """
    doc = fitz.open(file_path)
    pages_text = []
    for page_num, page in enumerate(doc):
        text = page.get_text("text")  # "text" mode = plain text extraction
        if text.strip():  # skip blank pages
            pages_text.append(text)
    doc.close()
    return "\n".join(pages_text)


# ── CHUNKING ──────────────────────────────────────────────────────────────────
def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> list:
    """
    Split text into overlapping word-based chunks.

    WHY OVERLAP?
      If a sentence spans two chunks without overlap, the model might miss context.
      Overlap = last 50 words of chunk N are also the first 50 words of chunk N+1.

    chunk_size=400 words  → ~600 tokens, well within GPT-4's context limits
    overlap=50 words      → smooth context continuity across chunk boundaries
    """
    words = text.split()
    chunks = []
    step = chunk_size - overlap  # how far we advance each iteration

    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)

    return chunks


# ── INGEST ONE FILE ───────────────────────────────────────────────────────────
def ingest_file(file_path: str) -> int:
    """
    Full pipeline for a single PDF file.
    Returns the number of chunks ingested.
    """
    print(f"\n📄 Ingesting: {file_path}")

    # Step 1: Extract text
    raw_text = extract_pdf_text(file_path)
    if not raw_text.strip():
        print(f"  ⚠️  No text found in {file_path}, skipping.")
        return 0

    # Step 2: Chunk text
    chunks = chunk_text(raw_text)
    print(f"  ✂️  Split into {len(chunks)} chunks")

    # Step 3: Embed + prepare vectors
    vectors = []
    base_name = os.path.splitext(os.path.basename(file_path))[0]  # filename without .pdf

    for i, chunk in enumerate(chunks):
        try:
            embedding = embed_text(chunk)
            vectors.append({
                "id": f"{base_name}_chunk_{i}",         # unique ID per chunk
                "values": embedding,                     # 1536-dim vector
                "metadata": {
                    "text": chunk,                       # original text (retrieved at query time)
                    "source": os.path.basename(file_path),
                    "chunk_index": i,
                },
            })
        except Exception as e:
            print(f"  ❌ Error embedding chunk {i}: {e}")

    # Step 4: Upsert to Pinecone in batches of 100 (Pinecone limit)
    batch_size = 100
    for batch_start in range(0, len(vectors), batch_size):
        batch = vectors[batch_start : batch_start + batch_size]
        index.upsert(vectors=batch)
        print(f"  ✅ Upserted batch {batch_start // batch_size + 1} ({len(batch)} vectors)")

    print(f"  🎉 Done! {len(vectors)} chunks stored for '{base_name}'")
    return len(vectors)


# ── MAIN: INGEST ALL PDFs IN /docs ────────────────────────────────────────────
if __name__ == "__main__":
    docs_folder = "docs"

    if not os.path.exists(docs_folder):
        print(f"❌ '{docs_folder}' folder not found. Create it and add your PDFs.")
        exit(1)

    pdf_files = [f for f in os.listdir(docs_folder) if f.endswith(".pdf")]

    if not pdf_files:
        print(f"❌ No PDF files found in '{docs_folder}'. Add your product docs there.")
        exit(1)

    print(f"📚 Found {len(pdf_files)} PDF(s): {pdf_files}")
    total = 0

    for filename in pdf_files:
        file_path = os.path.join(docs_folder, filename)
        total += ingest_file(file_path)

    print(f"\n✨ Ingestion complete! Total chunks in Pinecone: {total}")