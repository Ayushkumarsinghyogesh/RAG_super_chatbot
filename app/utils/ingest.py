import os
import fitz
from app.utils.vector_store import embed_text, index


def extract_pdf_text(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text


def chunk_text(text, chunk_size=500):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)

    return chunks


def ingest_file(file_path):
    print(f"Ingesting file → {file_path}")

    raw_text = extract_pdf_text(file_path)
    chunks = chunk_text(raw_text)

    vectors = []

    for i, chunk in enumerate(chunks):
        embedding = embed_text(chunk)
        vectors.append({
            "id": f"{os.path.basename(file_path)}_chunk_{i}",
            "values": embedding,
            "metadata": {"text": chunk}
        })

    index.upsert(vectors)
    print(f"Completed ingesting → {file_path}")


if __name__ == "__main__":
    folder = "docs"

    for f in os.listdir(folder):
        if f.endswith(".pdf"):
            ingest_file(os.path.join(folder, f))
