
from pathlib import Path
from uuid import uuid5, NAMESPACE_URL

import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    DATA_DIR,
    CHROMA_DIR,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    COLLECTION_NAME,
)


def load_documents():
    """Extract text and metadata from PDF files."""

    documents = []

    pdf_files = list(DATA_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in {DATA_DIR}. Add PDFs first."
        )

    total_pages = 0

    for pdf_path in pdf_files:
        print(f"\nReading: {pdf_path.name}")

        reader = PdfReader(str(pdf_path))

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""

            if text.strip():
                documents.append({
                    "text": text.strip(),
                    "source": pdf_path.name,
                    "page": page_number
                })

                total_pages += 1

    print(f"\nPDF files found: {len(pdf_files)}")
    print(f"Pages containing extractable text: {total_pages}")

    if total_pages < 10:
        print(
            "WARNING: Your dataset contains fewer than "
            "10 pages with extractable text."
        )

    if not documents:
        raise ValueError(
            "No readable text found. Your PDFs may contain scanned images. "
            "Use OCR to extract text from scanned PDFs."
        )

    return documents


def split_documents(documents):
    """Split document pages into overlapping chunks."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = []

    for document in documents:
        text_chunks = splitter.split_text(document["text"])

        for chunk_index, chunk_text in enumerate(text_chunks):
            chunks.append({
                "text": chunk_text,
                "source": document["source"],
                "page": document["page"],
                "chunk": chunk_index
            })

    return chunks


def store_in_chroma(chunks):
    """Generate embeddings and store chunks in ChromaDB."""

    print("\nLoading embedding model...")

    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL,
        device="cpu"
    )

    texts = [chunk["text"] for chunk in chunks]

    print(f"Generating embeddings for {len(texts)} chunks...")

    embeddings = embedding_model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True
    ).tolist()

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    # Rebuild the collection to avoid duplicate chunks
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    ids = [
        str(uuid5(
            NAMESPACE_URL,
            f"{chunk['source']}-{chunk['page']}-{chunk['chunk']}"
        ))
        for chunk in chunks
    ]

    metadatas = [
        {
            "source": chunk["source"],
            "page": int(chunk["page"]),
            "chunk": int(chunk["chunk"])
        }
        for chunk in chunks
    ]

    # Add in batches to avoid very large requests
    batch_size = 100

    for start in range(0, len(chunks), batch_size):
        end = start + batch_size

        collection.add(
            ids=ids[start:end],
            documents=texts[start:end],
            embeddings=embeddings[start:end],
            metadatas=metadatas[start:end]
        )

    print("\nIngestion completed successfully!")
    print(f"Total chunks stored: {collection.count()}")
    print(f"Database location: {CHROMA_DIR}")


def main():
    documents = load_documents()

    chunks = split_documents(documents)

    print(f"\nTotal chunks generated: {len(chunks)}")

    print("\nSample chunk for inspection:")
    print("-" * 60)
    print(chunks[0]["text"][:1000])
    print("-" * 60)
    print("Source:", chunks[0]["source"])
    print("Page:", chunks[0]["page"])

    store_in_chroma(chunks)


if __name__ == "__main__":
    main()
