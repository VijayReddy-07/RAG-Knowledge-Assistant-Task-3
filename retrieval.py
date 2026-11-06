
import chromadb
from sentence_transformers import SentenceTransformer

from config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    TOP_K
)


def load_vector_database():
    """Connect to the persistent Chroma database."""

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    return collection


def retrieve_chunks(question, embedding_model, collection, top_k=TOP_K):
    """Retrieve the most relevant chunks for a question."""

    query_embedding = embedding_model.encode(
        [question],
        normalize_embeddings=True
    ).tolist()

    result = collection.query(
        query_embeddings=query_embedding,
        n_results=min(top_k, collection.count()),
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    retrieved_chunks = []

    if not result["documents"] or not result["documents"][0]:
        return retrieved_chunks

    for text, metadata, distance in zip(
        result["documents"][0],
        result["metadatas"][0],
        result["distances"][0]
    ):
        retrieved_chunks.append({
            "text": text,
            "source": metadata["source"],
            "page": metadata["page"],
            "distance": float(distance)
        })

    return retrieved_chunks


def main():
    print("Loading embedding model...")

    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL,
        device="cpu"
    )

    collection = load_vector_database()

    print("Vector database connected!")
    print("Total stored chunks:", collection.count())

    while True:
        question = input("\nEnter your question (or type exit): ")

        if question.lower().strip() == "exit":
            break

        if not question.strip():
            continue

        results = retrieve_chunks(
            question,
            embedding_model,
            collection
        )

        print("\nRetrieved results:")
        print("=" * 60)

        for index, result in enumerate(results, start=1):
            print(f"\nResult {index}")
            print("Source:", result["source"])
            print("Page:", result["page"])
            print("Cosine distance:", round(result["distance"], 4))
            print("Text:", result["text"][:700])
            print("-" * 60)


if __name__ == "__main__":
    main()
