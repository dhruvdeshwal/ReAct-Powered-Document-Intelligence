"""
vector_store.py
----------------
Wraps ChromaDB + HuggingFace (sentence-transformers) embeddings.

Responsibilities:
- Initialize a persistent ChromaDB collection
- Embed and store document chunks with metadata
- Retrieve top-k similar chunks for a given query
- List indexed documents
"""

import os
import chromadb
from chromadb.utils import embedding_functions

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CHROMA_DIR = os.getenv(
    "CHROMA_DIR",
    os.path.join(BASE_DIR, "chroma_db")
)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
COLLECTION_NAME = "documents"
TOP_K = int(os.getenv("TOP_K", 4))


class VectorStore:
    def __init__(self, persist_dir: str = CHROMA_DIR, model_name: str = EMBEDDING_MODEL):
        self.client = chromadb.PersistentClient(path=persist_dir)

        # HuggingFace sentence-transformers embedding function (runs locally, free)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model_name
        )

        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.embedding_fn,
        )

    def add_chunks(self, chunks: list[dict]) -> int:
        """
        Add a list of chunks (from document_processor.chunk_document) to the vector store.

        Each chunk: {"text": str, "metadata": {"source": str, "page": int|None, "chunk_index": int}}
        Returns number of chunks added.
        """
        if not chunks:
            return 0

        documents, metadatas, ids = [], [], []
        for chunk in chunks:
            source = chunk["metadata"]["source"]
            idx = chunk["metadata"]["chunk_index"]
            chunk_id = f"{source}_{idx}"

            # Chroma metadata values cannot be None
            metadata = {
                "source": source,
                "page": chunk["metadata"]["page"] if chunk["metadata"]["page"] is not None else -1,
                "chunk_index": idx,
            }

            documents.append(chunk["text"])
            metadatas.append(metadata)
            ids.append(chunk_id)

        self.collection.upsert(documents=documents, metadatas=metadatas, ids=ids)
        return len(documents)

    def search(self, query: str, top_k: int = TOP_K, source_filter: str | None = None) -> list[dict]:
        """
        Retrieve the top_k most relevant chunks for a query.

        Returns a list of {"text": str, "metadata": dict, "distance": float}
        """
        where = {"source": source_filter} if source_filter else None

        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            where=where,
        )

        output = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        for doc, meta, dist in zip(docs, metas, dists):
            output.append({"text": doc, "metadata": meta, "distance": dist})

        return output

    def list_documents(self) -> list[str]:
        """Return a sorted list of unique source filenames currently indexed."""
        data = self.collection.get(include=["metadatas"])
        sources = {m["source"] for m in data.get("metadatas", []) if m and "source" in m}
        return sorted(sources)

    def delete_document(self, source: str) -> None:
        """Remove all chunks belonging to a given source file."""
        self.collection.delete(where={"source": source})

    def count(self) -> int:
        """Total number of chunks indexed."""
        return self.collection.count()


if __name__ == "__main__":
    # Quick standalone test
    import sys
    sys.path.append(os.path.dirname(__file__))
    from document_processor import chunk_document

    vs = VectorStore()

    test_file = sys.argv[1] if len(sys.argv) > 1 else "data/uploads/sample.txt"
    chunks = chunk_document(test_file)
    added = vs.add_chunks(chunks)
    print(f"Added {added} chunks. Total in store: {vs.count()}")

    print("\nIndexed documents:", vs.list_documents())

    print("\nSearch test for 'what is football':")
    for r in vs.search("what is football", top_k=2):
        print("---")
        print("Source:", r["metadata"]["source"], "| Distance:", round(r["distance"], 4))
        print(r["text"][:150])