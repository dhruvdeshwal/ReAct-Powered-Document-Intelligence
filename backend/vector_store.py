"""
vector_store.py
----------------
Wraps ChromaDB + HuggingFace Inference Router embeddings.

Why a custom embedding function:
- ChromaDB's built-in HuggingFaceEmbeddingFunction calls the legacy
  api-inference.huggingface.co endpoint, which is unreachable in some
  environments (DNS failures).
- This custom function calls HuggingFace's newer router endpoint:
  https://router.huggingface.co/hf-inference/models/<model>/pipeline/feature-extraction
- Embeddings are computed remotely, so the deployed app doesn't need to
  load PyTorch/sentence-transformers locally (avoids OOM on Render free tier).

Responsibilities:
- Initialize a persistent ChromaDB collection
- Embed and store document chunks with metadata (via HF router API)
- Retrieve top-k similar chunks for a given query
- List indexed documents
"""

import os
import requests
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CHROMA_DIR = os.getenv(
    "CHROMA_DIR",
    os.path.join(BASE_DIR, "chroma_db")
)

# HuggingFace Inference Router config
HF_API_KEY = os.getenv("HF_API_KEY")  # get free token from https://huggingface.co/settings/tokens
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

COLLECTION_NAME = "documents"
TOP_K = int(os.getenv("TOP_K", 4))


class HFRouterEmbeddingFunction(EmbeddingFunction):
    """
    Custom ChromaDB-compatible embedding function that calls HuggingFace's
    router endpoint for feature-extraction (embeddings), computed remotely.
    """

    def __init__(self, api_key: str, model_name: str = EMBEDDING_MODEL):
        if not api_key:
            raise RuntimeError(
                "HF_API_KEY not set. Get a free token from "
                "https://huggingface.co/settings/tokens and add it to your .env"
            )
        self.api_key = api_key
        self.model_name = model_name
        self.url = f"https://router.huggingface.co/hf-inference/models/{model_name}/pipeline/feature-extraction"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def __call__(self, input: Documents) -> Embeddings:
        # Chroma may call this with a single string or a list of strings
        texts = input if isinstance(input, list) else [input]

        response = requests.post(
            self.url,
            headers=self.headers,
            json={"inputs": texts, "options": {"wait_for_model": True}},
            timeout=60,
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"HuggingFace embedding request failed "
                f"({response.status_code}): {response.text}"
            )

        embeddings = response.json()

        # feature-extraction can return per-token embeddings (3D) for some models;
        # mean-pool over tokens if so, to get a single vector per input.
        result = []
        for emb in embeddings:
            if isinstance(emb[0], list):  # 2D -> token-level, needs pooling
                num_tokens = len(emb)
                dim = len(emb[0])
                pooled = [sum(token[i] for token in emb) / num_tokens for i in range(dim)]
                result.append(pooled)
            else:  # already a flat vector
                result.append(emb)

        return result


class VectorStore:
    def __init__(self, persist_dir: str = CHROMA_DIR, model_name: str = EMBEDDING_MODEL):
        self.client = chromadb.PersistentClient(path=persist_dir)

        # Custom embedding function using HF router endpoint (remote, low RAM)
        self.embedding_fn = HFRouterEmbeddingFunction(
            api_key=HF_API_KEY,
            model_name=model_name,
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

        # HF Inference API has request size limits - batch in chunks of 32
        batch_size = 32
        total_added = 0
        for i in range(0, len(documents), batch_size):
            self.collection.upsert(
                documents=documents[i:i + batch_size],
                metadatas=metadatas[i:i + batch_size],
                ids=ids[i:i + batch_size],
            )
            total_added += len(documents[i:i + batch_size])

        return total_added

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