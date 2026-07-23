"""
One-time setup script: embeds every chunk in knowledge_base.py and
upserts it into your Pinecone index/namespace.

Run this once before starting app.py for the first time (or whenever
you edit knowledge_base.py):

    python upload_kb.py
"""
import time

from config import PINECONE_NAMESPACE
from rag import embed_text, index
from knowledge_base import CIS_KNOWLEDGE_BASE


def upload_knowledge_base(chunks: list[dict], namespace: str, batch_size: int = 10):
    vectors = []
    for i, chunk in enumerate(chunks):
        print(f"  Embedding chunk {i + 1}/{len(chunks)}: {chunk['id']}")
        vector = embed_text(chunk["text"])
        vectors.append(
            {"id": chunk["id"], "values": vector, "metadata": {"text": chunk["text"]}}
        )

        if len(vectors) == batch_size or i == len(chunks) - 1:
            index.upsert(vectors=vectors, namespace=namespace)
            print(f"  Upserted batch of {len(vectors)} chunks.")
            vectors = []
            time.sleep(1)

    print(f"\nAll {len(chunks)} chunks uploaded to namespace '{namespace}'.")


if __name__ == "__main__":
    print("Uploading CIS department knowledge base to Pinecone...")
    upload_knowledge_base(CIS_KNOWLEDGE_BASE, namespace=PINECONE_NAMESPACE)
