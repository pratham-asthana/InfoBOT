from typing import List, Dict

import chromadb
import google.generativeai as genai


def configure_gemini(api_key: str) -> None:
    if not api_key:
        raise ValueError("GEMINI_API_KEY is required")
    genai.configure(api_key=api_key)


def get_chroma_collection(persist_dir: str) -> chromadb.api.models.Collection.Collection:
    client = chromadb.PersistentClient(path=persist_dir)
    return client.get_or_create_collection(
        name="documents",
        metadata={"hnsw:space": "cosine"},
    )


def embed_query(query: str) -> List[float]:
    result = genai.embed_content(
        model="gemini-embedding-001",
        content=query,
        task_type="retrieval_query",
    )
    return result["embedding"]


def retrieve_top_k(
    query: str,
    api_key: str,
    persist_dir: str,
    top_k: int = 5,
) -> List[Dict[str, str]]:
    configure_gemini(api_key)
    collection = get_chroma_collection(persist_dir)
    query_embedding = embed_query(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    retrieved: List[Dict[str, str]] = []
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for doc, meta, dist in zip(documents, metadatas, distances):
        if not meta:
            continue
        retrieved.append(
            {
                "source": meta.get("source", "unknown"),
                "chunk_text": meta.get("chunk_text", doc),
                "distance": dist,
            }
        )

    return retrieved
