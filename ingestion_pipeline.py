import hashlib
import os
import re
from typing import Iterable, List, Tuple

import chromadb
import google.generativeai as genai
from pypdf import PdfReader
from docx import Document


def configure_gemini(api_key: str) -> None:
    if not api_key:
        raise ValueError("GEMINI_API_KEY is required")
    genai.configure(api_key=api_key)


def _clean_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _read_pdf(path: str) -> str:
    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def _read_docx(path: str) -> str:
    doc = Document(path)
    paragraphs = [p.text for p in doc.paragraphs]
    return "\n".join(paragraphs)


def _read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def load_document(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return _read_pdf(path)
    if ext == ".docx":
        return _read_docx(path)
    if ext == ".txt":
        return _read_txt(path)
    raise ValueError(f"Unsupported file format: {ext}")


def _file_sha256(path: str) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _count_tokens(model: genai.GenerativeModel, text: str) -> int:
    try:
        return model.count_tokens(text).total_tokens
    except Exception:
        return max(1, len(text.split()))


def chunk_text(text: str, model: genai.GenerativeModel, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    words = text.split()
    if not words:
        return []

    chunks: List[str] = []
    start = 0
    while start < len(words):
        low = start + 1
        high = len(words)
        best_end = low

        while low <= high:
            mid = (low + high) // 2
            candidate = " ".join(words[start:mid])
            tokens = _count_tokens(model, candidate)
            if tokens <= chunk_size:
                best_end = mid
                low = mid + 1
            else:
                high = mid - 1

        chunk = " ".join(words[start:best_end])
        if chunk:
            chunks.append(chunk)

        if best_end == len(words):
            break

        next_start = max(best_end - overlap, start + 1)
        start = next_start

    return chunks


def _embed_batch(texts: List[str], task_type: str) -> List[List[float]]:
    try:
        result = genai.embed_content(
            model="gemini-embedding-001",
            content=texts,
            task_type=task_type,
        )
        return result["embedding"]
    except Exception:
        embeddings = []
        for text in texts:
            result = genai.embed_content(
                model="gemini-embedding-001",
                content=text,
                task_type=task_type,
            )
            embeddings.append(result["embedding"])
        return embeddings


def _batched(items: List[str], batch_size: int) -> Iterable[List[str]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def get_chroma_collection(persist_dir: str) -> chromadb.api.models.Collection.Collection:
    client = chromadb.PersistentClient(path=persist_dir)
    return client.get_or_create_collection(
        name="documents",
        metadata={"hnsw:space": "cosine"},
    )


def ingest_documents(
    file_paths: List[str],
    api_key: str,
    persist_dir: str,
    batch_size: int = 25,
) -> Tuple[int, int]:
    configure_gemini(api_key)
    model = genai.GenerativeModel("gemini-2.5-flash-lite")
    collection = get_chroma_collection(persist_dir)

    total_chunks = 0
    stored_chunks = 0

    for path in file_paths:
        source_name = os.path.basename(path)
        source_hash = _file_sha256(path)
        existing = collection.get(where={"source_hash": source_hash})
        if existing and existing.get("ids"):
            continue

        raw_text = load_document(path)
        cleaned = _clean_text(raw_text)
        chunks = chunk_text(cleaned, model=model, chunk_size=800, overlap=100)

        total_chunks += len(chunks)
        if not chunks:
            continue

        ids: List[str] = []
        metadatas: List[dict] = []
        documents: List[str] = []
        embeddings: List[List[float]] = []

        for batch_index, batch in enumerate(_batched(chunks, batch_size)):
            batch_embeddings = _embed_batch(batch, task_type="retrieval_document")

            for i, (chunk, embedding) in enumerate(zip(batch, batch_embeddings)):
                chunk_index = batch_index * batch_size + i
                chunk_id = f"{source_hash}-{chunk_index}"
                ids.append(chunk_id)
                documents.append(chunk)
                metadatas.append(
                    {
                        "source": source_name,
                        "source_hash": source_hash,
                        "chunk_index": chunk_index,
                        "chunk_text": chunk,
                    }
                )
                embeddings.append(embedding)

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        stored_chunks += len(ids)

    return total_chunks, stored_chunks
