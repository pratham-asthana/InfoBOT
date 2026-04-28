import os
import tempfile
from typing import List, Dict

import google.generativeai as genai
import streamlit as st
from dotenv import load_dotenv

from ingestion_pipeline import ingest_documents
from retrieval_pipeline import retrieve_top_k


PERSIST_DIR = os.path.join("data", "chroma")
MAX_CONTEXT_CHUNKS = 5


def load_api_key() -> str:
    load_dotenv()
    return os.getenv("GEMINI_API_KEY", "")


def ensure_dirs() -> str:
    os.makedirs(PERSIST_DIR, exist_ok=True)
    uploads_dir = os.path.join("data", "uploads")
    os.makedirs(uploads_dir, exist_ok=True)
    return uploads_dir


def save_uploads(files: List[st.runtime.uploaded_file_manager.UploadedFile], uploads_dir: str) -> List[str]:
    paths: List[str] = []
    for file in files:
        suffix = os.path.splitext(file.name)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, dir=uploads_dir, suffix=suffix) as tmp:
            tmp.write(file.getvalue())
            paths.append(tmp.name)
    return paths


def format_sources(chunks: List[Dict[str, str]]) -> str:
    if not chunks:
        return "Sources:\n- None"
    lines = ["Sources:"]
    for item in chunks:
        source = item.get("source", "unknown")
        excerpt = item.get("chunk_text", "").strip()
        if len(excerpt) > 280:
            excerpt = excerpt[:277] + "..."
        lines.append(f"- {source}: {excerpt}")
    return "\n".join(lines)


def build_prompt(query: str, chunks: List[Dict[str, str]], history: List[Dict[str, str]]) -> str:
    history_lines = []
    for turn in history[-6:]:
        history_lines.append(f"{turn['role'].upper()}: {turn['content']}")

    context_blocks = []
    for item in chunks[:MAX_CONTEXT_CHUNKS]:
        context_blocks.append(f"Source: {item['source']}\n{item['chunk_text']}")

    return (
        "You are a helpful assistant. Use only the context below to answer. "
        "If the context does not contain the answer, respond exactly: "
        '"I couldn\'t find relevant information in the uploaded documents."\n\n'
        "CONTEXT:\n"
        + "\n\n".join(context_blocks)
        + "\n\nCHAT HISTORY:\n"
        + "\n".join(history_lines)
        + "\n\nQUESTION:\n"
        + query
        + "\n\nANSWER:" 
    )


def generate_answer(query: str, chunks: List[Dict[str, str]], history: List[Dict[str, str]], api_key: str) -> str:
    if not chunks:
        return "I couldn't find relevant information in the uploaded documents."

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash-lite")
    prompt = build_prompt(query, chunks, history)
    response = model.generate_content(prompt)
    text = response.text.strip() if response.text else ""

    if not text:
        return "I couldn't find relevant information in the uploaded documents."

    return text


def main() -> None:
    st.set_page_config(page_title="InfoBOT", page_icon="📄", layout="wide")
    st.title("InfoBOT - Document Grounded Chatbot")

    api_key = load_api_key()
    if not api_key:
        st.warning("Set GEMINI_API_KEY in a .env file to continue.")

    uploads_dir = ensure_dirs()

    with st.sidebar:
        st.header("Upload documents")
        files = st.file_uploader(
            "Upload PDF, DOCX, or TXT files",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
        )
        if st.button("Ingest documents"):
            if not files:
                st.warning("Please upload at least one document.")
            elif not api_key:
                st.warning("Set GEMINI_API_KEY before ingesting.")
            else:
                file_paths = save_uploads(files, uploads_dir)
                with st.spinner("Indexing documents..."):
                    total, stored = ingest_documents(
                        file_paths=file_paths,
                        api_key=api_key,
                        persist_dir=PERSIST_DIR,
                    )
                st.success(f"Processed {total} chunks. Stored {stored} new chunks.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    query = st.chat_input("Ask a question grounded in the uploaded documents")
    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                chunks = retrieve_top_k(
                    query=query,
                    api_key=api_key,
                    persist_dir=PERSIST_DIR,
                )
                answer = generate_answer(
                    query=query,
                    chunks=chunks,
                    history=st.session_state.messages,
                    api_key=api_key,
                )
                response_text = answer + "\n\n" + format_sources(chunks)
                st.markdown(response_text)

        st.session_state.messages.append({"role": "assistant", "content": response_text})


if __name__ == "__main__":
    main()
