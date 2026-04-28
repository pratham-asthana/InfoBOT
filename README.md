# InfoBOT
Production-ready RAG chatbot grounded strictly in uploaded documents using Gemini APIs and ChromaDB.

## Features
- Upload PDF, DOCX, and TXT files
- Chunking at 800 tokens with 100 overlap using Gemini token counting
- Gemini embeddings (`models/embedding-001`) stored in ChromaDB
- Retrieval top-5 with cosine similarity
- Strict grounding and source-backed responses
- Streamlit UI with chat history

## Setup
1. Create and activate a virtual environment (recommended).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file:

```bash
copy .env.example .env
```

4. Add your API key to `.env`:

```text
GEMINI_API_KEY=your_api_key_here
```

## Run
```bash
streamlit run app.py
```

## Notes
- ChromaDB persists locally to `./data/chroma`.
- If Gemini token counting is unavailable, chunking falls back to a word-based approximation.
