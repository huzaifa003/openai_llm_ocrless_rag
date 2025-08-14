# OpenAI Vision + Embeddings RAG for PDFs (No OCR deps)

**Simple & effective** pipeline that avoids traditional OCR:
1) Extract **text blocks** and **images** from PDFs (PyMuPDF).
2) For each image, send it to **OpenAI Vision (gpt-4o)** to:
   - extract machine-readable text (JSON) and
   - produce a concise description.
3) Embed all text (PDF blocks + image text/description) with **OpenAI `text-embedding-3-large`**.
4) Store in **ChromaDB**.
5) Query with text; optionally synthesize an answer with **OpenAI (gpt-4o)** using retrieved chunks.

## Setup
- Set `OPENAI_API_KEY` in your environment.
- Install requirements and run ingest/query:

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt

# Ingest PDFs from ./docs into ./vector_store
python examples/ingest.py --pdfs ./docs --store ./vector_store

# Query the store (retrieval only)
python examples/query.py --store ./vector_store --query "warranty terms and conditions"

# Query + answer synthesis with OpenAI
python examples/query.py --store ./vector_store --query "what is the warranty coverage?" --answer
```
