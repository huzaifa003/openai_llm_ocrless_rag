from __future__ import annotations
import os
from pathlib import Path
import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import (
    OpenAIEmbeddingFunction as ChromaOpenAIEmbeddingFunction,
)

class ChromaStore:
    def __init__(self, path: str, collection: str = "pdf_openai"):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

        # Create persistent client
        self.client = chromadb.PersistentClient(
            path=str(self.path),
            settings=Settings(allow_reset=True),
        )

        # Use Chroma's built-in OpenAI EF (implements correct protocol)
        self.ef = ChromaOpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large"),
        )

        # IMPORTANT: If you previously created this collection with a different EF,
        # you'll get a conflict. Either delete the old collection or use a new name.
        self.coll = self.client.get_or_create_collection(
            name=collection,
            embedding_function=self.ef,
        )

    # oa_rag/store.py  (inside ChromaStore)
    def upsert(self, docs: list[dict]):
        if not docs:
            print("[ChromaStore] No docs passed to upsert.")
            return

        existing = self.coll.get()
        start = len(existing.get("ids", [])) if isinstance(existing, dict) else 0

        ids, documents, metadatas = [], [], []
        kept = 0
        for d in docs:
            content = (d.get("text") or d.get("extracted_text") or d.get("description") or "").strip()

            # Keep even short strings (>=1 char); only skip truly empty
            if content == "":
                continue

            md = {
                "content_type": str(d.get("type", "")),
                "page": int(d.get("page", 0)),
                "source": str(d.get("source", "")),
                "image_path": str(d.get("image_path", "")),
            }
            bbox = d.get("bbox")
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                md["bbox_x0"], md["bbox_y0"], md["bbox_x1"], md["bbox_y1"] = map(float, bbox)
            elif bbox is not None:
                md["bbox_str"] = str(bbox)

            ids.append(str(start + len(ids)))
            documents.append(content)
            metadatas.append(md)
            kept += 1

        if kept == 0:
            print("[ChromaStore] All docs were empty after normalization; nothing to upsert.")
            return

        self.coll.upsert(ids=ids, documents=documents, metadatas=metadatas)
        print(f"[ChromaStore] Upserted: {kept} records (collection now has ~{self.coll.count()}).")



    def query(self, text: str, top_k: int = 10) -> list[dict]:
        res = self.coll.query(query_texts=[text], n_results=top_k)
        return [{"text": doc, **md} for doc, md in zip(res["documents"][0], res["metadatas"][0])]
