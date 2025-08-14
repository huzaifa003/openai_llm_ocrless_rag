import base64, os, io, json
from typing import List, Dict, Any, Optional
from openai import OpenAI
from PIL import Image

OPENAI_EMBEDDING_MODEL = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
OPENAI_VISION_MODEL = os.environ.get("OPENAI_VISION_MODEL", "gpt-5-mini")
OPENAI_LLM_MODEL = os.environ.get("OPENAI_LLM_MODEL", "gpt-5-mini")

_client_singleton: Optional[OpenAI] = None

def get_client() -> OpenAI:
    global _client_singleton
    if _client_singleton is None:
        _client_singleton = OpenAI()
    return _client_singleton

def to_b64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def vision_extract(image_path: str) -> Dict[str, str]:
    """
    Ask the Vision model to extract machine-readable text and produce a concise description.
    Returns: {"extracted_text": "...", "description": "..."}
    """
    import base64, io, json, sys
    from PIL import Image

    client = get_client()

    # Load & resize (long edge ~1600px to keep tokens reasonable and help OCR)
    try:
        img = Image.open(image_path).convert("RGB")
        max_side = 1600
        w, h = img.size
        scale = min(1.0, max_side / float(max(w, h)))
        if scale < 1.0:
            img = img.resize((int(w * scale), int(h * scale)))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"[Vision] Cannot open/prepare image: {image_path} ({e})", file=sys.stderr)
        return {"extracted_text": "", "description": ""}

    prompt = (
        "Return STRICT JSON with keys exactly 'extracted_text' and 'description'. "
        "'extracted_text': all readable text (tables, labels, printed text) as UTF-8. "
        "'description': 1-3 sentences summarizing visible content. "
        "Do not add extra keys or commentary."
    )

    try:
        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_VISION_MODEL", OPENAI_VISION_MODEL),
            messages=[
                {"role": "system", "content": "You convert document images into text + a short description."},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
                ]},
            ],
            
        )
        content = (resp.choices[0].message.content or "").strip()
        # Allow fenced code blocks
        content = content.strip().strip("`")
        if content.lower().startswith("json"):
            content = content[4:].lstrip(": \n")

        data = {}
        try:
            data = json.loads(content)
        except Exception:
            # If the model failed to emit pure JSON, just use the raw text as description
            data = {"extracted_text": "", "description": content}

        return {
            "extracted_text": (data.get("extracted_text") or "").strip(),
            "description": (data.get("description") or "").strip(),
        }
    except Exception as e:
        print(f"[Vision] OpenAI call failed for {image_path}: {e}", file=sys.stderr)
        return {"extracted_text": "", "description": ""}


def embed_texts(texts: List[str]) -> List[List[float]]:
    client = get_client()
    resp = client.embeddings.create(model=OPENAI_EMBEDDING_MODEL, input=texts)
    return [d.embedding for d in resp.data]

def synthesize_answer(question: str, hits: List[dict]) -> str:
    """
    Use the LLM to answer based on retrieved 'hits' (each with text + metadata).
    """
    client = get_client()
    context = ""
    for i, h in enumerate(hits, 1):
        body = h.get("text") or h.get("extracted_text") or h.get("description") or ""
        context += f"[{i}] page={h.get('page','?')} bbox={h.get('bbox','?')} source={h.get('source','')}\n{body}\n\n"
    messages = [
        {"role": "system", "content": "Answer the user's question *only* using the provided context. If unsure, say you don't know."},
        {"role": "user", "content": f"Question:\n{question}\n\nContext:\n{context}\n\nAnswer clearly and concisely."},
    ]
    resp = client.chat.completions.create(model=OPENAI_LLM_MODEL, messages=messages)
    return resp.choices[0].message.content.strip()
