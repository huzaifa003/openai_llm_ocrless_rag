import argparse, os
from pathlib import Path
from tqdm import tqdm
from oa_rag.extract import extract_pdf
from oa_rag.openai_helpers import vision_extract
from oa_rag.store import ChromaStore

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdfs", type=str, required=True, help="Folder with PDF files")
    ap.add_argument("--store", type=str, required=True, help="Chroma persistent dir")
    ap.add_argument("--max_pages", type=int, default=None)
    args = ap.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY env var is required.")

    store = ChromaStore(path=args.store)

    for pdf in sorted(Path(args.pdfs).glob("*.pdf")):
        print(f"[PDF] {pdf.name}")
        records = extract_pdf(str(pdf), out_dir=args.store, max_pages=args.max_pages)

        docs = []
        for r in tqdm(records, desc="Processing records"):
            if r["type"] in ("image", "page_image"):
                info = vision_extract(r["image_path"])
                r = {**r, **info}
            docs.append(r)


        store.upsert(docs)

    print(f"✓ Ingested into {args.store}")

if __name__ == "__main__":
    main()
