import argparse, os
from rich.console import Console
from rich.table import Table
from oa_rag.store import ChromaStore
from oa_rag.openai_helpers import synthesize_answer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", type=str, required=True)
    ap.add_argument("--query", type=str, required=True)
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--answer", action="store_true", help="Also synthesize an answer with OpenAI (uses OPENAI_API_KEY)")
    args = ap.parse_args()
    print(f"OPENAI_API_KEY={os.getenv('OPENAI_API_KEY')}")

    store = ChromaStore(path=args.store)
    hits = store.query(args.query, top_k=args.top_k)

    console = Console()
    t = Table("type", "page", "bbox", "text / extracted (trunc)", "image_path", "source")
    for h in hits:
        txt = h.get("text") or ""
        if len(txt) > 180:
            txt = txt[:180] + "…"
        t.add_row(h.get("content_type",""), str(h.get("page","")), str(h.get("bbox","")), txt, h.get("image_path",""), h.get("source",""))
    console.print(t)

    if args.answer:
        if not os.getenv("OPENAI_API_KEY"):
            console.print("[red]OPENAI_API_KEY is required for answer synthesis.[/red]")
        else:
            
            answer = synthesize_answer(args.query, hits)
            console.rule("[bold]Answer")
            console.print(answer)

if __name__ == "__main__":
    main()
