from pathlib import Path
import fitz  # PyMuPDF
from PIL import Image
import io

def extract_pdf(pdf_path: str, out_dir: str, max_pages: int | None = None) -> list[dict]:
    """
    Returns flat list of records:
      {"type":"text","page":int,"bbox":[x0,y0,x1,y1],"text":str}
      {"type":"image","page":int,"bbox":[x0,y0,x1,y1],"image_path":str}
    """
    out_dir = Path(out_dir)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    records: list[dict] = []

    for page_idx in range(len(doc)):
        if max_pages is not None and page_idx >= max_pages:
            break
        page = doc[page_idx]

        # Text blocks
        for block in page.get_text("blocks"):
            x0, y0, x1, y1, text, *_ = block
            text = (text or "").strip()
            if not text:
                continue
            records.append({
                "type": "text",
                "page": page_idx + 1,
                "bbox": [float(x0), float(y0), float(x1), float(y1)],
                "text": text,
                "source": str(pdf_path),
            })

        # Images (extract + save) — robust across masks/alpha/indexed/jp2
         # Render full page as PNG for Vision OCR (helps when text blocks are empty)
        page_img_name = f"{Path(pdf_path).stem}-page-{page_idx+1}.png"
        page_img_path = img_dir / page_img_name
        try:
            # DPI 200 is a good balance for OCR
            pix = page.get_pixmap(dpi=200, alpha=False)
            pix.save(str(page_img_path))
            records.append({
                "type": "page_image",
                "page": page_idx + 1,
                "bbox": [0.0, 0.0, float(page.rect.width), float(page.rect.height)],
                "image_path": str(page_img_path),
                "source": str(pdf_path),
            })
        except Exception:
            pass
    

    for ii, info in enumerate(page.get_image_info(xrefs=True)):
        bbox = info.get("bbox")
        xref = info.get("xref")
        if not bbox or not xref:
            continue

        img_name = f"{Path(pdf_path).stem}-p{page_idx+1}-{ii}.png"
        img_path = img_dir / img_name

        img = None

        # 1) Safest: extract original bytes if available
        try:
            img_dict = doc.extract_image(xref)
            img_bytes = img_dict.get("image")
            if img_bytes:
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except Exception:
            img = None

        # 2) Fallback: render pixmap → PNG bytes
        if img is None:
            try:
                pix = fitz.Pixmap(doc, xref)
                # Convert to RGB if CMYK/Indexed/Alpha
                if pix.n > 4 or pix.alpha:  # CMYK or has alpha
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                png_bytes = pix.tobytes("png")
                img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
            except Exception:
                img = None

        # If still not decodable, skip this image
        if img is None:
            continue

        img.save(img_path)

        records.append({
            "type": "image",
            "page": page_idx + 1,
            "bbox": [float(b) for b in bbox],
            "image_path": str(img_path),
            "source": str(pdf_path),
        })
        
        
   


    return records
