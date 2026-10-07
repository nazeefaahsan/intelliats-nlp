"""Safe PDF text extraction with clear error states."""
import pymupdf as fitz
class PDFError(ValueError): pass
def extract_text_from_pdf(file):
    try:
        raw = file.read() if hasattr(file, "read") else file
        if not raw: raise PDFError("The uploaded PDF is empty.")
        doc = fitz.open(stream=raw, filetype="pdf")
        pages = [page.get_text("text") or "" for page in doc]
        doc.close()
    except PDFError: raise
    except Exception as exc: raise PDFError("This file is not a valid or readable PDF.") from exc
    text = "\n\n".join(p.strip() for p in pages if p.strip())
    if not text.strip(): raise PDFError("This PDF appears to contain scanned images rather than selectable text. Please upload a text-based PDF.")
    return text
