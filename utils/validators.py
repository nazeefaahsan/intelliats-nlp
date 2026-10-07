"""Upload validation helpers."""
def is_pdf(filename, content_type):
    return bool(filename and filename.lower().endswith(".pdf") and content_type in ("application/pdf", "application/octet-stream", ""))
