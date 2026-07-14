import re


def clean_text(text: str) -> str:
    """Basic cleaning: normalize whitespace, remove control chars, preserve legal markers."""
    if not text:
        return ""
    # Remove null bytes and other control characters except newline
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]+", " ", text)
    # Normalize whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200):
    """Split text into overlapping character-based chunks."""
    if chunk_size <= 0:
        yield text
        return
    start = 0
    L = len(text)
    while start < L:
        end = min(start + chunk_size, L)
        yield text[start:end]
        if end == L:
            break
        start = end - overlap
