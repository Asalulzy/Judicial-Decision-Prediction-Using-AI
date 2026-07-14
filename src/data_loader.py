import os
import zipfile
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def ensure_extracted(dataset_path: str, extract_to: str = "dataset") -> str:
    """If dataset_path is a zip, extract it to extract_to and return extracted folder path."""
    dataset_path = Path(dataset_path)
    if dataset_path.suffix.lower() == ".zip":
        out = Path(extract_to)
        if not out.exists():
            logger.info("Extracting dataset zip to %s", out)
            with zipfile.ZipFile(dataset_path, "r") as z:
                z.extractall(out)
        return str(out)
    return str(dataset_path)


def load_txt_documents(root_dir: str):
    """Yield (text, metadata) for each .txt under root_dir.

    Metadata includes filename and document_id (filename without extension).
    """
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dataset folder not found: {root_dir}")

    for p in root.rglob("*.txt"):
        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = p.read_text(encoding="latin-1")
        metadata = {"filename": str(p.relative_to(root)), "document_id": p.stem}
        yield text, metadata
