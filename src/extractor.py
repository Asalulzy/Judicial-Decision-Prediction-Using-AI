import json
import logging
from llm import generate

logger = logging.getLogger(__name__)


EXTRACTION_PROMPT = """
Anda adalah asisten hukum. Ekstrak informasi dari teks putusan berikut ke dalam JSON dengan field:
- case_type
- law_reference
- articles (list)
- prosecutor_demand
- judge_decision
- sentence_duration

Teks:
"""


def extract_structured(text: str, backend: str = "hf", ollama_url: str = None, hf_model: str = "google/flan-t5-small"):
    prompt = EXTRACTION_PROMPT + text + "\n\nKeluarkan hanya JSON valid."
    out = generate(prompt, backend=backend, ollama_url=ollama_url, hf_model=hf_model)
    try:
        # Try to find JSON in output
        start = out.find('{')
        if start >= 0:
            j = out[start:]
            return json.loads(j)
    except Exception:
        logger.exception("Failed to parse extraction output")
    # fallback empty structure
    return {"case_type": "", "law_reference": "", "articles": [], "prosecutor_demand": "", "judge_decision": "", "sentence_duration": ""}
