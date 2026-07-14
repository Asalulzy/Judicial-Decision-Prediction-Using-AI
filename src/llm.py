import os
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def generate_with_ollama(prompt: str, ollama_url: str, model: Optional[str] = None) -> str:
    import requests

    payload = {"prompt": prompt}
    if model:
        payload["model"] = model
    try:
        url = ollama_url.rstrip("/") + "/api/generate"
        r = requests.post(url, json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        # Ollama responses vary; try to pull text
        if isinstance(data, dict) and "text" in data:
            return data["text"]
        if isinstance(data, list):
            return "\n".join([str(x) for x in data])
        return str(data)
    except Exception as e:
        logger.exception("Ollama call failed: %s", e)
        raise


def generate_with_hf(prompt: str, model_name: str = "google/flan-t5-small") -> str:
    from transformers import pipeline

    logger.info("Using HF model %s", model_name)
    pipe = pipeline("text2text-generation", model=model_name)
    out = pipe(prompt, max_length=512, truncation=True)
    if isinstance(out, list) and out:
        return out[0].get("generated_text") or out[0].get("summary_text") or str(out[0])
    return str(out)


def generate(prompt: str, backend: str = "hf", ollama_url: str = None, hf_model: str = "google/flan-t5-small") -> str:
    try:
        if backend == "ollama" and ollama_url:
            return generate_with_ollama(prompt, ollama_url)
        return generate_with_hf(prompt, hf_model)
    except Exception as e:
        logger.exception("LLM generation failed: %s", e)
        return f"[LLM error] {e}"
