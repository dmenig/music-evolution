from __future__ import annotations

import json
import os
from pathlib import Path

from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential

from .prompts import EXPAND_NODE_SYSTEM, render_expand_prompt
from .schema import GeminiGenreResponse

MODEL = "gemini-3.1-pro-preview"
RAW_DIR = Path("data/raw")


def _client() -> genai.Client:
    return genai.Client(api_key=os.environ["GOOGLE_API_KEY"])


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=2, max=60))
def _call_gemini(name: str) -> GeminiGenreResponse:
    client = _client()
    resp = client.models.generate_content(
        model=MODEL,
        contents=render_expand_prompt(name),
        config=types.GenerateContentConfig(
            system_instruction=EXPAND_NODE_SYSTEM,
            temperature=0.2,
            response_mime_type="application/json",
            response_schema=GeminiGenreResponse,
            thinking_config=types.ThinkingConfig(thinking_level="low"),
        ),
    )
    parsed = resp.parsed
    if parsed is None:
        parsed = GeminiGenreResponse.model_validate_json(resp.text)
    return parsed


def expand_node(canonical: str, display_name: str) -> GeminiGenreResponse:
    """Cache-aware expansion. Reads data/raw/<canonical>.json if present."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cache = RAW_DIR / f"{canonical}.json"
    if cache.exists():
        return GeminiGenreResponse.model_validate_json(cache.read_text())
    result = _call_gemini(display_name)
    cache.write_text(json.dumps(result.model_dump(), indent=2, ensure_ascii=False))
    return result
