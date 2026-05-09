from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

MODEL = "gemini-3.1-pro-preview"
CACHE_DIR = Path("data/rescore")
BATCH = 30

SYSTEM = (
    "You are a precise musicologist. You return floats grounded in historical "
    "demographics, not vibes. Use decimals freely. Do not round to integers."
)

INSTRUCTIONS = """For each genre below, estimate its NEW popularity score.

DEFINITION
popularity = the percentage of ALL music-listening humans alive ANYWHERE IN THE WORLD at the genre's peak_year who had this specific genre as a meaningful part of their listening or cultural identity. Float in [0.0, 100.0].

The denominator is the entire global music-listening population at peak_year — NOT the region, NOT the country, NOT the cultural sphere. A genre dominant in Europe but absent in Asia/Africa/Americas has a small world-share even if its regional share is high. A genre confined to one country has world-share ≈ (that country's listeners / world's listeners).

Overlap allowed: a person who listened to both Rock and Pop counts toward both. Use decimals freely (0.0001, 0.003, 2.5, 17.3, etc.). Be as specific and accurate as you possibly can. Do not return identical numbers for clearly different genres.

REASONING
world population at peak_year × ~all-listen-to-music × (genre's geographic reach / world) × (within-region adoption). 1400 ≈ 400M humans → even a Christendom-wide tradition tops ~15-18% world-share. 2000 ≈ 6B humans, all listen to music → a planet-wide hit can reach 30-40%.

ANCHORS (calibrated to the WORLD denominator — use as calibration, not bins)
- Ancient Indigenous Music (peak ~1400) = 8-15  (one of many parallel traditions)
- Religious Chant (medieval peak)        = 10-18 (Christendom + parts of Islamic/Buddhist)
- Folk Music (Western, peak ~1965)       = 2-5
- Western Classical (peak ~1880)         = 4-8
- Rock (peak ~1975)                      = 22-30
- Pop (peak ~2000)                       = 28-38
- Hip-Hop (peak ~2015)                   = 18-26
- Jazz (peak ~1950)                      = 6-10
- Bebop (peak ~1955)                     = 0.5-1.5
- Disco (peak ~1978)                     = 6-12
- Reggaeton (peak ~2020)                 = 4-8
- Vaporwave (peak ~2015)                 = 0.02-0.1
- Witch House (peak ~2012)               = 0.005-0.02
- Single-scene microgenre / one club     = 0.0001-0.001

Critical: a regional tradition that "everyone in its region listened to" is still a small fraction of world music listeners if that region was a small slice of humanity. Conversely, a 20th-century mainstream genre can score higher than any pre-modern tradition because radio + recordings let it cross continents. Do not inflate ancient/medieval scores above their realistic global share, but do not crush them to zero either.

GENRES
"""


class Score(BaseModel):
    id: str
    popularity: float = Field(ge=0.0, le=100.0)


class Batch(BaseModel):
    items: list[Score]


def _client() -> genai.Client:
    return genai.Client(api_key=os.environ["GOOGLE_API_KEY"])


def _batch_payload(nodes: list[dict]) -> str:
    lines = []
    for n in nodes:
        d = (n.get("description") or "").strip().replace("\n", " ")
        if len(d) > 220:
            d = d[:220] + "…"
        lines.append(
            f"- id: {n['id']}\n"
            f"  name: {n['name']}\n"
            f"  birth_year: {n['birth_year']}  peak_year: {n['peak_year']}  death_year: {n.get('death_year')}\n"
            f"  countries: {','.join(n.get('countries') or [])}\n"
            f"  description: {d}"
        )
    return INSTRUCTIONS + "\n" + "\n".join(lines)


def _cache_key(nodes: list[dict]) -> str:
    h = hashlib.sha256()
    for n in nodes:
        h.update(n["id"].encode())
    return h.hexdigest()[:16]


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=2, max=60))
def _call(prompt: str) -> Batch:
    client = _client()
    resp = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM,
            temperature=0.2,
            response_mime_type="application/json",
            response_schema=Batch,
            thinking_config=types.ThinkingConfig(thinking_level="low"),
        ),
    )
    parsed = resp.parsed
    if parsed is None:
        parsed = Batch.model_validate_json(resp.text)
    return parsed


def _rescore_batch(nodes: list[dict]) -> dict[str, float]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = _cache_key(nodes)
    cache_file = CACHE_DIR / f"{key}.json"
    if cache_file.exists():
        cached = json.loads(cache_file.read_text())
        return {item["id"]: float(item["popularity"]) for item in cached["items"]}
    parsed = _call(_batch_payload(nodes))
    cache_file.write_text(parsed.model_dump_json(indent=2))
    return {item.id: item.popularity for item in parsed.items}


def rescore(genres_path: Path, workers: int = 6) -> None:
    data = json.loads(genres_path.read_text())
    nodes = data["nodes"]
    by_id = {n["id"]: n for n in nodes}

    batches = [nodes[i : i + BATCH] for i in range(0, len(nodes), BATCH)]
    print(f"[rescore] {len(nodes)} nodes -> {len(batches)} batches of {BATCH}", flush=True)

    pool = ThreadPoolExecutor(max_workers=workers)
    futs = {pool.submit(_rescore_batch, b): i for i, b in enumerate(batches)}
    done_count = 0
    for fut in as_completed(futs):
        idx = futs[fut]
        try:
            result = fut.result()
        except Exception as exc:  # noqa: BLE001
            print(f"[err] batch {idx}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        for node_id, pop in result.items():
            if node_id in by_id:
                by_id[node_id]["popularity"] = pop
        done_count += 1
        if done_count % 10 == 0 or done_count == len(batches):
            print(f"[rescore] {done_count}/{len(batches)} batches", flush=True)
            genres_path.write_text(json.dumps(data, ensure_ascii=False))

    pool.shutdown(wait=True)
    genres_path.write_text(json.dumps(data, ensure_ascii=False))
    print(f"[rescore] done -> {genres_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--genres", type=Path, default=Path("genres.json"))
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    rescore(args.genres, args.workers)


if __name__ == "__main__":
    main()
