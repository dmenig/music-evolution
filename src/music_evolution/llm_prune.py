"""LLM-driven prune pass. Asks Gemini, per node, whether it actually is a music
genre (vs. a concept, technology, instrument, person, list, era, label, etc).
Conservative bias: keep when uncertain.

Drops are cascaded through edges, same as prune.py.
"""

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
CACHE_DIR = Path("data/llm_prune")
BATCH = 60

SYSTEM = (
    "You are a precise musicologist. You decide whether each candidate is "
    "actually a music genre or style — i.e. a recognizable, scene-backed musical "
    "tradition with practitioners and a body of work. You reject non-genres: "
    "distribution methods (streaming, downloading), technologies, formats (MP3, "
    "vinyl), instruments, people, songs, albums, labels, venues, festivals, "
    "regions / 'music of X' overviews, decades / eras, scenes that lack "
    "musical-style coherence, and broad parent terms that are categories rather "
    "than genres (e.g. 'Popular music', 'Traditional music' on their own). "
    "When the candidate is genuinely a genre or style, keep it. When clearly not, "
    "reject it. When uncertain, KEEP it — false drops are worse than false keeps."
)

INSTRUCTIONS = """For each candidate below, decide is_genre (true / false).

CANDIDATES
"""


class Verdict(BaseModel):
    id: str
    is_genre: bool
    reason: str = Field(description="≤ 12 words: why kept or rejected.")


class VerdictBatch(BaseModel):
    items: list[Verdict]


def _client() -> genai.Client:
    return genai.Client(api_key=os.environ["GOOGLE_API_KEY"])


def _payload(nodes: list[dict]) -> str:
    lines: list[str] = []
    for n in nodes:
        d = (n.get("description") or "").replace("\n", " ").strip()
        if len(d) > 220:
            d = d[:220] + "…"
        lines.append(f"- id: {n['id']}\n  name: {n['name']}\n  description: {d}")
    return INSTRUCTIONS + "\n".join(lines)


def _cache_key(nodes: list[dict]) -> str:
    h = hashlib.sha256()
    for n in nodes:
        h.update(n["id"].encode())
    return h.hexdigest()[:16]


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=2, max=60))
def _call(prompt: str) -> VerdictBatch:
    client = _client()
    resp = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM,
            temperature=0.1,
            response_mime_type="application/json",
            response_schema=VerdictBatch,
            thinking_config=types.ThinkingConfig(thinking_level="low"),
        ),
    )
    parsed = resp.parsed
    if parsed is None:
        parsed = VerdictBatch.model_validate_json(resp.text)
    return parsed


def _judge_batch(nodes: list[dict]) -> dict[str, tuple[bool, str]]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = _cache_key(nodes)
    cache_file = CACHE_DIR / f"{key}.json"
    if cache_file.exists():
        cached = json.loads(cache_file.read_text())
        return {item["id"]: (item["is_genre"], item["reason"]) for item in cached["items"]}
    parsed = _call(_payload(nodes))
    cache_file.write_text(parsed.model_dump_json(indent=2))
    return {v.id: (v.is_genre, v.reason) for v in parsed.items}


def llm_prune(nodes: list[dict], workers: int = 8) -> tuple[list[dict], list[tuple[str, str, str]]]:
    """Returns (kept_nodes, dropped) where dropped is list of (id, name, reason)."""
    batches = [nodes[i : i + BATCH] for i in range(0, len(nodes), BATCH)]
    print(f"[llm_prune] {len(nodes)} nodes -> {len(batches)} batches of {BATCH}", flush=True)
    verdicts: dict[str, tuple[bool, str]] = {}
    pool = ThreadPoolExecutor(max_workers=workers)
    futs = {pool.submit(_judge_batch, b): i for i, b in enumerate(batches)}
    done = 0
    for fut in as_completed(futs):
        idx = futs[fut]
        try:
            verdicts.update(fut.result())
        except Exception as exc:  # noqa: BLE001
            print(f"[err] batch {idx}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        done += 1
        if done % 10 == 0 or done == len(batches):
            print(f"[llm_prune] {done}/{len(batches)} batches", flush=True)
    pool.shutdown(wait=True)

    bad_ids: set[str] = set()
    dropped: list[tuple[str, str, str]] = []
    for n in nodes:
        v = verdicts.get(n["id"])
        if v is None:
            continue  # uncertain (failed batch) -> keep
        is_genre, reason = v
        if not is_genre:
            bad_ids.add(n["id"])
            dropped.append((n["id"], n["name"], reason))

    surviving = [n for n in nodes if n["id"] not in bad_ids]
    for n in surviving:
        n["parent_edges"] = [e for e in n.get("parent_edges", []) if e["id"] not in bad_ids]
        n["child_edges"] = [e for e in n.get("child_edges", []) if e["id"] not in bad_ids]
        n["influence_edges"] = [e for e in n.get("influence_edges", []) if e["id"] not in bad_ids]
        n["parents"] = sorted({e["id"] for e in n["parent_edges"]})
        n["children"] = sorted({e["id"] for e in n["child_edges"]})
        n["influences"] = sorted({e["id"] for e in n["influence_edges"]})
    return surviving, dropped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="inp", type=Path, default=Path("genres.json"))
    parser.add_argument("--out", dest="outp", type=Path, default=Path("genres.json"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("logs/llm_prune_drops.json"),
        help="Write the list of dropped (id, name, reason) for review.",
    )
    args = parser.parse_args()
    payload = json.loads(args.inp.read_text())
    surviving, dropped = llm_prune(payload["nodes"], workers=args.workers)
    payload["nodes"] = surviving
    args.outp.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(
            [{"id": i, "name": n, "reason": r} for i, n, r in dropped], indent=2, ensure_ascii=False
        )
    )
    print(
        f"[llm_prune] dropped {len(dropped)} non-genre nodes -> {len(surviving)} kept "
        f"(report: {args.report})"
    )


if __name__ == "__main__":
    main()
