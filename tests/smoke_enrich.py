from __future__ import annotations

import asyncio
import json
from pathlib import Path

from music_evolution.enrich_previews import enrich

# Hand-picked spread: a guaranteed-canonical pop/rock track, an obscure jazz track,
# a non-Latin example, a "Traditional" that should be skipped, and a fusion microgenre.
SAMPLE = {
    "nodes": [
        {
            "examples": [
                {"artist": "Michael Jackson", "title": "Billie Jean", "year": 1982},
                {"artist": "Charlie Parker", "title": "Ko-Ko", "year": 1945},
                {"artist": "Fela Kuti", "title": "Zombie", "year": 1976},
                {"artist": "Traditional", "title": "Navajo Night Chant", "year": None},
                {"artist": "Macintosh Plus", "title": "リサフランク420 / 現代のコンピュー", "year": 2011},
                {"artist": "Witch House Made-Up Group", "title": "Definitely Not Real Track", "year": 2012},
            ]
        }
    ]
}


async def main() -> None:
    tmp = Path("/tmp/_smoke_enrich")
    tmp.mkdir(exist_ok=True)
    in_p = tmp / "in.json"
    out_p = tmp / "out.json"
    cache_p = tmp / "cache.json"
    if cache_p.exists():
        cache_p.unlink()
    in_p.write_text(json.dumps(SAMPLE))

    await enrich(in_p, out_p, cache_p, concurrency=4, refresh=True)

    out = json.loads(out_p.read_text())
    for ex in out["nodes"][0]["examples"]:
        print(
            f"  {ex['artist'][:30]:30s} | {ex['title'][:40]:40s} -> "
            f"{ex.get('preview_provider') or '—':6s} {'✓' if ex.get('preview_url') else '✗'}"
        )


if __name__ == "__main__":
    asyncio.run(main())
