from __future__ import annotations

import asyncio
import json
import random
from pathlib import Path

import httpx

from music_evolution.enrich_previews import (
    _itunes,
    _deezer,
    _score,
    _norm,
    _overlap,
    SKIP_ARTISTS,
)

random.seed(42)


async def main() -> None:
    d = json.load(open("genres.json"))
    # Look at modern genres only (post-1960) where iTunes coverage should be near-total.
    misses: list[tuple[str, dict]] = []
    for n in d["nodes"]:
        if n.get("birth_year", 0) < 1960:
            continue
        for ex in n.get("examples", []):
            if ex.get("preview_url"):
                continue
            if ex.get("artist", "").strip().lower() in SKIP_ARTISTS:
                continue
            misses.append((n["name"], ex))
    print(f"modern misses: {len(misses)}")
    sample = random.sample(misses, 25)

    async with httpx.AsyncClient(headers={"User-Agent": "music-evolution-debug"}) as client:
        for genre, ex in sample:
            artist, title, year = ex["artist"], ex["title"], ex.get("year")
            print(f"\n[{genre}] {artist!r} — {title!r} ({year})")
            for fetcher_name, fetcher in (("iTunes", _itunes), ("Deezer", _deezer)):
                try:
                    cands = await fetcher(client, artist, title)
                except Exception as e:
                    print(f"  {fetcher_name}: ERROR {e}")
                    continue
                if not cands:
                    print(f"  {fetcher_name}: no results")
                    continue
                print(f"  {fetcher_name}: {len(cands)} results")
                for c in cands[:3]:
                    s = _score(artist, title, year, c)
                    a_o = _overlap(artist, c.artist)
                    t_o = _overlap(title, c.title)
                    has = "✓" if c.preview_url else "✗"
                    print(
                        f"    {has} score={s:.2f}  a_o={a_o:.2f} t_o={t_o:.2f}  "
                        f"yr={c.year}  {c.artist[:30]!r} — {c.title[:40]!r}"
                    )
                    print(
                        f"        norm_q_artist={_norm(artist)!r}  norm_r_artist={_norm(c.artist)!r}"
                    )
                    print(
                        f"        norm_q_title ={_norm(title)!r}  norm_r_title ={_norm(c.title)!r}"
                    )


if __name__ == "__main__":
    asyncio.run(main())
