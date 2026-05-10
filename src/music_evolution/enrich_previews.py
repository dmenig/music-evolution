from __future__ import annotations

import argparse
import asyncio
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import httpx

ITUNES_URL = "https://itunes.apple.com/search"
DEEZER_URL = "https://api.deezer.com/search"

JUNK_PATTERNS = re.compile(
    r"\b(karaoke|tribute|made famous by|in the style of|originally performed|"
    r"performed by|cover version|instrumental version|remake|backing track|"
    r"as made famous|sound[- ]?alike)\b",
    re.IGNORECASE,
)

# "Traditional" / "Anonymous" / "Various" examples (medieval chant, folk, indigenous
# music) almost never resolve to a usable commercial recording — skip them rather
# than risk a confidently wrong match.
SKIP_ARTISTS = {"traditional", "anonymous", "various", "various artists", "unknown"}

ACCEPT_THRESHOLD = 1.6  # artist (≤1) + title (≤1) + year bonus (≤0.4)


@dataclass(slots=True)
class Candidate:
    provider: str
    artist: str
    title: str
    year: int | None
    preview_url: str | None


def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def _norm(s: str) -> str:
    s = _strip_accents(s).lower()
    s = re.sub(r"[\(\[\{].*?[\)\]\}]", " ", s)  # drop parentheticals
    s = re.sub(r"\bfeat\.?\b|\bft\.?\b|\bfeaturing\b", " ", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _tokens(s: str) -> set[str]:
    return {t for t in _norm(s).split() if len(t) > 1}


def _overlap(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    return inter / min(len(ta), len(tb))


def _score(query_artist: str, query_title: str, query_year: int | None, c: Candidate) -> float:
    if JUNK_PATTERNS.search(c.title) or JUNK_PATTERNS.search(c.artist):
        return -1.0
    artist_o = _overlap(query_artist, c.artist)
    title_o = _overlap(query_title, c.title)
    if artist_o < 0.5 or title_o < 0.5:
        return -1.0
    year_bonus = 0.0
    if query_year and c.year:
        delta = abs(query_year - c.year)
        if delta <= 2:
            year_bonus = 0.4
        elif delta <= 5:
            year_bonus = 0.2
        elif delta > 12:
            return -1.0
    return artist_o + title_o + year_bonus


async def _get_with_retry(
    client: httpx.AsyncClient, url: str, params: dict, attempts: int = 3
) -> httpx.Response | None:
    backoff = 0.6
    for i in range(attempts):
        try:
            r = await client.get(url, params=params, timeout=15.0)
        except httpx.HTTPError:
            await asyncio.sleep(backoff * (2**i))
            continue
        if r.status_code == 200:
            return r
        if r.status_code in (429, 403, 500, 502, 503, 504):
            await asyncio.sleep(backoff * (2**i))
            continue
        return r
    return None


async def _itunes(client: httpx.AsyncClient, artist: str, title: str) -> list[Candidate]:
    params = {"term": f"{artist} {title}", "entity": "song", "limit": 8, "media": "music"}
    r = await _get_with_retry(client, ITUNES_URL, params)
    if r is None or r.status_code != 200:
        return []
    try:
        body = r.json()
    except json.JSONDecodeError:
        return []
    out: list[Candidate] = []
    for it in body.get("results", []):
        rd = it.get("releaseDate") or ""
        year = int(rd[:4]) if rd[:4].isdigit() else None
        out.append(
            Candidate(
                provider="itunes",
                artist=it.get("artistName", "") or "",
                title=it.get("trackName", "") or "",
                year=year,
                preview_url=it.get("previewUrl"),
            )
        )
    return out


async def _deezer(client: httpx.AsyncClient, artist: str, title: str) -> list[Candidate]:
    # Free-form first — strict field syntax (`artist:"X" track:"Y"`) fails for non-Latin
    # scripts and apostrophe variants more often than it helps.
    r = await _get_with_retry(client, DEEZER_URL, {"q": f"{artist} {title}", "limit": 8})
    data: list = []
    if r is not None and r.status_code == 200:
        try:
            data = r.json().get("data") or []
        except json.JSONDecodeError:
            data = []
    if not data:
        r2 = await _get_with_retry(
            client, DEEZER_URL, {"q": f'artist:"{artist}" track:"{title}"', "limit": 8}
        )
        if r2 is not None and r2.status_code == 200:
            try:
                data = r2.json().get("data") or []
            except json.JSONDecodeError:
                data = []
    out: list[Candidate] = []
    for it in data:
        out.append(
            Candidate(
                provider="deezer",
                artist=(it.get("artist") or {}).get("name", "") or "",
                title=it.get("title", "") or "",
                year=None,
                preview_url=it.get("preview") or None,
            )
        )
    return out


async def _resolve_one(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    artist: str,
    title: str,
    year: int | None,
) -> tuple[str | None, str | None]:
    if artist.strip().lower() in SKIP_ARTISTS:
        return None, None
    async with sem:
        best: tuple[float, Candidate | None] = (-1.0, None)
        # Deezer first: iTunes Search aggressively rate-limits / 403s on bulk runs
        # from a single IP, so it's the fallback rather than the primary.
        for fetcher in (_deezer, _itunes):
            try:
                cands = await fetcher(client, artist, title)
            except (httpx.HTTPError, json.JSONDecodeError):
                cands = []
            for c in cands:
                if not c.preview_url:
                    continue
                s = _score(artist, title, year, c)
                if s > best[0]:
                    best = (s, c)
            if best[0] >= ACCEPT_THRESHOLD:
                break
    if best[1] is None or best[0] < ACCEPT_THRESHOLD:
        return None, None
    return best[1].preview_url, best[1].provider


def _cache_key(artist: str, title: str, year: int | None) -> str:
    return f"{_norm(artist)}|{_norm(title)}|{year or ''}"


async def enrich(
    in_path: Path,
    out_path: Path,
    cache_path: Path,
    concurrency: int,
    refresh: bool,
) -> None:
    payload = json.loads(in_path.read_text())
    cache: dict[str, dict[str, str | None]] = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text())

    jobs: list[tuple[str, str, int | None, dict]] = []
    for node in payload["nodes"]:
        for ex in node.get("examples", []):
            artist, title = ex.get("artist", ""), ex.get("title", "")
            if not artist or not title:
                continue
            year = ex.get("year")
            key = _cache_key(artist, title, year)
            cached = cache.get(key)
            # Re-resolve cached null entries: a null result is much more likely to be a
            # transient rate-limit / 403 / empty-page artifact than a genuine "no preview
            # exists anywhere" verdict, so we always retry misses unless the user passes
            # --keep-misses. Successful (url-bearing) cache entries are honored as-is.
            if cached and cached.get("url") and not refresh:
                ex["preview_url"] = cached["url"]
                ex["preview_provider"] = cached.get("provider")
                continue
            if not refresh and ex.get("preview_url"):
                continue
            jobs.append((artist, title, year, ex))

    print(
        f"[enrich] {len(jobs)} examples to resolve "
        f"(cache hits: {sum(1 for n in payload['nodes'] for e in n.get('examples', []) if e.get('preview_url'))})"
    )

    sem = asyncio.Semaphore(concurrency)
    headers = {"User-Agent": "music-evolution/0.1 (preview-resolver)"}
    done = 0
    hits = 0
    cache_dirty = 0
    flush_every = 200

    async with httpx.AsyncClient(headers=headers, http2=False) as client:

        async def _do(artist: str, title: str, year: int | None, ex: dict) -> None:
            nonlocal done, hits, cache_dirty
            url, provider = await _resolve_one(client, sem, artist, title, year)
            ex["preview_url"] = url
            ex["preview_provider"] = provider
            cache[_cache_key(artist, title, year)] = {"url": url, "provider": provider}
            done += 1
            cache_dirty += 1
            if url:
                hits += 1
            if done % 100 == 0:
                print(f"[enrich] {done}/{len(jobs)} resolved · hit rate {hits / done:.1%}")
            if cache_dirty >= flush_every:
                cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=0))
                cache_dirty = 0

        await asyncio.gather(*(_do(*j) for j in jobs))

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=0))

    total_examples = sum(len(n.get("examples", [])) for n in payload["nodes"])
    with_preview = sum(
        1 for n in payload["nodes"] for e in n.get("examples", []) if e.get("preview_url")
    )
    out_path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    print(
        f"[enrich] wrote {out_path} · {with_preview}/{total_examples} examples "
        f"have a preview ({with_preview / total_examples:.1%})"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", type=Path, default=Path("genres.json"))
    ap.add_argument("--out", dest="out_path", type=Path, default=Path("genres.json"))
    ap.add_argument("--cache", type=Path, default=Path("data/previews.json"))
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--refresh", action="store_true", help="ignore cache and re-resolve")
    args = ap.parse_args()
    asyncio.run(enrich(args.in_path, args.out_path, args.cache, args.concurrency, args.refresh))


if __name__ == "__main__":
    main()
