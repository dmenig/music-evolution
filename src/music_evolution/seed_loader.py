from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from .normalize import canonical_id

WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKI_REST = "https://en.wikipedia.org/api/rest_v1/page/summary/"
SEED_PAGE = "List_of_music_genres_and_styles"
SEED_CACHE = Path("data/wiki_seed.json")
USER_AGENT = "music-evolution-crawler/0.1 (research; contact: damien.menigaux@veesion.com)"


@dataclass(slots=True, frozen=True)
class WikiSeed:
    id: str
    name: str
    description: str
    url: str


import re

META_PREFIXES = ("List of ", "Styles of ", "Outline of ", "Index of ", "Glossary of ")
META_KEYWORDS = ("disambiguation",)
NON_GENRE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\d{3,4}s? in ", re.IGNORECASE),
    re.compile(r"^\d{3,4}s? music$", re.IGNORECASE),
    re.compile(r"^Music of ", re.IGNORECASE),
    re.compile(r"^Music in ", re.IGNORECASE),
    re.compile(r"^Music from ", re.IGNORECASE),
    re.compile(r"^History of ", re.IGNORECASE),
    re.compile(r"^Timeline of ", re.IGNORECASE),
    re.compile(r"^Category:", re.IGNORECASE),
)


def _is_genre_title(title: str) -> bool:
    if title.startswith(META_PREFIXES):
        return False
    low = title.lower()
    if any(k in low for k in META_KEYWORDS):
        return False
    return not any(p.search(title) for p in NON_GENRE_PATTERNS)


@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=1, max=20))
def _fetch_link_titles(client: httpx.Client) -> list[str]:
    """All blue-link page titles on the genres list (namespace 0 only)."""
    titles: list[str] = []
    plcontinue: str | None = None
    while True:
        params: dict[str, str] = {
            "action": "parse",
            "page": SEED_PAGE,
            "prop": "links",
            "format": "json",
            "redirects": "1",
        }
        if plcontinue:
            params["plcontinue"] = plcontinue
        r = client.get(WIKI_API, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        for link in data["parse"]["links"]:
            if link.get("ns") == 0 and "exists" in link and _is_genre_title(link["*"]):
                titles.append(link["*"])
        cont = data.get("continue", {}).get("plcontinue")
        if not cont:
            break
        plcontinue = cont
    seen: set[str] = set()
    out: list[str] = []
    for t in titles:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


@retry(stop=stop_after_attempt(6), wait=wait_exponential(multiplier=2, min=2, max=30))
def _fetch_summary(client: httpx.Client, title: str) -> WikiSeed | None:
    r = client.get(WIKI_REST + title.replace(" ", "_"), timeout=30)
    if r.status_code in (404, 410):
        return None
    if r.status_code in (429, 503):
        r.raise_for_status()  # tenacity retries
    r.raise_for_status()
    j = r.json()
    if j.get("type") == "disambiguation":
        return None
    extract = (j.get("extract") or "").strip()
    if not extract:
        return None
    name = j.get("title") or title
    return WikiSeed(
        id=canonical_id(name),
        name=name,
        description=extract,
        url=j.get("content_urls", {}).get("desktop", {}).get("page", ""),
    )


def _load_cache() -> dict[str, dict]:
    if SEED_CACHE.exists():
        return json.loads(SEED_CACHE.read_text())
    return {}


def _save_cache(cache: dict[str, dict]) -> None:
    SEED_CACHE.parent.mkdir(parents=True, exist_ok=True)
    SEED_CACHE.write_text(json.dumps(cache, indent=2, ensure_ascii=False))


def load_seeds(workers: int = 4, refresh_titles: bool = False) -> dict[str, WikiSeed]:
    """Return canonical_id -> WikiSeed for every genre on the master list page.

    Caches everything to data/wiki_seed.json. Subsequent calls are offline."""
    cache = _load_cache()
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    with httpx.Client(headers=headers, follow_redirects=True) as client:
        if refresh_titles or not cache.get("_titles"):
            titles = _fetch_link_titles(client)
            cache["_titles"] = titles
            _save_cache(cache)
        else:
            titles = cache["_titles"]

        # Re-attempt entries previously marked _failed (transient rate limits).
        missing = [t for t in titles if t not in cache or cache[t].get("_failed")]
        if missing:
            print(f"[seed] fetching {len(missing)} missing summaries", flush=True)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futs = {pool.submit(_fetch_summary, client, t): t for t in missing}
                done = 0
                for fut in as_completed(futs):
                    title = futs[fut]
                    try:
                        seed = fut.result()
                    except Exception as exc:  # noqa: BLE001
                        print(f"[seed err] {title}: {type(exc).__name__}: {exc}", file=sys.stderr)
                        cache[title] = {"_failed": True}
                        seed = None
                    if seed is not None:
                        cache[title] = {
                            "id": seed.id,
                            "name": seed.name,
                            "description": seed.description,
                            "url": seed.url,
                        }
                    else:
                        cache.setdefault(title, {"_skipped": True})
                    done += 1
                    if done % 100 == 0:
                        _save_cache(cache)
                        print(f"[seed] {done}/{len(missing)}", flush=True)
            _save_cache(cache)

    out: dict[str, WikiSeed] = {}
    for title, entry in cache.items():
        if title == "_titles" or "id" not in entry:
            continue
        seed = WikiSeed(
            id=entry["id"], name=entry["name"], description=entry["description"], url=entry["url"]
        )
        # If two wiki titles canonicalize to the same id, prefer the one with the
        # longer description (richer signal); ignore the shorter duplicate.
        if seed.id in out and len(out[seed.id].description) >= len(seed.description):
            continue
        out[seed.id] = seed
    return out


def main() -> None:
    seeds = load_seeds()
    print(f"[seed] {len(seeds)} canonical genres seeded from Wikipedia")


if __name__ == "__main__":
    main()
