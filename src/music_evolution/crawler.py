from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .gemini_client import expand_node
from .normalize import canonical_id, merge_alias, resolve
from .schema import GeminiGenreResponse, GenreNode

DATA_DIR = Path("data")
ALIASES_FILE = DATA_DIR / "aliases.json"
FRONTIER_FILE = DATA_DIR / "frontier.json"
VISITED_FILE = DATA_DIR / "visited.json"

SEEDS: list[str] = [
    "Folk Music",
    "Western Classical Music",
    "Gregorian Chant",
    "Religious Chant",
    "Work Song",
    "Sea Shanty",
    "Court Music",
    "Military Music",
    "Indigenous Music",
    "Hindustani Classical Music",
    "Carnatic Music",
    "Gagaku",
    "Gamelan",
    "Andalusian Classical Music",
    "Persian Traditional Music",
    "Chinese Traditional Music",
    "Arabic Maqam",
    "African Traditional Music",
    "Native American Music",
    "Polynesian Music",
    "Throat Singing",
    "Klezmer",
    "Flamenco",
    "Fado",
]

MAX_NODES = 6000


def _load_json(path: Path, default):  # type: ignore[no-untyped-def]
    if path.exists():
        return json.loads(path.read_text())
    return default


def _save_json(path: Path, obj) -> None:  # type: ignore[no-untyped-def]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))


def _to_node(cid: str, raw: GeminiGenreResponse, aliases: dict[str, str]) -> GenreNode:
    parents = [resolve(aliases, p) for p in raw.parents if p.strip()]
    children = [resolve(aliases, c) for c in raw.children if c.strip()]
    death = raw.death_year if raw.death_year and raw.death_year > raw.birth_year else None
    peak = max(raw.birth_year, min(raw.peak_year, death or 2026))
    return GenreNode(
        id=cid,
        name=raw.name,
        aliases=raw.aliases,
        parents=sorted(set(parents)),
        children=sorted(set(children)),
        birth_year=raw.birth_year,
        death_year=death,
        peak_year=peak,
        popularity=raw.popularity,
        countries=raw.countries,
        artists=raw.artists,
        examples=raw.examples,
        description=raw.description,
        sources=raw.sources,
    )


def _expand_one(name: str) -> tuple[str, GeminiGenreResponse | None, str | None]:
    cid = canonical_id(name)
    try:
        return cid, expand_node(cid, name), None
    except Exception as exc:  # noqa: BLE001
        return cid, None, f"{type(exc).__name__}: {exc}"


def crawl(workers: int = 8, max_nodes: int = MAX_NODES) -> dict[str, GenreNode]:
    aliases: dict[str, str] = _load_json(ALIASES_FILE, {})
    visited_raw: dict[str, dict] = _load_json(VISITED_FILE, {})
    visited: dict[str, GenreNode] = {k: GenreNode(**v) for k, v in visited_raw.items()}
    frontier_raw: list[str] = _load_json(FRONTIER_FILE, list(SEEDS))
    frontier: deque[str] = deque(frontier_raw)

    def remember() -> None:
        _save_json(ALIASES_FILE, aliases)
        _save_json(VISITED_FILE, {k: v.model_dump() for k, v in visited.items()})
        _save_json(FRONTIER_FILE, list(frontier))

    pool = ThreadPoolExecutor(max_workers=workers)
    inflight: dict = {}
    queued: set[str] = set()

    def schedule(name: str) -> None:
        cid = resolve(aliases, name)
        if cid in visited or cid in queued or len(visited) + len(inflight) >= max_nodes:
            return
        queued.add(cid)
        fut = pool.submit(_expand_one, name)
        inflight[fut] = (cid, name)

    while frontier and len(visited) < max_nodes:
        while frontier and len(inflight) < workers and len(visited) + len(inflight) < max_nodes:
            schedule(frontier.popleft())
        if not inflight:
            break
        done = next(as_completed(inflight))
        cid_done, raw, err = done.result()
        _, name = inflight.pop(done)
        queued.discard(cid_done)
        if err:
            print(f"[err] {name}: {err}", file=sys.stderr)
            continue
        assert raw is not None
        merge_alias(aliases, raw.name, cid_done)
        for a in raw.aliases:
            merge_alias(aliases, a, cid_done)
        node = _to_node(cid_done, raw, aliases)
        visited[cid_done] = node
        for nxt in raw.parents + raw.children:
            ncid = resolve(aliases, nxt)
            if ncid and ncid not in visited and ncid not in queued:
                frontier.append(nxt)
        if len(visited) % 25 == 0:
            remember()
            print(
                f"[crawl] visited={len(visited)} frontier={len(frontier)} "
                f"inflight={len(inflight)}",
                flush=True,
            )

    pool.shutdown(wait=True)
    remember()
    return visited


def stitch_and_finalize(nodes: dict[str, GenreNode]) -> dict[str, GenreNode]:
    """Make parent/child symmetric, drop dangling refs, break cycles."""
    ids = set(nodes)
    for n in nodes.values():
        n.parents = [p for p in n.parents if p in ids and p != n.id]
        n.children = [c for c in n.children if c in ids and c != n.id]
    for n in list(nodes.values()):
        for p in n.parents:
            if n.id not in nodes[p].children:
                nodes[p].children.append(n.id)
        for c in n.children:
            if n.id not in nodes[c].parents:
                nodes[c].parents.append(n.id)
    _break_cycles(nodes)
    _enforce_year_invariants(nodes)
    return nodes


def _break_cycles(nodes: dict[str, GenreNode]) -> None:
    """Demote any parent->child edge where parent.birth >= child.birth into influences."""
    for n in nodes.values():
        keep_parents: list[str] = []
        for p in n.parents:
            if nodes[p].birth_year < n.birth_year:
                keep_parents.append(p)
            else:
                if p not in n.influences:
                    n.influences.append(p)
                if n.id in nodes[p].children:
                    nodes[p].children.remove(n.id)
        n.parents = keep_parents


def _enforce_year_invariants(nodes: dict[str, GenreNode]) -> None:
    for n in nodes.values():
        if n.death_year is not None and n.death_year < n.birth_year:
            n.death_year = None
        n.peak_year = max(n.birth_year, min(n.peak_year, n.death_year or 2026))


def write_genres_json(nodes: dict[str, GenreNode], path: Path) -> None:
    payload = {"version": 1, "generated_at": int(time.time()), "nodes": [
        n.model_dump() for n in sorted(nodes.values(), key=lambda x: (x.birth_year, x.id))
    ]}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-nodes", type=int, default=MAX_NODES)
    parser.add_argument("--out", type=Path, default=Path("genres.json"))
    args = parser.parse_args()
    nodes = crawl(workers=args.workers, max_nodes=args.max_nodes)
    nodes = stitch_and_finalize(nodes)
    write_genres_json(nodes, args.out)
    print(f"[done] {len(nodes)} nodes -> {args.out}")


if __name__ == "__main__":
    main()
