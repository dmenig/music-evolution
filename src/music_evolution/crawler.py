from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

from .gemini_client import expand_node
from .normalize import canonical_id, merge_alias, resolve
from .prompts import OriginatorContext
from .schema import (
    EdgeStrength,
    GeminiGenreResponse,
    GenreEdge,
    GenreNode,
    StoredEdge,
)
from .seed_loader import WikiSeed, load_seeds

DATA_DIR = Path("data")
ALIASES_FILE = DATA_DIR / "aliases.json"
FRONTIER_FILE = DATA_DIR / "frontier.json"
VISITED_FILE = DATA_DIR / "visited.json"

STRENGTH_VAL: dict[str, int] = {"minor": 1, "mid": 2, "major": 3}
VAL_STRENGTH: dict[int, EdgeStrength] = {1: "minor", 2: "mid", 3: "major"}


def _load_json(path: Path, default):  # type: ignore[no-untyped-def]
    if path.exists():
        return json.loads(path.read_text())
    return default


def _save_json(path: Path, obj) -> None:  # type: ignore[no-untyped-def]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))


@dataclass(slots=True)
class FrontierEntry:
    name: str
    contexts: list[OriginatorContext]


def _to_node(
    cid: str,
    raw: GeminiGenreResponse,
    aliases: dict[str, str],
) -> GenreNode:
    parent_edges = _resolve_edges(raw.parents, aliases)
    child_edges = _resolve_edges(raw.children, aliases)
    death = raw.death_year if raw.death_year and raw.death_year > raw.birth_year else None
    peak = max(raw.birth_year, min(raw.peak_year, death or 2026))
    parents = sorted({e.id for e in parent_edges if e.id != cid})
    children = sorted({e.id for e in child_edges if e.id != cid})
    return GenreNode(
        id=cid,
        name=raw.name,
        aliases=raw.aliases,
        parents=parents,
        children=children,
        parent_edges=[e for e in parent_edges if e.id != cid],
        child_edges=[e for e in child_edges if e.id != cid],
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


def _resolve_edges(edges: list[GenreEdge], aliases: dict[str, str]) -> list[StoredEdge]:
    out: dict[str, StoredEdge] = {}
    for e in edges:
        if not e.name.strip():
            continue
        cid = resolve(aliases, e.name)
        if not cid:
            continue
        if cid in out:
            # First write wins; later self-mentions ignored.
            continue
        out[cid] = StoredEdge(id=cid, name=e.name, description=e.description, strength=e.strength)
    return list(out.values())


def _expand_one(
    name: str, wiki_desc: str | None, contexts: list[OriginatorContext]
) -> tuple[str, GeminiGenreResponse | None, str | None]:
    cid = canonical_id(name)
    try:
        return cid, expand_node(cid, name, wiki_desc, contexts), None
    except Exception as exc:  # noqa: BLE001
        return cid, None, f"{type(exc).__name__}: {exc}"


def _seed_frontier(seeds: dict[str, WikiSeed]) -> deque[FrontierEntry]:
    frontier: deque[FrontierEntry] = deque()
    for s in seeds.values():
        frontier.append(FrontierEntry(name=s.name, contexts=[]))
    return frontier


def crawl(workers: int = 8, max_nodes: int | None = None) -> dict[str, GenreNode]:
    seeds = load_seeds()
    print(f"[crawl] {len(seeds)} Wikipedia seed genres", flush=True)

    aliases: dict[str, str] = _load_json(ALIASES_FILE, {})
    visited_raw: dict[str, dict] = _load_json(VISITED_FILE, {})
    visited: dict[str, GenreNode] = {k: GenreNode(**v) for k, v in visited_raw.items()}

    frontier_raw = _load_json(FRONTIER_FILE, None)
    if frontier_raw is None:
        frontier = _seed_frontier(seeds)
    else:
        frontier = deque(
            FrontierEntry(
                name=e["name"],
                contexts=[OriginatorContext(**c) for c in e.get("contexts", [])],
            )
            for e in frontier_raw
        )

    pending_contexts: dict[str, list[OriginatorContext]] = defaultdict(list)
    for fe in frontier:
        if fe.contexts:
            pending_contexts[canonical_id(fe.name)].extend(fe.contexts)

    def remember() -> None:
        _save_json(ALIASES_FILE, aliases)
        _save_json(VISITED_FILE, {k: v.model_dump() for k, v in visited.items()})
        _save_json(
            FRONTIER_FILE,
            [
                {
                    "name": e.name,
                    "contexts": [asdict(c) for c in e.contexts],
                }
                for e in frontier
            ],
        )

    pool = ThreadPoolExecutor(max_workers=workers)
    inflight: dict = {}
    queued: set[str] = set()

    def can_schedule_more() -> bool:
        return max_nodes is None or len(visited) + len(inflight) < max_nodes

    def schedule(entry: FrontierEntry) -> None:
        cid = resolve(aliases, entry.name)
        if cid in visited or cid in queued or not can_schedule_more():
            return
        queued.add(cid)
        wiki = seeds.get(cid)
        wiki_desc = wiki.description if wiki else None
        merged = list(entry.contexts) + pending_contexts.pop(cid, [])
        fut = pool.submit(_expand_one, entry.name, wiki_desc, merged)
        inflight[fut] = (cid, entry.name)

    while frontier and can_schedule_more():
        while frontier and len(inflight) < workers and can_schedule_more():
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
        # Push parents and children onto frontier with originator context.
        for direction, edges in (("parent", raw.parents), ("child", raw.children)):
            for e in edges:
                ncid = resolve(aliases, e.name)
                if not ncid or ncid == cid_done or ncid in visited:
                    continue
                ctx = OriginatorContext(
                    upstream_name=raw.name,
                    upstream_description=raw.description,
                    direction=direction,
                    asserted_strength=e.strength,
                    asserted_description=e.description,
                )
                pending_contexts[ncid].append(ctx)
                if ncid in queued:
                    continue
                frontier.append(FrontierEntry(name=e.name, contexts=[ctx]))
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
    """Make parent/child symmetric, drop dangling refs, break cycles, average strengths."""
    ids = set(nodes)

    # 1) Filter dangling and self-edges.
    for n in nodes.values():
        n.parent_edges = [e for e in n.parent_edges if e.id in ids and e.id != n.id]
        n.child_edges = [e for e in n.child_edges if e.id in ids and e.id != n.id]
        n.parents = [p for p in n.parents if p in ids and p != n.id]
        n.children = [c for c in n.children if c in ids and c != n.id]

    # 2) Mirror edges so every parent->child also has child->parent and vice versa.
    #    When mirroring, reuse the existing description if any; default strength = the original's.
    for n in list(nodes.values()):
        for e in n.parent_edges:
            par = nodes[e.id]
            if not any(c.id == n.id for c in par.child_edges):
                par.child_edges.append(
                    StoredEdge(id=n.id, name=n.name, description=n.description, strength=e.strength)
                )
        for e in n.child_edges:
            ch = nodes[e.id]
            if not any(p.id == n.id for p in ch.parent_edges):
                ch.parent_edges.append(
                    StoredEdge(id=n.id, name=n.name, description=n.description, strength=e.strength)
                )

    # 3) Average strengths across the two directions of each edge.
    _average_strengths(nodes)

    # 4) Break cycles using birth-year monotonicity.
    _break_cycles(nodes)

    # 5) Rebuild flat parents/children id lists from edges.
    for n in nodes.values():
        n.parents = sorted({e.id for e in n.parent_edges})
        n.children = sorted({e.id for e in n.child_edges})

    _enforce_year_invariants(nodes)
    return nodes


def _average_strengths(nodes: dict[str, GenreNode]) -> None:
    """For every (parent, child) pair, average the two asserted strengths and write
    the same canonical strength back to both sides."""
    for n in nodes.values():
        for e in n.child_edges:
            child = nodes[e.id]
            mate = next((p for p in child.parent_edges if p.id == n.id), None)
            if mate is None:
                continue
            avg = round((STRENGTH_VAL[e.strength] + STRENGTH_VAL[mate.strength]) / 2)
            label = VAL_STRENGTH[max(1, min(3, avg))]
            e.strength = label
            mate.strength = label


def _break_cycles(nodes: dict[str, GenreNode]) -> None:
    """Demote any parent->child edge where parent.birth >= child.birth into influences."""
    for n in nodes.values():
        keep_parents: list[StoredEdge] = []
        for e in n.parent_edges:
            par = nodes[e.id]
            if par.birth_year < n.birth_year:
                keep_parents.append(e)
                continue
            if not any(i.id == e.id for i in n.influence_edges):
                n.influence_edges.append(e)
            par.child_edges = [c for c in par.child_edges if c.id != n.id]
        n.parent_edges = keep_parents
    for n in nodes.values():
        n.influences = sorted({e.id for e in n.influence_edges})


def _enforce_year_invariants(nodes: dict[str, GenreNode]) -> None:
    for n in nodes.values():
        if n.death_year is not None and n.death_year < n.birth_year:
            n.death_year = None
        n.peak_year = max(n.birth_year, min(n.peak_year, n.death_year or 2026))


def write_genres_json(nodes: dict[str, GenreNode], path: Path) -> None:
    payload = {
        "version": 2,
        "generated_at": int(time.time()),
        "nodes": [
            n.model_dump() for n in sorted(nodes.values(), key=lambda x: (x.birth_year, x.id))
        ],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=None,
        help="Hard cap on nodes (default: no cap, drains the frontier).",
    )
    parser.add_argument("--out", type=Path, default=Path("genres.json"))
    args = parser.parse_args()
    nodes = crawl(workers=args.workers, max_nodes=args.max_nodes)
    nodes = stitch_and_finalize(nodes)
    write_genres_json(nodes, args.out)
    print(f"[done] {len(nodes)} nodes -> {args.out}")


if __name__ == "__main__":
    main()
