"""Merge nodes whose names canonicalize differently but are explicit aliases of each other.

The crawler's alias map catches LLM-asserted aliases at expansion time, but two
parallel expansions (e.g. "Stride" and "Stride music") can produce two separate
nodes before either's alias claim arrives. This pass runs after stitching: it
unions nodes whose `aliases` field cross-references one another's canonical id.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from .normalize import canonical_id

STRENGTH_RANK = {"minor": 1, "mid": 2, "major": 3}
RANK_STRENGTH = {1: "minor", 2: "mid", 3: "major"}


def _find_clusters(nodes: list[dict]) -> list[list[str]]:
    """Union-find. Two nodes are merged if either:
    (a) one's alias canonicalizes to the other's id, or
    (b) they share any alias canonical_id (even one that isn't itself a node).
    """
    by_id = {n["id"]: n for n in nodes}
    parent: dict[str, str] = {n["id"]: n["id"] for n in nodes}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    alias_to_owners: dict[str, list[str]] = defaultdict(list)
    for n in nodes:
        for a in n.get("aliases") or []:
            aid = canonical_id(a)
            if not aid:
                continue
            if aid in by_id and aid != n["id"]:
                union(n["id"], aid)
            alias_to_owners[aid].append(n["id"])

    for aid, owners in alias_to_owners.items():
        if len(owners) < 2:
            continue
        first = owners[0]
        for o in owners[1:]:
            union(first, o)

    groups: dict[str, list[str]] = defaultdict(list)
    for nid in parent:
        groups[find(nid)].append(nid)
    return [g for g in groups.values() if len(g) > 1]


def _winner(group: list[dict]) -> dict:
    return max(
        group,
        key=lambda n: (
            len(n.get("sources") or []),
            len(n.get("parents") or []) + len(n.get("children") or []),
            n.get("popularity") or 0.0,
            -len(n["id"]),
        ),
    )


def _merge_edge_lists(*lists: list[dict]) -> list[dict]:
    """Union by edge id; keep strongest strength."""
    by_id: dict[str, dict] = {}
    for lst in lists:
        for e in lst:
            existing = by_id.get(e["id"])
            if existing is None:
                by_id[e["id"]] = dict(e)
                continue
            if STRENGTH_RANK[e["strength"]] > STRENGTH_RANK[existing["strength"]]:
                by_id[e["id"]] = dict(e)
    return list(by_id.values())


def _remap_edges(edges: list[dict], remap: dict[str, str]) -> list[dict]:
    seen: dict[str, dict] = {}
    for e in edges:
        new_id = remap.get(e["id"], e["id"])
        e2 = dict(e)
        e2["id"] = new_id
        existing = seen.get(new_id)
        if existing is None:
            seen[new_id] = e2
            continue
        if STRENGTH_RANK[e2["strength"]] > STRENGTH_RANK[existing["strength"]]:
            seen[new_id] = e2
    return list(seen.values())


def _merge_into_winner(winner: dict, losers: list[dict]) -> None:
    win_aliases = set(winner.get("aliases") or [])
    win_artists = set(winner.get("artists") or [])
    win_countries = list(winner.get("countries") or [])
    win_country_set = set(win_countries)
    win_sources = list(winner.get("sources") or [])
    win_source_set = set(win_sources)
    win_examples_seen = {(e.get("artist"), e.get("title")) for e in winner.get("examples") or []}

    for o in losers:
        # Loser's primary name becomes an alias.
        win_aliases.add(o["name"])
        for a in o.get("aliases") or []:
            win_aliases.add(a)
        for a in o.get("artists") or []:
            win_artists.add(a)
        for c in o.get("countries") or []:
            if c not in win_country_set:
                win_countries.append(c)
                win_country_set.add(c)
        for s in o.get("sources") or []:
            if s not in win_source_set:
                win_sources.append(s)
                win_source_set.add(s)
        for ex in o.get("examples") or []:
            key = (ex.get("artist"), ex.get("title"))
            if key not in win_examples_seen:
                winner["examples"].append(ex)
                win_examples_seen.add(key)

    win_aliases.discard(winner["name"])
    winner["aliases"] = sorted(win_aliases)
    winner["artists"] = sorted(win_artists)
    winner["countries"] = win_countries
    winner["sources"] = win_sources

    # Edges: union losers' edges, then dedupe by id.
    winner["parent_edges"] = _merge_edge_lists(
        winner.get("parent_edges") or [], *(o.get("parent_edges") or [] for o in losers)
    )
    winner["child_edges"] = _merge_edge_lists(
        winner.get("child_edges") or [], *(o.get("child_edges") or [] for o in losers)
    )
    winner["influence_edges"] = _merge_edge_lists(
        winner.get("influence_edges") or [], *(o.get("influence_edges") or [] for o in losers)
    )


def dedupe(nodes: list[dict]) -> tuple[list[dict], int]:
    """Returns (deduped_nodes, num_merged)."""
    clusters = _find_clusters(nodes)
    if not clusters:
        return nodes, 0
    by_id = {n["id"]: n for n in nodes}
    remap: dict[str, str] = {}
    losers_to_drop: set[str] = set()
    for group in clusters:
        winner = _winner([by_id[i] for i in group])
        losers = [by_id[i] for i in group if i != winner["id"]]
        _merge_into_winner(winner, losers)
        for o in losers:
            remap[o["id"]] = winner["id"]
            losers_to_drop.add(o["id"])

    surviving = [n for n in nodes if n["id"] not in losers_to_drop]
    for n in surviving:
        n["parent_edges"] = [
            e for e in _remap_edges(n["parent_edges"], remap) if e["id"] != n["id"]
        ]
        n["child_edges"] = [e for e in _remap_edges(n["child_edges"], remap) if e["id"] != n["id"]]
        n["influence_edges"] = [
            e for e in _remap_edges(n["influence_edges"], remap) if e["id"] != n["id"]
        ]
        n["parents"] = sorted({e["id"] for e in n["parent_edges"]})
        n["children"] = sorted({e["id"] for e in n["child_edges"]})
        n["influences"] = sorted({e["id"] for e in n["influence_edges"]})

    return surviving, len(losers_to_drop)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="inp", type=Path, default=Path("genres.json"))
    parser.add_argument("--out", dest="outp", type=Path, default=Path("genres.json"))
    args = parser.parse_args()
    payload = json.loads(args.inp.read_text())
    deduped, removed = dedupe(payload["nodes"])
    payload["nodes"] = deduped
    args.outp.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"[dedupe] merged {removed} alias duplicates -> {len(deduped)} nodes")


if __name__ == "__main__":
    main()
