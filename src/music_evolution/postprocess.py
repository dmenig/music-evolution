from __future__ import annotations

import colorsys
import hashlib
import json
import math
from pathlib import Path

# Anchor hues (degrees) for major root families. Roots not listed get auto-spaced.
ROOT_HUE_OVERRIDES: dict[str, float] = {
    "folk-music": 30,
    "western-classical-music": 250,
    "gregorian-chant": 280,
    "religious-chant": 290,
    "work-song": 50,
    "sea-shanty": 200,
    "court-music": 220,
    "military-music": 0,
    "indigenous-music": 130,
    "hindustani-classical-music": 320,
    "carnatic-music": 340,
    "gagaku": 100,
    "gamelan": 160,
    "andalusian-classical-music": 70,
    "persian-traditional-music": 20,
    "chinese-traditional-music": 180,
    "arabic-maqam": 10,
    "african-traditional-music": 60,
    "native-american-music": 120,
    "polynesian-music": 170,
    "throat-singing": 240,
    "klezmer": 270,
    "flamenco": 350,
    "fado": 310,
}

JITTER_DEG = 18.0
DRIFT_CAP_DEG = 140.0
REANCHOR_EVERY = 4


def _seeded_jitter(node_id: str) -> float:
    h = hashlib.sha256(node_id.encode()).digest()
    n = int.from_bytes(h[:4], "big") / 0xFFFFFFFF
    return (n - 0.5) * 2 * JITTER_DEG


def _hue_to_hex(hue: float, sat: float, light: float) -> str:
    r, g, b = colorsys.hls_to_rgb((hue % 360) / 360.0, light, sat)
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"


def _circular_mean(degrees: list[float], weights: list[float]) -> float:
    x = sum(w * math.cos(math.radians(d)) for d, w in zip(degrees, weights, strict=True))
    y = sum(w * math.sin(math.radians(d)) for d, w in zip(degrees, weights, strict=True))
    return math.degrees(math.atan2(y, x)) % 360


def _root_hue(node_id: str, idx: int, n_roots: int) -> float:
    if node_id in ROOT_HUE_OVERRIDES:
        return ROOT_HUE_OVERRIDES[node_id]
    # Hash-based, well-distributed across the wheel; offset by golden angle
    # so consecutive ids are visually distant.
    h = hashlib.sha256(node_id.encode()).digest()
    base = int.from_bytes(h[:4], "big") / 0xFFFFFFFF * 360
    return (base + idx * 137.508) % 360


PARENT_PULL = 0.45  # 0 = own hash-hue only, 1 = full inherit (collapses lineages)


def _own_hue(node_id: str) -> float:
    h = hashlib.sha256(("hue:" + node_id).encode()).digest()
    return int.from_bytes(h[:4], "big") / 0xFFFFFFFF * 360.0


def _blend_hue(parent_mean: float, own: float, pull: float) -> float:
    """Circular interpolation from `own` toward `parent_mean` by `pull`."""
    delta = ((parent_mean - own + 180) % 360) - 180
    return (own + pull * delta) % 360


def assign_colors(nodes: list[dict]) -> None:
    by_id = {n["id"]: n for n in nodes}
    roots = [n for n in nodes if not n["parents"]]
    base: dict[str, float] = {}
    depth: dict[str, int] = {}

    for i, r in enumerate(roots):
        base[r["id"]] = _root_hue(r["id"], i, len(roots))
        depth[r["id"]] = 0

    order = _topo_order(nodes)
    for nid in order:
        if nid in base:
            continue
        n = by_id[nid]
        parents = [p for p in n["parents"] if p in base]
        if not parents:
            base[nid] = _own_hue(nid)
            depth[nid] = 0
            continue
        weights = [max(1, by_id[p].get("popularity", 5)) for p in parents]
        hues = [base[p] for p in parents]
        parent_mean = _circular_mean(hues, weights)
        own = _own_hue(nid)
        # Strong pull toward parent so visual lineage is preserved, but the
        # node's own hash adds enough deviation to keep deep lineages varied.
        pull = PARENT_PULL if len(parents) == 1 else min(0.85, PARENT_PULL + 0.15)
        base[nid] = _blend_hue(parent_mean, own, pull)
        depth[nid] = max(depth[p] for p in parents) + 1

    for n in nodes:
        hue = base[n["id"]]
        d = depth.get(n["id"], 0)
        sat = 0.62 + (0.08 if len(n["parents"]) >= 2 else 0.0)
        light = 0.55 + 0.04 * ((d % REANCHOR_EVERY) - 1.5)
        n["color"] = _hue_to_hex(hue, sat, light)
        n["depth"] = d


def _topo_order(nodes: list[dict]) -> list[str]:
    by_id = {n["id"]: n for n in nodes}
    indeg = {nid: 0 for nid in by_id}
    for n in nodes:
        for p in n["parents"]:
            if p in by_id:
                indeg[n["id"]] += 1
    queue = [nid for nid, d in indeg.items() if d == 0]
    out: list[str] = []
    seen: set[str] = set()
    while queue:
        nid = queue.pop(0)
        if nid in seen:
            continue
        seen.add(nid)
        out.append(nid)
        for n in nodes:
            if nid in n["parents"]:
                indeg[n["id"]] -= 1
                if indeg[n["id"]] == 0:
                    queue.append(n["id"])
    for nid in by_id:
        if nid not in seen:
            out.append(nid)
    return out


def assign_y_positions(nodes: list[dict]) -> None:
    """Assign a stable y in [0, 1] via DFS post-order rank, normalized at the end."""
    by_id = {n["id"]: n for n in nodes}
    roots = [n["id"] for n in nodes if not n["parents"]]
    roots.sort(key=lambda i: ROOT_HUE_OVERRIDES.get(i, 999))

    raw: dict[str, float] = {}
    cursor = [0.0]

    def place(nid: str, on_path: set[str]) -> None:
        if nid in raw or nid in on_path:
            return
        on_path = on_path | {nid}
        kids = [c for c in by_id[nid]["children"] if c in by_id and c not in raw]
        if not kids:
            raw[nid] = cursor[0]
            cursor[0] += 1
            return
        first = cursor[0]
        for c in kids:
            place(c, on_path)
        if cursor[0] > first:
            raw[nid] = (first + cursor[0] - 1) / 2
        else:
            raw[nid] = cursor[0]
            cursor[0] += 1

    for r in roots:
        place(r, set())
    for nid in by_id:
        if nid not in raw:
            raw[nid] = cursor[0]
            cursor[0] += 1

    # Interleave the lineage rank with a per-node birth-year offset so modern
    # genres of every lineage spread vertically instead of clumping by parent.
    span = max(raw.values()) or 1
    by_year: dict[int, list[str]] = {}
    for nid in raw:
        by_year.setdefault(by_id[nid]["birth_year"], []).append(nid)
    for nid, r in raw.items():
        n = by_id[nid]
        peers = by_year[n["birth_year"]]
        peers.sort()
        idx = peers.index(nid)
        # Shift up to ±0.15 of full span based on peer index, evenly spread.
        if len(peers) > 1:
            offset = (idx / (len(peers) - 1) - 0.5) * 0.30 * span
            r = r + offset
        n["y"] = max(0.0, min(1.0, r / span))


def postprocess(genres_path: Path, out_path: Path) -> None:
    from .layout import assign_layout

    payload = json.loads(genres_path.read_text())
    nodes = payload["nodes"]
    assign_colors(nodes)
    assign_y_positions(nodes)
    assign_layout(nodes)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def main() -> None:
    postprocess(Path("genres.json"), Path("genres.json"))


if __name__ == "__main__":
    main()
