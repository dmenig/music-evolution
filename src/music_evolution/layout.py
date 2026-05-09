from __future__ import annotations

"""Static layout: precompute (x_norm, y_norm) and a virtual canvas aspect for
each node so the renderer can place them in O(1). All in [0, 1] coords; the
template multiplies by viewport size."""

import math
from collections import defaultdict

# Virtual canvas tuned so that on a typical viewport (>=1100×700) the fit-zoom
# scale is >=1, which means bubbles drawn at screen-constant size will have
# enough world spacing to avoid overlap at every zoom-in level.
CANVAS_W = 1100.0
CANVAS_H = 650.0

# Bubble radius formula must match the template's `radius(d)`.
RADIUS_BASE = 18.0
RADIUS_PER_SCORE = 30.0
INFLUENCE_WEIGHT = 0.85
SIZE_MULTIPLIER = 1.4
BIN_YEARS = 5


def _radius(node: dict) -> float:
    pop = node.get("popularity", 5)
    kids = len(node.get("children", []))
    score = pop + INFLUENCE_WEIGHT * kids
    area = RADIUS_BASE + RADIUS_PER_SCORE * score
    return max(2.5, math.sqrt(area / math.pi)) * SIZE_MULTIPLIER


def _build_cdf(nodes: list[dict]) -> tuple[list[tuple[int, float, float]], float]:
    """Density-driven X scale: empty centuries collapse, dense decades stretch.
    Returns list of (year, cum_start, cum_end) bins and total cumulative."""
    counts: dict[int, int] = defaultdict(int)
    for n in nodes:
        b = (n["peak_year"] // BIN_YEARS) * BIN_YEARS
        counts[b] += 1
    if not nodes:
        return [], 1.0
    lo = (min(n["peak_year"] for n in nodes) // BIN_YEARS) * BIN_YEARS
    hi = (max(n["peak_year"] for n in nodes) // BIN_YEARS + 1) * BIN_YEARS
    bins: list[tuple[int, float, float]] = []
    cum = 0.0
    y = lo
    while y <= hi:
        c = counts.get(y, 0)
        weight = (1.0 + c**0.7) if c > 0 else 0.01
        bins.append((y, cum, cum + weight))
        cum += weight
        y += BIN_YEARS
    return bins, cum or 1.0


def _cdf_at(year: int, bins: list[tuple[int, float, float]], total: float) -> float:
    if not bins:
        return 0.0
    if year <= bins[0][0]:
        return 0.0
    if year >= bins[-1][0] + BIN_YEARS:
        return 1.0
    for start, cs, ce in bins:
        end = start + BIN_YEARS
        if start <= year <= end:
            t = (year - start) / BIN_YEARS
            return (cs + t * (ce - cs)) / total
    return 1.0


def _x_jitter(nodes: list[dict]) -> dict[str, float]:
    """Spread same-year ties; ±20y pre-1900, ±5 to 1950, ±0.5 modern."""
    groups: dict[int, list[dict]] = defaultdict(list)
    for n in nodes:
        groups[n["peak_year"]].append(n)
    jit: dict[str, float] = {}
    for year, group in groups.items():
        group.sort(key=lambda x: x["id"])
        span = 20.0 if year < 1900 else (5.0 if year < 1950 else 0.5)
        n = len(group)
        for i, node in enumerate(group):
            t = 0 if n == 1 else (i / (n - 1) - 0.5) * 2
            jit[node["id"]] = t * span
    return jit


def _x_world(node: dict, jit: dict[str, float], bins, total: float) -> float:
    year = node["peak_year"] + jit.get(node["id"], 0.0)
    return CANVAS_W * _cdf_at(year, bins, total)


def _initial_y_targets(nodes: list[dict]) -> dict[str, float]:
    """Distribute ranks evenly within each X column so we don't start clumped."""
    by_id = {n["id"]: n for n in nodes}
    ranked = sorted(nodes, key=lambda n: (n["peak_year"], n.get("y", 0.5)))
    cols: dict[int, list[str]] = defaultdict(list)
    for n in ranked:
        col = n["peak_year"] // 10
        cols[col].append(n["id"])
    target: dict[str, float] = {}
    for col, ids in cols.items():
        m = len(ids)
        for i, nid in enumerate(ids):
            t = 0.5 if m == 1 else i / (m - 1)
            # Bias toward postprocess-assigned y if present, mixed with column rank.
            base = by_id[nid].get("y", 0.5) or 0.5
            target[nid] = (0.6 * t + 0.4 * base) * CANVAS_H
    return target


def _relax_y(nodes: list[dict], xs: dict[str, float], y_target: dict[str, float]) -> dict[str, float]:
    """Bucket by X column, push pairs apart by collision radius."""
    y: dict[str, float] = dict(y_target)
    # Bubbles render at constant screen size; layout uses the true screen
    # radius. With CANVAS sized below typical viewport, fit-zoom k>=1 so
    # screen spacing = world_spacing * k >= 2*radius. No overlap at fit or
    # any zoom-in.
    radii = {n["id"]: _radius(n) for n in nodes}
    cols = 320
    x_min = min(xs.values())
    x_max = max(xs.values())
    col_w = (x_max - x_min) / cols or 1.0
    pad = 4.0
    # Allow Y to escape the canvas during relaxation; clamp at the end.
    for _ in range(110):
        buckets: dict[int, list[str]] = defaultdict(list)
        for n in nodes:
            i = max(0, min(cols, int((xs[n["id"]] - x_min) / col_w)))
            buckets[i].append(n["id"])
        for ids in buckets.values():
            ids.sort(key=lambda i: y[i])
            for i in range(1, len(ids)):
                a, b = ids[i - 1], ids[i]
                min_gap = radii[a] + radii[b] + pad
                gap = y[b] - y[a]
                if gap < min_gap:
                    push = (min_gap - gap) / 2
                    y[a] -= push
                    y[b] += push
        for nid in y:
            y[nid] += (y_target[nid] - y[nid]) * 0.015
    # Renormalize Y to [0, CANVAS_H]: span the full canvas.
    y_min = min(y.values())
    y_max = max(y.values())
    span = (y_max - y_min) or 1.0
    for nid in y:
        y[nid] = 30.0 + (CANVAS_H - 80.0) * (y[nid] - y_min) / span
    return y


def assign_layout(nodes: list[dict]) -> None:
    """Mutates nodes in place: writes node['x_norm'], node['y_norm'], node['radius']."""
    bins, total = _build_cdf(nodes)
    jit = _x_jitter(nodes)
    xs = {n["id"]: _x_world(n, jit, bins, total) for n in nodes}
    y_target = _initial_y_targets(nodes)
    ys = _relax_y(nodes, xs, y_target)
    for n in nodes:
        n["x_norm"] = xs[n["id"]] / CANVAS_W
        n["y_norm"] = ys[n["id"]] / CANVAS_H
        n["radius"] = _radius(n)
