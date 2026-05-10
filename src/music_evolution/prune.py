"""Drop nodes that aren't actually genres (decades, overview articles, lists)
and cascade-remove any edges pointing to them."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

NON_GENRE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^\d{3,4}s? in ", re.IGNORECASE),
    re.compile(r"^\d{3,4}s? music$", re.IGNORECASE),
    re.compile(r"^Music of ", re.IGNORECASE),
    re.compile(r"^Music in ", re.IGNORECASE),
    re.compile(r"^Music from ", re.IGNORECASE),
    re.compile(r"^History of ", re.IGNORECASE),
    re.compile(r"^Timeline of ", re.IGNORECASE),
    re.compile(r"^Lists? of ", re.IGNORECASE),
    re.compile(r"^Outline of ", re.IGNORECASE),
    re.compile(r"^Glossary of ", re.IGNORECASE),
    re.compile(r"^Index of ", re.IGNORECASE),
    re.compile(r"^Styles of ", re.IGNORECASE),
    re.compile(r"^Category:", re.IGNORECASE),
]


def is_non_genre(name: str) -> bool:
    return any(p.search(name) for p in NON_GENRE_PATTERNS)


def prune(nodes: list[dict]) -> tuple[list[dict], int]:
    bad_ids = {n["id"] for n in nodes if is_non_genre(n["name"])}
    if not bad_ids:
        return nodes, 0
    surviving = [n for n in nodes if n["id"] not in bad_ids]
    for n in surviving:
        n["parent_edges"] = [e for e in n.get("parent_edges", []) if e["id"] not in bad_ids]
        n["child_edges"] = [e for e in n.get("child_edges", []) if e["id"] not in bad_ids]
        n["influence_edges"] = [e for e in n.get("influence_edges", []) if e["id"] not in bad_ids]
        n["parents"] = sorted({e["id"] for e in n["parent_edges"]})
        n["children"] = sorted({e["id"] for e in n["child_edges"]})
        n["influences"] = sorted({e["id"] for e in n["influence_edges"]})
    return surviving, len(bad_ids)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="inp", type=Path, default=Path("genres.json"))
    parser.add_argument("--out", dest="outp", type=Path, default=Path("genres.json"))
    args = parser.parse_args()
    payload = json.loads(args.inp.read_text())
    pruned, removed = prune(payload["nodes"])
    payload["nodes"] = pruned
    args.outp.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"[prune] dropped {removed} non-genre nodes -> {len(pruned)} nodes")


if __name__ == "__main__":
    main()
