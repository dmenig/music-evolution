from __future__ import annotations

"""Build a genres.json from the live data/visited.json without stopping the crawl."""

import json
import time
from pathlib import Path

from .crawler import VISITED_FILE, stitch_and_finalize, write_genres_json
from .schema import GenreNode


def main() -> None:
    if not VISITED_FILE.exists():
        print("no visited.json yet")
        return
    raw = json.loads(VISITED_FILE.read_text())
    nodes = {k: GenreNode(**v) for k, v in raw.items()}
    nodes = stitch_and_finalize(nodes)
    out = Path("genres.json")
    write_genres_json(nodes, out)
    print(f"[snapshot] {len(nodes)} nodes at {time.strftime('%H:%M:%S')}")


if __name__ == "__main__":
    main()
