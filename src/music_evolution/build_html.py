from __future__ import annotations

import argparse
import json
from pathlib import Path

TEMPLATE_PATH = Path(__file__).parent / "template.html"


def build(data_path: Path, out_path: Path) -> None:
    payload = json.loads(data_path.read_text())
    template = TEMPLATE_PATH.read_text()
    inlined = template.replace(
        "/*__GENRES_DATA__*/null",
        json.dumps(payload, ensure_ascii=False),
    )
    out_path.write_text(inlined)
    print(f"[build] {len(payload['nodes'])} nodes -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("genres.json"))
    parser.add_argument("--out", type=Path, default=Path("index.html"))
    args = parser.parse_args()
    build(args.data, args.out)


if __name__ == "__main__":
    main()
