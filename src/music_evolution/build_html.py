from __future__ import annotations

import argparse
import json
from pathlib import Path

TEMPLATE_PATH = Path(__file__).parent / "template.html"


def build(data_path: Path, out_path: Path) -> None:
    payload = json.loads(data_path.read_text())
    template = TEMPLATE_PATH.read_text()
    # JSON.parse('...string...') has a fast path in V8 that's ~2x faster than
    # parsing the equivalent JS object literal for large payloads. We dump the
    # JSON twice: inner = the data; outer = a JS string literal containing it.
    payload_js = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    inlined_expr = f"JSON.parse({json.dumps(payload_js)})"
    inlined = template.replace("/*__GENRES_DATA__*/null", inlined_expr)
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
