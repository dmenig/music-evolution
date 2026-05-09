from __future__ import annotations

from slugify import slugify


def canonical_id(name: str) -> str:
    """Stable slug for a genre name. Idempotent."""
    return slugify(name.strip().lower(), separator="-", lowercase=True)


def merge_alias(aliases: dict[str, str], raw: str, canonical: str) -> None:
    """Record that `raw` resolves to `canonical`."""
    key = canonical_id(raw)
    if key and key != canonical:
        aliases[key] = canonical


def resolve(aliases: dict[str, str], raw: str) -> str:
    """Return canonical id for any raw name, following alias chains."""
    cid = canonical_id(raw)
    seen: set[str] = set()
    while cid in aliases and cid not in seen:
        seen.add(cid)
        cid = aliases[cid]
    return cid
