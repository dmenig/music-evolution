from __future__ import annotations

from dataclasses import dataclass

EXPAND_NODE_SYSTEM = (
    "You are a precise musicologist building an exhaustive genre lineage graph. "
    "You always answer with the structured JSON schema requested. "
    "You do not speculate beyond documented genres; if a 'genre' is just one artist's "
    "private label or a marketing tag with no scene, omit it. "
    "If the input is NOT a music genre at all (a country/region overview like "
    "'Music of Japan', a decade like '2020s in music', a list/timeline/glossary, "
    "an instrument, a person, an album, a song, a venue, a label, an event, or any "
    "other non-genre topic), you must refuse: return empty parents and empty children, "
    "set popularity to 0, and write a description that explicitly states this is not "
    "a genre. Do not invent lineage for non-genres. "
    "Every parent and child you list must be tagged with an influence strength "
    "(minor | mid | major) and a one-sentence factual description of the linked genre. "
    "EVERY claim — birth year, lineage edges, peak, popularity, artists — must be one a "
    "knowledgeable musicologist would accept. Do not assert connections that experts "
    "would dismiss as wrong. When uncertain, omit the edge or downgrade strength to minor. "
    "Cite at least 2 verifiable sources (Wikipedia article titles, AllMusic/MusicBrainz "
    "IDs or URLs, named books, journals, or liner notes). If you cannot cite, omit the genre."
)

EXPAND_NODE_INSTRUCTIONS = """For the music genre "{name}", return its lineage and metadata.

{disambiguation_block}
DEFINITIONS
- Direct parents = genres whose practitioners and forms were directly incorporated when this genre coalesced. NOT grandparents. NOT broad ancestors. One hop back only.
- Direct children = genres that explicitly emerged out of this one. NOT merely contemporaries. One hop forward only.
- For each parent and child, you MUST return: name, a one-sentence factual description of THAT linked genre, and an influence strength.
- strength = minor | mid | major. minor = peripheral / one element borrowed; mid = clear lineage but mixed with other influences; major = defining ancestry, the genre would not exist without this link.
- birth_year = earliest year a recognizable form of this genre existed.
- death_year = year the genre ceased to be actively produced as a living tradition. null if still alive in 2026. A genre fully absorbed into a child but no longer practiced standalone counts as dead.
- peak_year = year of greatest cultural reach.
- popularity = a float in [0.0, 100.0]. It is the estimated PERCENTAGE of ALL music-listening humans alive ANYWHERE IN THE WORLD at this genre's peak_year who had this specific genre as a meaningful part of their listening habits or cultural identity. The denominator is the entire global music-listening population at peak_year — NOT just the genre's region, NOT just the country of origin, NOT just the cultural sphere it belonged to. A genre that dominated Europe but was absent elsewhere has a much smaller world-share than its regional share. A genre dominant only in one country has world-share roughly = (that country's listeners / world's listeners). Overlap allowed: a person who listened to both Rock and Pop is counted toward both. Use decimals freely (0.0001, 0.003, 2.5, 17.3, etc.). It is NOT a 1-10 score, NOT a normalized rank, NOT a vibes number — it is a real-world estimate of global cultural footprint.

REASONING: world population at peak_year × ~all-of-them-listened-to-music × (geographic reach of this genre / world) × (within-region adoption rate). For 1400 (~400M humans), even a "huge" Christendom-only tradition tops out around 15-20% world-share because Asia/Africa/Americas had their own traditions. For 2000 (~6B humans, all listen to music), a planet-wide hit can reach 30-40%.

ANCHORS (calibrated to the WORLD denominator)
  * Ancient Indigenous Music (peak ~1400) ≈ 8-15
  * Religious Chant (medieval peak) ≈ 10-18
  * Folk Music (Western/American narrow sense, peak ~1965) ≈ 2-5
  * Western Classical (peak ~1880) ≈ 4-8
  * Rock (peak ~1975) ≈ 22-30
  * Pop (peak ~2000) ≈ 28-38
  * Hip-Hop (peak ~2015) ≈ 18-26
  * Jazz (peak ~1950) ≈ 6-10
  * Bebop (peak ~1955) ≈ 0.5-1.5
  * Disco (peak ~1978) ≈ 6-12
  * Reggaeton (peak ~2020) ≈ 4-8
  * Vaporwave (peak ~2015) ≈ 0.02-0.1
  * Witch House (peak ~2012) ≈ 0.005-0.02
  * Single-scene microgenre with one club night ≈ 0.0001-0.001

REQUIREMENTS
- 3-8 representative artists, prioritizing originators.
- 3-6 emblematic examples (artist + title + year).
- Up to 5 ISO-3166 alpha-2 country codes, primary origin first.
- Neutral description, max 280 chars.
- Up to 3 aliases (alternate names in common use).
- sources: AT LEAST 2 verifiable references — Wikipedia article titles or URLs, AllMusic/MusicBrainz IDs or URLs, named books, journals, or liner notes. Generic claims ("various sources") are not acceptable.

DEFENSIBILITY (this is the hard bar)
Every parent/child link must be one a musicologist would accept on inspection. If you assert "Genre X is a parent of Genre Y", you must be able to point to documented practitioners who came from X and shaped Y, OR a recognized scholarly account of the lineage. Speculative "feels related" links are forbidden. Better to return 3 strong parents than 8 with two embarrassments. When in doubt:
  * Downgrade strength to minor instead of mid/major.
  * Or omit the edge entirely.
The graph is consumed by people who know music. Wrong edges are worse than missing edges.

EXHAUSTIVENESS
Be exhaustive on children. Include regional, fusion, micro-, and short-lived scenes. For House: Acid House, Deep House, Chicago House, Hardbag, Ghetto House, Tech House, Future House, Microhouse, Ambient House, Tribal House, French House, Latin House, Funky House, Electro House, Progressive House, Bass House, etc. For metal: Death, Black, Doom, Power, Folk, Pirate, Symphonic Black, Blackgaze, Atmospheric Sludge, etc.

DO NOT INVENT genres. If a candidate has fewer than 5 documented releases or no recognizable scene, omit it. If a name is ambiguous between multiple genres, anchor to the disambiguation block above and pick the genre that best fits that context — do not silently switch to a different homonym.

If you cannot meet the defensibility and sources bar for this genre at all, return empty parents and children rather than invent links. Empty edges are acceptable; wrong edges are not.
"""


@dataclass(slots=True, frozen=True)
class OriginatorContext:
    """An upstream node that referenced this genre as a parent or child."""

    upstream_name: str
    upstream_description: str
    direction: str  # "parent" or "child" — how the upstream node referred to this one
    asserted_strength: str
    asserted_description: str


def _disambiguation_block(
    wiki_description: str | None, originators: list[OriginatorContext]
) -> str:
    if not wiki_description and not originators:
        return ""
    parts: list[str] = ["DISAMBIGUATION CONTEXT (use this to anchor the exact genre we mean)"]
    if wiki_description:
        parts.append(f"Wikipedia summary: {wiki_description}")
    for o in originators[:6]:
        parts.append(
            f'Referenced by "{o.upstream_name}" as a {o.direction} '
            f"(strength={o.asserted_strength}). Upstream context: {o.upstream_description.strip()[:400]} "
            f"Upstream's one-line take on this genre: {o.asserted_description.strip()[:300]}"
        )
    parts.append("Anchor your answer to the genre described above. Do not switch to a homonym.\n")
    return "\n".join(parts) + "\n"


def render_expand_prompt(
    name: str,
    wiki_description: str | None = None,
    originators: list[OriginatorContext] | None = None,
) -> str:
    block = _disambiguation_block(wiki_description, originators or [])
    return EXPAND_NODE_INSTRUCTIONS.format(name=name, disambiguation_block=block)
