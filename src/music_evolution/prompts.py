from __future__ import annotations

EXPAND_NODE_SYSTEM = (
    "You are a precise musicologist building an exhaustive genre lineage graph. "
    "You always answer with the structured JSON schema requested. "
    "You do not speculate beyond documented genres; if a 'genre' is just one artist's "
    "private label or a marketing tag with no scene, omit it."
)

EXPAND_NODE_INSTRUCTIONS = """For the music genre "{name}", return its lineage and metadata.

DEFINITIONS
- Direct parents = genres whose practitioners and forms were directly incorporated when this genre coalesced. NOT grandparents. NOT broad ancestors. One hop back only.
- Direct children = genres that explicitly emerged out of this one. NOT merely contemporaries. One hop forward only.
- birth_year = earliest year a recognizable form of this genre existed.
- death_year = year the genre ceased to be actively produced as a living tradition. null if still alive in 2026. A genre fully absorbed into a child but no longer practiced standalone counts as dead.
- peak_year = year of greatest cultural reach.
- popularity (1-10), global all-time scale: Western Classical = 10, Rock = 10, Hip-Hop = 10, Jazz = 9, Bebop = 7, Vaporwave = 3, Witch House = 2, a single-scene microgenre = 1.

REQUIREMENTS
- 3-8 representative artists, prioritizing originators.
- 3-6 emblematic examples (artist + title + year).
- Up to 5 ISO-3166 alpha-2 country codes, primary origin first.
- Neutral description, max 280 chars.
- Up to 3 aliases (alternate names in common use).

EXHAUSTIVENESS
Be exhaustive on children. Include regional, fusion, micro-, and short-lived scenes. For House, this means listing things like Acid House, Deep House, Chicago House, Hardbag, Ghetto House, Tech House, Future House, Microhouse, Ambient House, Tribal House, French House, Latin House, Funky House, Electro House, Progressive House, Bass House, etc. For metal, include Death Metal, Black Metal, Doom, Power, Folk Metal, Pirate Metal, Symphonic Black Metal, Blackgaze, Atmospheric Sludge, etc. If a child has multiple parents, list this genre as one parent (Gemini does not need to name the others here; downstream calls will catch them).

Do not invent genres. If a candidate has fewer than 5 documented releases or no recognizable scene, omit it.
"""


def render_expand_prompt(name: str) -> str:
    return EXPAND_NODE_INSTRUCTIONS.format(name=name)
