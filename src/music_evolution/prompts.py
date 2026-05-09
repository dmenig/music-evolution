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
- popularity = a float in [0.0, 100.0]. It is the estimated PERCENTAGE of ALL music-listening humans alive ANYWHERE IN THE WORLD at this genre's peak_year who had this specific genre as a meaningful part of their listening habits or cultural identity. The denominator is the entire global music-listening population at peak_year — NOT just the genre's region, NOT just the country of origin, NOT just the cultural sphere it belonged to. A genre that dominated Europe but was absent elsewhere has a much smaller world-share than its regional share. A genre dominant only in one country has world-share roughly = (that country's listeners / world's listeners). Overlap allowed: a person who listened to both Rock and Pop is counted toward both. Use decimals freely (0.0001, 0.003, 2.5, 17.3, etc.). It is NOT a 1-10 score, NOT a normalized rank, NOT a vibes number — it is a real-world estimate of global cultural footprint.

REASONING: world population at peak_year × ~all-of-them-listened-to-music × (geographic reach of this genre / world) × (within-region adoption rate). For 1400 (~400M humans), even a "huge" Christendom-only tradition tops out around 15-20% world-share because Asia/Africa/Americas had their own traditions. For 2000 (~6B humans, all listen to music), a planet-wide hit can reach 30-40%.

ANCHORS (calibrated to the WORLD denominator)
  * Ancient Indigenous Music (peak ~1400) ≈ 8-15 (one of many parallel traditions worldwide)
  * Religious Chant (medieval peak) ≈ 10-18 (Christendom + parts of Islamic/Buddhist world, not universal)
  * Folk Music (Western/American narrow sense, peak ~1965 revival) ≈ 2-5
  * Western Classical (peak ~1880) ≈ 4-8 (Europe + colonies + concert circuits)
  * Rock (peak ~1975) ≈ 22-30 (truly global)
  * Pop (peak ~2000) ≈ 28-38
  * Hip-Hop (peak ~2015) ≈ 18-26
  * Jazz (peak ~1950) ≈ 6-10
  * Bebop (peak ~1955) ≈ 0.5-1.5
  * Disco (peak ~1978) ≈ 6-12
  * Reggaeton (peak ~2020) ≈ 4-8
  * Vaporwave (peak ~2015) ≈ 0.02-0.1
  * Witch House (peak ~2012) ≈ 0.005-0.02
  * Single-scene microgenre with one club night ≈ 0.0001-0.001

Critical: a regional tradition that "everyone in its region listened to" is still a small fraction of world music listeners if that region was a small slice of humanity. Conversely, a 20th-century mainstream genre can score higher than any pre-modern tradition because radio + recordings let it cross continents. Do not inflate ancient/medieval scores above their realistic global share — but also do not crush them to zero, since pre-modern listeners did concentrate on a small handful of traditions.

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
