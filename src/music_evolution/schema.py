from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

EdgeStrength = Literal["minor", "mid", "major"]


class ExampleTrack(BaseModel):
    artist: str
    title: str
    year: int | None = None
    preview_url: str | None = None
    preview_provider: str | None = None


class GenreEdge(BaseModel):
    """LLM-asserted link to another genre, with self-rated influence weight."""

    name: str
    description: str = Field(
        description="One-sentence factual description of the linked genre as understood here."
    )
    strength: EdgeStrength = Field(
        description="minor = peripheral influence; mid = clear lineage; major = defining ancestry."
    )


class StoredEdge(BaseModel):
    """Edge as written to genres.json after canonicalization + symmetric averaging."""

    id: str
    name: str
    description: str
    strength: EdgeStrength


class GeminiGenreResponse(BaseModel):
    """Schema Gemini fills for one expand-node call."""

    name: str
    aliases: list[str] = Field(default_factory=list)
    parents: list[GenreEdge] = Field(
        default_factory=list,
        description="Direct parent genres (one hop back, not grandparents).",
    )
    children: list[GenreEdge] = Field(
        default_factory=list,
        description="Direct child genres (one hop forward).",
    )
    birth_year: int
    death_year: int | None = None
    peak_year: int
    popularity: float = Field(
        ge=0.0,
        le=100.0,
        description="% of music-listening humans alive at peak_year who were fans/practitioners.",
    )
    countries: list[str] = Field(
        default_factory=list,
        description="ISO-3166 alpha-2 codes, primary origin first.",
    )
    artists: list[str] = Field(default_factory=list)
    examples: list[ExampleTrack] = Field(default_factory=list)
    description: str
    sources: list[str] = Field(default_factory=list)


class GenreNode(BaseModel):
    """Canonical node stored in genres.json."""

    id: str
    name: str
    aliases: list[str]
    parents: list[str]
    children: list[str]
    parent_edges: list[StoredEdge] = Field(default_factory=list)
    child_edges: list[StoredEdge] = Field(default_factory=list)
    influence_edges: list[StoredEdge] = Field(default_factory=list)
    birth_year: int
    death_year: int | None
    peak_year: int
    popularity: float
    countries: list[str]
    artists: list[str]
    examples: list[ExampleTrack]
    description: str
    sources: list[str]
    influences: list[str] = Field(default_factory=list)
    color: str | None = None
    depth: int = 0
