from __future__ import annotations

from pydantic import BaseModel, Field


class ExampleTrack(BaseModel):
    artist: str
    title: str
    year: int | None = None


class GeminiGenreResponse(BaseModel):
    """Schema Gemini fills for one expand-node call."""

    name: str
    aliases: list[str] = Field(default_factory=list)
    parents: list[str] = Field(
        default_factory=list,
        description="Direct parent genre names (one hop back, not grandparents).",
    )
    children: list[str] = Field(
        default_factory=list,
        description="Direct child genre names (one hop forward).",
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
