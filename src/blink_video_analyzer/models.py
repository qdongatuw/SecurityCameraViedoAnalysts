from __future__ import annotations

from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field


class VideoMetadata(BaseModel):
    path: Path
    video_name: str
    modified_time: datetime
    duration_seconds: float | None = None
    frame_count: int | None = None
    fps: float | None = None


class VideoAnalysis(BaseModel):
    video_name: str
    video_path: str
    modified_time: datetime
    duration_seconds: float | None = None
    motion_objects: list[str] = Field(default_factory=list)
    animal_species: list[str] = Field(default_factory=list)
    scene: str = ""
    description: str = ""
    confidence: float = 0.0
    evidence_frames: list[str] = Field(default_factory=list)
    raw_model_output: str | None = None
    error: str | None = None

