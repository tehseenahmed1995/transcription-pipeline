"""Shared data models for the transcription pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Segment:
    """One time-stamped chunk of transcript.

    Times are in seconds from the start of the source file.
    """
    start: float
    end: float
    text: str


@dataclass
class TranscriptionResult:
    """Full output of a transcription run."""
    text: str                 # full concatenated transcript
    segments: list[Segment]   # time-stamped segments
    language: str              # detected language code, e.g. "en"


@dataclass
class AudioInput:
    """Validated audio file descriptor produced by the ingest stage.

    Returning a typed object (instead of a raw string) means downstream
    stages never re-stat the file and always have the metadata they need.
    """
    path: Path
    fmt: str            # normalized extension, e.g. "mp3"
    size_bytes: int
    duration_sec: float | None = None
