"""The transcription pipeline — stages + orchestrator.

Stages (each a function with a clear input/output contract):

  ingest        validate an audio file, return AudioInput          (Q1)
  preprocess    normalize any supported format to 16 kHz mono WAV   (Q4)
  transcribe    speech -> text  (lives in transcribe.py)            (Q2)
  to_json/srt   timestamped output                                   (Q3)
  transcribe_long  chunk + offset-merge for long audio               (Q5)

`transcribe_file` wires the common case together in one call.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

from models import AudioInput, Segment, TranscriptionResult
from transcribe import Transcriber

# ================================================================== ingest
SUPPORTED_FORMATS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm", ".mp4"}
MAX_BYTES = 500 * 1024 * 1024  # 500 MB upload guard


class IngestError(Exception):
    """Raised when an audio file cannot be accepted."""


def ingest(raw_path: str) -> AudioInput:
    """Validate *raw_path* and return a typed AudioInput.

    Fails fast with a specific error so problems surface at the boundary,
    not three stages later inside the model.
    """
    path = Path(raw_path).resolve()

    if not path.exists():
        raise IngestError(f"File not found: {path}")
    if path.suffix.lower() not in SUPPORTED_FORMATS:
        raise IngestError(
            f"Unsupported format '{path.suffix}'. "
            f"Supported: {sorted(SUPPORTED_FORMATS)}"
        )
    size = path.stat().st_size
    if size == 0:
        raise IngestError("File is empty.")
    if size > MAX_BYTES:
        raise IngestError(
            f"File too large ({size / 1024 / 1024:.1f} MB); "
            f"limit is {MAX_BYTES // 1024 // 1024} MB."
        )
    return AudioInput(
        path=path,
        fmt=path.suffix.lower().lstrip("."),
        size_bytes=size,
        duration_sec=_probe_duration(path),
    )


def _probe_duration(path: Path) -> float | None:
    """Best-effort duration via ffprobe; None if ffprobe is unavailable."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_format", str(path)],
            capture_output=True, text=True, check=True,
        )
        return float(json.loads(out.stdout)["format"]["duration"])
    except (FileNotFoundError, subprocess.CalledProcessError, KeyError):
        return None


# ============================================================== preprocess
# Canonical form: 16 kHz mono 16-bit PCM WAV.
# Whisper is trained on 16 kHz audio, so this is both a format-handling AND
# an accuracy decision.
SAMPLE_RATE = 16000
CHANNELS = 1


class PreprocessError(Exception):
    """Raised when audio cannot be normalized to the canonical form."""


def preprocess(audio: AudioInput) -> Path:
    """Return a path to a 16 kHz mono WAV.

    Uses ffmpeg when available (handles every supported format). If ffmpeg is
    missing but the input is already canonical, passes straight through — this
    keeps the demo / CI path runnable without external binaries.
    """
    if shutil.which("ffmpeg"):
        return _convert_with_ffmpeg(audio.path)
    if _is_canonical_wav(audio.path):
        return audio.path
    raise PreprocessError(
        "ffmpeg is required to normalize this format but was not found. "
        "Install ffmpeg, or provide a 16 kHz mono WAV."
    )


def _convert_with_ffmpeg(src: Path) -> Path:
    out = Path(tempfile.mkdtemp()) / "normalized.wav"
    # -vn drops any video track (e.g. audio lifted from .webm / .mp4)
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-vn",
         "-ac", str(CHANNELS), "-ar", str(SAMPLE_RATE),
         "-sample_fmt", "s16", str(out)],
        check=True, capture_output=True,
    )
    return out


def _is_canonical_wav(path: Path) -> bool:
    try:
        with wave.open(str(path), "rb") as w:
            return (
                w.getframerate() == SAMPLE_RATE
                and w.getnchannels() == CHANNELS
                and w.getsampwidth() == 2  # 16-bit = 2 bytes/sample
            )
    except (wave.Error, OSError):
        return False


# ================================================================ format
def to_json(result: TranscriptionResult) -> str:
    """Structured JSON with per-segment timestamps — for APIs / databases."""
    return json.dumps(
        {
            "language": result.language,
            "text": result.text,
            "segments": [
                {"start": s.start, "end": s.end, "text": s.text}
                for s in result.segments
            ],
        },
        indent=2,
        ensure_ascii=False,
    )


def to_srt(result: TranscriptionResult) -> str:
    """SubRip subtitles, converting seconds to 00:00:00,000."""
    def fmt(seconds: float) -> str:
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        return f"{int(h):02d}:{int(m):02d}:{s:06.3f}".replace(".", ",")

    lines = []
    for i, seg in enumerate(result.segments, start=1):
        lines.append(str(i))
        lines.append(f"{fmt(seg.start)} --> {fmt(seg.end)}")
        lines.append(seg.text)
        lines.append("")  # blank line separates cue blocks
    return "\n".join(lines).strip()


def to_text(result: TranscriptionResult) -> str:
    """Plain transcript — for search indexing / display."""
    return result.text


FORMATTERS = {"json": to_json, "srt": to_srt, "text": to_text}


# ============================================================ long audio
# Whisper processes ~30s windows. For most files, faster-whisper's VAD splits
# and timestamps automatically. For very long files we split on silence with
# ffmpeg, transcribe each chunk, then merge with a cumulative time offset so
# timestamps stay correct across the whole file.
MAX_CHUNK_SECONDS = 25.0


def transcribe_long(
    audio_path: str,
    *,
    engine: Transcriber,
    language: str | None = None,
) -> TranscriptionResult:
    """Transcribe a long file: chunk -> transcribe each -> offset-merge."""
    chunks = split_on_silence(Path(audio_path), Path(tempfile.mkdtemp()))

    all_segments: list[Segment] = []
    full_text: list[str] = []
    offset = 0.0          # cumulative seconds already processed
    detected_lang = language or "en"

    for i, chunk in enumerate(chunks, start=1):
        # Progress reporting is trivial once you've chunked.
        print(f"  [chunk {i}/{len(chunks)}] transcribing...")
        result = engine.transcribe(str(chunk), language)

        # KEY STEP: each chunk's clock starts at 0, so shift by `offset` to
        # land on the correct timeline position in the original file.
        for seg in result.segments:
            all_segments.append(
                Segment(seg.start + offset, seg.end + offset, seg.text)
            )

        full_text.append(result.text)
        detected_lang = result.language
        offset += _audio_duration(chunk)   # advance the timeline

    return TranscriptionResult(
        text=" ".join(full_text).strip(),
        segments=all_segments,
        language=detected_lang,
    )


def split_on_silence(
    src: Path, out_dir: Path, min_silence_ms: int = 700
) -> list[Path]:
    """Split *src* at silence boundaries into ordered WAV chunks (<=25s)."""
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is required for silence-based chunking.")
    out_dir.mkdir(parents=True, exist_ok=True)

    silences = _silence_midpoints(src, min_silence_ms)
    boundaries = _build_boundaries(src, silences)

    chunks = []
    for i, (start, end) in enumerate(boundaries):
        chunk_path = out_dir / f"chunk_{i:03d}.wav"
        _ffmpeg_cut(src, start, end, chunk_path)
        chunks.append(chunk_path)
    return chunks or [src]


def _silence_midpoints(src: Path, min_silence_ms: int) -> list[float]:
    """Return silence-midpoint timestamps using ffmpeg's silencedetect."""
    proc = subprocess.run(
        ["ffmpeg", "-i", str(src), "-af",
         f"silencedetect=n=-30dB:d={min_silence_ms / 1000}",
         "-f", "null", "-"],
        capture_output=True, text=True,
    )
    starts = re.findall(r"silence_start: ([\d.]+)", proc.stderr)
    ends = re.findall(r"silence_end: ([\d.]+)", proc.stderr)
    # midpoint of each silence is a robust cut point
    return [(float(s) + float(e)) / 2 for s, e in zip(starts, ends)]


def _build_boundaries(
    src: Path, silences: list[float]
) -> list[tuple[float, float]]:
    """Turn silence points into (start, end) ranges, capped at MAX_CHUNK_SECONDS."""
    total = _audio_duration(src)
    edges = [0.0, *silences, total]
    ranges = []
    for start, stop in zip(edges[:-1], edges[1:]):
        cur = start
        while stop - cur > MAX_CHUNK_SECONDS:   # sub-split long continuous speech
            ranges.append((cur, cur + MAX_CHUNK_SECONDS))
            cur += MAX_CHUNK_SECONDS
        if stop > cur:
            ranges.append((cur, stop))
    return ranges


def _ffmpeg_cut(src: Path, start: float, end: float, out: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-ss", str(start), "-to", str(end),
         "-vn", "-ac", "1", "-ar", "16000", "-sample_fmt", "s16", str(out)],
        check=True, capture_output=True,
    )


def _audio_duration(path: Path) -> float:
    """Duration in seconds; stdlib for WAV, ffprobe otherwise."""
    try:
        with wave.open(str(path), "rb") as w:
            return w.getnframes() / w.getframerate()
    except wave.Error:
        pass
    proc = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(proc.stdout.strip())


# ============================================================ orchestrator
def transcribe_file(
    raw_path: str,
    *,
    engine: Transcriber,
    language: str | None = None,
    fmt: str = "json",
) -> str:
    """Run the standard pipeline and return the result as a string.

    Wires Q1 (accept) -> Q4 (normalize) -> Q2 (transcribe) -> Q3 (format).
    """
    audio = ingest(raw_path)
    wav = preprocess(audio)
    result = engine.transcribe(str(wav), language)
    return FORMATTERS[fmt](result)


def transcribe_file_structured(
    raw_path: str,
    *,
    engine: Transcriber,
    language: str | None = None,
) -> TranscriptionResult:
    """Same as transcribe_file but returns the structured result object."""
    audio = ingest(raw_path)
    wav = preprocess(audio)
    return engine.transcribe(str(wav), language)
