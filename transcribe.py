"""Stage 3 — Transcribe speech to text.

The transcriber sits behind an interface so the engine is swappable:
  - WhisperTranscriber uses faster-whisper (CTranslate2 backend; lighter +
    faster than the reference openai-whisper, same model weights).
  - MockTranscriber returns canned output — no model install required.

The pipeline depends on the interface, never a concrete engine, so we can
unit-test every stage without torch and swap in a cloud API later.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from models import Segment, TranscriptionResult


class Transcriber(ABC):
    """Interface every speech-to-text engine implements."""

    @abstractmethod
    def transcribe(
        self,
        wav_path: str,
        language: str | None = None,
    ) -> TranscriptionResult:
        ...


class WhisperTranscriber(Transcriber):
    """Real engine using faster-whisper.

    Heavy imports (torch / faster_whisper) happen only when an instance is
    created, so importing this module never forces a giant install.
    """

    def __init__(self, model_size: str = "base"):
        from faster_whisper import WhisperModel

        self._model = WhisperModel(model_size, compute_type="int8")

    def transcribe(self, wav_path: str, language: str | None = None):
        segments, info = self._model.transcribe(
            wav_path,
            language=language,
            vad_filter=True,   # skip silence via voice-activity detection
            beam_size=5,
        )
        segs = [Segment(s.start, s.end, s.text.strip()) for s in segments]
        full_text = " ".join(s.text for s in segs).strip()
        return TranscriptionResult(
            text=full_text,
            segments=segs,
            language=info.language,
        )


class MockTranscriber(Transcriber):
    """Deterministic stub for tests / CI / offline demos — zero dependencies."""

    def transcribe(self, wav_path: str, language: str | None = None):
        return TranscriptionResult(
            text="This is a sample transcription of the audio file.",
            segments=[
                Segment(0.0, 2.5, "This is a sample transcription"),
                Segment(2.5, 4.2, "of the audio file."),
            ],
            language=language or "en",
        )
