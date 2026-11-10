"""CLI + dependency-free demo.

  python main.py                       # demo with MockTranscriber (zero installs)
  python main.py audio.mp3 --whisper --fmt srt
  python main.py long.wav --whisper --long --fmt json
"""
from __future__ import annotations

import argparse
import math
import struct
import sys
import tempfile
import wave
from pathlib import Path

from pipeline import (
    IngestError,
    ingest,
    preprocess,
    to_json,
    to_srt,
    to_text,
    transcribe_file,
    transcribe_long,
)
from transcribe import MockTranscriber, WhisperTranscriber


def make_sample_wav(out: Path, seconds: float = 4.0, rate: int = 16000) -> None:
    """Generate a short 16 kHz mono WAV (a 440 Hz tone) using only stdlib."""
    with wave.open(str(out), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        frames = b"".join(
            struct.pack("<h", int(32767 * math.sin(2 * math.pi * 440 * t / rate)))
            for t in range(int(seconds * rate))
        )
        w.writeframes(frames)


def run_demo() -> None:
    """Demonstrate Q1-Q5 against a generated sample, no installs needed."""
    sample = Path(tempfile.mkdtemp()) / "sample.wav"
    make_sample_wav(sample)
    print(f"[demo] generated sample audio: {sample}\n")

    engine = MockTranscriber()

    # --- Q1: accept the file ---
    audio = ingest(str(sample))
    print(f"[Q1 ingest] accepted {audio.fmt} | {audio.size_bytes} B | "
          f"duration={audio.duration_sec}")

    # --- Q4: normalize the format ---
    canonical = preprocess(audio)
    print(f"[Q4 preprocess] canonical 16 kHz mono WAV -> {canonical}")

    # --- Q2: transcribe speech -> text ---
    result = engine.transcribe(str(canonical))
    print(f"[Q2 transcribe] language={result.language} | "
          f"{len(result.segments)} segments | text={result.text!r}")

    # --- Q3: timestamped output in three formats ---
    print("\n--- Q3 JSON ---")
    print(to_json(result))
    print("\n--- Q3 SRT ---")
    print(to_srt(result))
    print("\n--- Q3 TEXT ---")
    print(to_text(result))

    # --- Q5: long-audio chunking + offset merge ---
    print("\n[Q5 long audio] running chunked transcribe on the same file...")
    try:
        long_result = transcribe_long(str(canonical), engine=engine)
        print(to_json(long_result))
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"(skipped live run: {exc})")
        print("In production this runs the chunk -> transcribe -> offset-merge "
              "path behind a job queue.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcription pipeline")
    parser.add_argument("file", nargs="?", help="audio file to transcribe")
    parser.add_argument("--whisper", action="store_true",
                        help="use real Whisper (needs faster-whisper + ffmpeg)")
    parser.add_argument("--long", action="store_true",
                        help="use the long-audio chunking path")
    parser.add_argument("--fmt", default="json",
                        choices=["json", "srt", "text"])
    parser.add_argument("--language", default=None)
    args = parser.parse_args()

    if not args.file:
        run_demo()
        return

    engine = WhisperTranscriber() if args.whisper else MockTranscriber()

    try:
        if args.long:
            audio = ingest(args.file)
            wav = preprocess(audio)
            result = transcribe_long(str(wav), engine=engine,
                                     language=args.language)
            print({"json": to_json, "srt": to_srt, "text": to_text}[args.fmt](result))
        else:
            print(transcribe_file(args.file, engine=engine,
                                  language=args.language, fmt=args.fmt))
    except IngestError as exc:
        print(f"ingest error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
