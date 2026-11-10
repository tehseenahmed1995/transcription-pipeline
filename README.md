# Transcription Pipeline

A small, well-structured speech-to-text pipeline: accept an audio file,
normalize it, transcribe speech to text, and return timestamped segments —
with explicit handling for different formats and long audio.

The focus of this project is **engineering decisions**, not model training.
It uses OpenAI's open-source Whisper (via `faster-whisper`) and is designed so
the model is swappable and the whole thing runs without installing it.

---

## Quick start

```bash
# Zero installs — runs the full pipeline with a mock transcriber + generated audio:
python main.py

# Real transcription (needs `pip install -r requirements.txt` + ffmpeg):
python main.py audio.mp3 --whisper --fmt srt
python main.py long.wav --whisper --long --fmt json
```

---

## Architecture

The pipeline is a series of stages, each a function with a clear input/output
contract. Stages are independent, testable, and replaceable.

```
audio file
  │
  ▼
1. ingest        validate the file; return a typed AudioInput         (Q1)
  │
  ▼
2. preprocess    normalize any format → 16 kHz mono WAV               (Q4)
  │
  ▼
3. transcribe    speech → text + segments (Whisper behind an interface) (Q2)
  │
  ▼
4. format        JSON / SRT / plain text with timestamps              (Q3)

For long audio (Q5), step 3 is wrapped by a chunk → transcribe → offset-merge
path (see "Long audio" below).
```

### Project structure

| File | Responsibility |
|------|----------------|
| `models.py` | Shared data models: `Segment`, `TranscriptionResult`, `AudioInput` |
| `transcribe.py` | `Transcriber` interface + `WhisperTranscriber` + `MockTranscriber` |
| `pipeline.py` | Stages (ingest, preprocess, format, long-audio) + orchestrator |
| `main.py` | CLI + dependency-free demo |
| `requirements.txt` | Optional deps for the real Whisper path only |

---

## Key design decisions

### 1. Swappable engine via an interface (dependency inversion)
`Transcriber` is an abstract interface; `WhisperTranscriber` (real) and
`MockTranscriber` (canned output) both implement it. The pipeline depends on
the interface, never on a concrete engine, so:
- we can unit-test every stage without installing the model,
- a cloud STT API can drop in later without touching the pipeline.

### 2. Mock-first; heavy deps are lazy
`torch` / `faster_whisper` are imported only when a `WhisperTranscriber`
instance is created, so importing the module never forces a giant install.
The default path (`python main.py` with no args) runs end-to-end with **zero
dependencies** — it generates a sample WAV with the stdlib `wave` module and
transcribes it with `MockTranscriber`. This keeps CI green and demos runnable
on any machine.

### 3. Canonical normalization for both format-handling and accuracy
Every format is normalized to **16 kHz mono 16-bit PCM WAV** before the model
sees it. This isn't just format handling — Whisper is trained on 16 kHz audio,
so normalizing also maximizes accuracy. `ffmpeg` does the decoding, so one code
path handles MP3, M4A, FLAC, OGG, WebM, etc. If the input is already canonical
and `ffmpeg` is absent, it passes straight through (keeps the demo runnable).

### 4. Fail fast at the boundary
`ingest` validates (exists, supported format, non-empty, under size limit) and
raises a specific `IngestError` immediately. Problems surface at the front
door instead of three stages later inside the model.

### 5. One pass feeds every output format
Whisper emits time-stamped segments; one transcription run produces JSON (for
APIs/databases), SRT (for subtitles), and plain text (for search indexing).
Timestamps are stored in seconds internally and only formatted at the edge, so
adding a new format touches only the formatter, never the core.

### 6. Long audio: chunk + offset-merge (the key detail)
Whisper processes ~30-second windows. For most files `faster-whisper`'s VAD
splits and timestamps automatically. For very long files we split on silence
with `ffmpeg`, transcribe each chunk independently, then merge with a
**cumulative time offset** — because each chunk's clock resets to zero, we add
the elapsed time so final timestamps are correct across the whole file. Chunks
are capped at 25 s and cut at silence midpoints (robust to detection jitter,
avoids slicing words). This also enables per-chunk progress reporting and retry.

---

## Tech choices

- **Language: Python.** Native ecosystem for ML/audio; Whisper's libraries are
  Python-first.
- **STT: `faster-whisper`** (CTranslate2 backend) over the reference
  `openai-whisper` — same weights, lighter memory, faster on CPU/GPU.
- **Audio: `ffmpeg`** for resampling/format conversion; stdlib `wave` for
  test fixtures and duration probing of WAV files (no external dep).

---

## Part 2 — System design (production scaling)

The code above is the core engine. Here is how it scales into a production
service.

### Concurrent uploads
- **Decouple upload from processing.** The API streams the upload to object
  storage and returns a job ID immediately (HTTP 202). Transcription is slow, so
  it never blocks the request thread.
- **Stateless web tier behind a load balancer**, scaled horizontally.
- **Durable queue** (Redis / SQS / RabbitMQ) decouples web from worker tier;
  workers pull jobs and scale independently (CPU- or GPU-sized pools).
- **Backpressure & limits**: per-user rate limiting, bounded queue depth, max
  concurrent jobs per worker — reject/queue beyond capacity rather than OOM.
- **Dedup by content hash** so the same file uploaded twice isn't transcribed
  twice.

### Storing audio and transcripts
- **Audio → object storage** (S3/GCS/MinIO), not a database or local disk:
  immutable large blobs, cheap, with content-hash keys and lifecycle rules
  (move to cold storage / delete after N days — raw audio is rarely needed
  long-term).
- **Transcripts → document store** (MongoDB is a great fit: one document per
  transcription with the `segments` array and metadata; Postgres+JSONB is a
  fine alternative). Store the structured segments (with timestamps) as the
  source of truth; derive SRT/plain-text on demand or cache them.
- **Full-text index** (Elasticsearch / Meilisearch / Mongo text index) if
  downstream search is a use case.
- **Metadata**: job id, user id, status, language, duration, model, source
  hash, created_at.

### Retry / recovery
- **Durable job records** with explicit statuses (queued → processing →
  done / failed) so a crash never loses work.
- **Retry with exponential backoff** for transient errors (network, model
  load); **fail fast** for permanent errors (corrupt audio, unsupported
  format) — don't retry forever.
- **Chunk-level recovery**: because long audio is chunked, a failure in chunk 7
  retries just that chunk, not the whole file. Per-chunk status preserves
  partial progress.
- **Idempotent writes** (job id + chunk index as key) so retries are safe.
- **Visibility timeout / lease** so a dead worker's job is reassigned to
  another (at-least-once delivery, hence idempotency).
- **Dead-letter queue + alerting** after N retries; **circuit breaker** when
  the model/infra is down.

### Exposing as an API
- `POST /v1/transcriptions` — upload (or pass a pre-signed S3 URL for large
  files), returns 202 + job id + status URL.
- `GET /v1/transcriptions/{id}` — status + result; `?format=srt|json|text` or
  content negotiation for output format.
- `DELETE /v1/transcriptions/{id}` — cleanup.
- **Async pattern** (202 + polling) because transcription is slow; optional
  webhooks or SSE/WebSocket for streaming progress.
- **Pre-signed S3 uploads** for large files keep the API server out of the
  data path.
- **Auth** (API key / JWT), per-user quotas, rate limits, idempotency-key
  header so retried upload requests don't create duplicate jobs.
- Deploy API and workers as separate, independently-scaled containers.

---

## License / model note
Whisper is MIT-licensed (OpenAI). `faster-whisper` is MIT-licensed. No model is
trained here — only open weights are used.
