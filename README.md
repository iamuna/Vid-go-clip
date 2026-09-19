# Vid-go-clip

**Local-first AI video clipping / highlight extraction.**

Upload a long video and Vid-go-clip analyzes both **what is said** and **what is happening on screen**, then ranks the strongest moments for clipping.

## Goal

```
long video
   ↓
speech transcription
   ↓
scene / shot detection
   ↓
candidate conversation windows
   ↓
AI semantic scoring
   ↓
visual frame analysis
   ↓
context-boundary refinement
   ↓
ranked important / controversial / interesting moments
   ↓
one-click clip export
```

The project is designed to find more than keyword hits. A useful clip should:

- contain a complete thought rather than ending mid-sentence,
- make sense without requiring the previous five minutes,
- contain something important, interesting, emotional, surprising, debatable, or visually notable,
- start early enough to preserve the setup,
- end after the payoff,
- avoid near-duplicate clips,
- and eventually support smart 9:16 reframing/captions.

## v0.1 MVP

The first implementation provides:

- Windows desktop GUI.
- Local video selection.
- Local transcription with **faster-whisper**.
- Shot/scene boundaries with **PySceneDetect**.
- Candidate clip generation aligned to speech timing.
- Local semantic scoring through **Ollama**.
- Local visual understanding from sampled video frames through **Qwen3-VL**.
- Separate scores for:
  - importance
  - controversy / disagreement potential
  - interest
  - emotion
  - visual value
  - context completeness
- Explainable reason + suggested clip title for every candidate.
- De-duplication of overlapping candidates.
- Ranked results with timestamps and transcript excerpts.
- Exact clip export through FFmpeg.
- Optional 9:16 center-crop export.
- Analysis caching so reopening the same video does not require full transcription again.

## Free / local by default

No paid API is required.

Default stack:

- **faster-whisper** — local speech recognition.
- **PySceneDetect** — local scene detection.
- **Ollama + Qwen3-VL 4B** — local transcript/visual reasoning.
- **FFmpeg** — extraction/rendering.
- **SQLite/JSON files** — local project state.

Qwen3-VL 4B is intentionally the starting model because it is small enough to be practical on many consumer GPUs while still accepting image input. The model/provider layer is replaceable.

## Quick Windows setup

1. Clone/download this repository.
2. Double-click `setup.bat`.
3. Let setup install Python dependencies and FFmpeg if needed.
4. Let setup install Ollama if needed.
5. Let setup pull `qwen3-vl:4b`.
6. Double-click `start.bat`.
7. Click **Choose video**.
8. Click **ANALYZE VIDEO**.
9. Select a ranked moment.
10. Click **EXPORT SELECTED**.

The first transcription downloads the selected Whisper model and caches it locally.

## Why the architecture is multi-stage

Research into current clipping products points to several recurring problems:

- transcript-only systems miss visually important moments,
- one-pass clipping often cuts clips mid-thought,
- automatic vertical crops can follow the wrong person or remove important visual evidence,
- a single opaque "viral score" is hard to trust.

Vid-go-clip therefore separates **discovery**, **visual verification**, **context completeness**, and **export**.

See `docs/RESEARCH.md` for the research notes and sources.

## Project map

```
Vid-go-clip/
├─ AGENTS.md
├─ README.md
├─ app.py
├─ setup.bat
├─ start.bat
├─ requirements.txt
├─ docs/
│  └─ RESEARCH.md
├─ tests/
│  └─ ...
└─ vidgoclip/
   ├─ ai.py
   ├─ candidates.py
   ├─ config.py
   ├─ exporter.py
   ├─ frames.py
   ├─ media.py
   ├─ models.py
   ├─ pipeline.py
   ├─ scenes.py
   ├─ storage.py
   └─ transcribe.py
```

## Current limitations

v0.1 is optimized first for podcasts, interviews, commentary, discussions, documentaries, livestreams, meetings, and videos where speech matters.

Pure sports/gameplay/action highlight detection needs specialized motion/audio-event scoring and is a planned extension.

The optional 9:16 export in v0.1 uses a safe center crop. **Active-speaker / face-aware dynamic reframing is a later milestone.**

## Development rule

Read `AGENTS.md` before making major changes. It records the architecture, research conclusions, cost constraints, and next priorities.
