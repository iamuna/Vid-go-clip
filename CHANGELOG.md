# Changelog

## v0.2 — Smart Clipping

- Added word-level faster-whisper timestamps.
- Added word/pause/sentence-aware clip boundary refinement.
- Added local audio timeline analysis with RMS, spectral-flux and zero-crossing signals.
- Added audio excitement score and heuristic loud-speech / applause-like / laughter-like hints.
- Added audio as an explicit ranking dimension across focus presets.
- Added cached in-app preview video plus Windows audio playback.
- Added face-aware dynamic 9:16 reframing with motion fallback.
- Added smoothed crop tracking to reduce visual jitter.
- Added word-timed ASS caption generation with transcript fallback.
- Added independent 9:16 / Smart track / Captions export controls.
- Upgraded the Windows smoke test to validate smart 1080×1920 export, caption burn-in and AAC audio remux.
- Bumped analysis cache generation so old v0.1 analyses re-run with v0.2 signals.
- Added tests for word-boundary refinement, audio scoring, tracking smoothing and captions.
- Speaker-turn handling remains an explicit pause/word heuristic; identity-level diarization is not claimed.

## v0.1 — Research-driven clipping MVP

- Researched current AI clipping approaches and failure modes.
- Added local faster-whisper transcription.
- Added PySceneDetect visual shot boundaries.
- Added speech-aligned candidate generation.
- Added heuristic prefilter for long videos.
- Added batched local semantic scoring via Ollama.
- Added Qwen3-VL visual frame analysis.
- Added OpenCV motion scoring.
- Added independent importance / controversy / interest / emotion / visual / context scores.
- Added ranking focus presets.
- Added context expansion and overlap de-duplication.
- Added full candidate-pool caching so focus changes can re-rank correctly.
- Added Windows desktop GUI.
- Added precise FFmpeg clip export.
- Added optional 9:16 center-crop export.
- Added free/local Windows setup.
- Added research and AI/developer handoff documentation.
- Added unit tests and real FFmpeg export smoke test in Windows CI.
