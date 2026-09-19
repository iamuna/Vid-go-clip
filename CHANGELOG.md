# Changelog

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
