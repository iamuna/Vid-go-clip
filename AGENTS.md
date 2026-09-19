# AGENTS.md — Vid-go-clip developer / AI handoff

Read this file before making major changes.

## Product goal

Vid-go-clip is a Windows-first local AI clipping application.

User flow:

**choose/upload a long video → analyze → ranked moments → inspect reasons/transcript → export clips**

The application should find:
- important moments,
- controversial/debatable conversations,
- interesting/surprising ideas,
- emotional moments,
- visually notable scenes/events,
- complete standalone conversations or story beats.

## Core design rule

Do not reduce everything to one unexplained "viral score."

The data model deliberately keeps separate scores:
- importance
- controversy
- interest
- emotion
- visual
- context completeness

The user chooses a focus preset and the final rank is derived from those stored component scores.

"Controversy" means disagreement/debate potential. It is not a fact-check, moral rating, political recommendation, or claim that a statement is false.

## Current v0.2 architecture

### Transcription
`vidgoclip/transcribe.py`

- faster-whisper
- defaults to `small`
- attempts CUDA when CTranslate2 reports a CUDA device; otherwise CPU int8
- speech/VAD timestamps feed candidate boundaries
- word-level timestamps are stored for exact boundary cleanup and captions

### Scene detection
`vidgoclip/scenes.py`

- PySceneDetect AdaptiveDetector
- used as a structural/visual signal

### Candidate discovery
`vidgoclip/candidates.py`

- creates overlapping speech-aligned windows
- configurable min/target/max duration
- local heuristic prefilter
- context expansion at connective/incomplete boundaries
- final overlap de-duplication

### Semantic + visual AI
`vidgoclip/ai.py`

Default:
- Ollama
- `qwen3-vl:4b`
- local only

Semantic analysis is batched to reduce model calls.

Only the strongest semantic candidates receive multimodal frame analysis in v0.1.

If Ollama is unavailable, the app still returns fallback heuristic candidates and clearly labels that limitation.

### Visual analysis
`vidgoclip/frames.py`

- 3 chronological frame samples for visual AI
- OpenCV frame-difference motion signal

This is a deliberate compromise between actually looking at the video and keeping local inference practical.

### Pipeline
`vidgoclip/pipeline.py`

Order:
1. metadata/cache key
2. cache reuse if available
3. transcription
4. scene detection
5. candidate generation
6. heuristic prefilter
7. semantic scoring
8. top-candidate visual analysis
9. context expansion
10. user-focus weighting
11. overlap de-duplication
12. cache full scored pool
13. return ranked subset

Important: **cache the full scored candidate pool, not only the current deduplicated ranking**. Otherwise switching focus cannot produce a genuinely different result.

### Audio excitement
`vidgoclip/audio_events.py`

- extracts local mono PCM audio once and caches it
- computes normalized RMS energy, spectral flux and zero-crossing signals
- derives an explainable audio score plus heuristic hints for loud speech / applause-like / laughter-like energy
- these labels are heuristics, not definitive audio-event classification

### Boundary refinement
`vidgoclip/candidates.py`

After segment-level context expansion, `refine_word_boundaries()` uses word timestamps, punctuation and meaningful pauses to clean clip starts/ends.

The current "speaker-turn" behavior is a **word/pause boundary heuristic**, not identity-level diarization. Do not claim speaker identity.

### Smart reframing
`vidgoclip/reframe.py`

- samples the selected clip for faces
- prefers a stable large face near the previous crop center
- falls back to motion centroid when no face is usable
- exponentially smooths the crop path
- renders 1080×1920 locally with OpenCV
- original audio is remuxed by FFmpeg

### Captions
`vidgoclip/captions.py`

- creates ASS subtitles from word timestamps
- falls back to transcript-segment timing
- exporter can burn captions into normal or vertical clips

### Preview
`vidgoclip/preview.py` + `app.py`

- caches a lightweight preview clip
- OpenCV drives in-app video playback
- Windows `winsound` plays the matching preview WAV asynchronously

### Export
`vidgoclip/exporter.py`

- FFmpeg H.264/AAC precise output
- optional 1080×1920
- optional smart face/motion reframing
- optional burned timed captions
- center crop remains as a fallback when smart tracking is disabled

## Cost / privacy rule

The default workflow should not require a metered cloud API.

Local dependencies/models can require downloads and electricity/GPU time, but Vid-go-clip should not silently create API charges.

If a paid provider is added later:
- make it optional,
- mark it clearly,
- never automatically fall back to it.

Input video, analysis cache, transcripts, extracted frames, and clips stay local by default and are gitignored.

## Research conclusions

See `docs/RESEARCH.md`.

Most important conclusions:
- transcript-only clipping misses visual events,
- one-pass boundary selection can cut thoughts off,
- reframing is a separate computer-vision problem,
- complete context is an explicit quality dimension,
- explainable category scores are preferable to one opaque number.

## Immediate engineering priorities after v0.2

1. Runtime/playtest on real podcasts, interviews, livestreams and documentaries.
2. Identity-level local speaker diarization as an optional plugin.
3. Better audio-event classification using a dedicated local audio model.
4. Active-speaker lip-motion correlation for multi-person shots.
5. Caption style presets and word highlighting.
6. Manual trim handles in the preview.
7. Sports/gameplay event plugins.
8. Native temporal video-model analysis for top candidates.
9. Batch-folder mode and watch-folder automation.
10. Export directly into the YT SMB project as a source queue.

## Coding discipline

- Keep GUI simple.
- Preserve cache migrations where reasonable.
- Avoid hidden paid dependencies.
- Keep AI model/provider replaceable.
- Do not make editorial scores sound objectively true.
- Keep expensive stages cached.
- Add tests for ranking/boundary changes.
- Record architecture changes here and in README/CHANGELOG.
