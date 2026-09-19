# Research — automatic video clipping / highlight extraction

Research date: 2026-09-19.

## What this category is called

Common terms:

- AI video clipping
- automatic highlight extraction
- long-form to short-form clipping
- highlight detection
- clip-worthy moment detection
- social clip generation

The user's target is broader than simple trimming: the software should **watch/listen to a long video, identify the strongest moments, explain why, and export context-complete clips**.

## What current commercial systems emphasize

### OpusClip

OpusClip's current material emphasizes that reframing is not cosmetic: a crop should keep the active speaker and relevant evidence in view. Their 2026 reframe benchmark weights active-speaker focus as an explicit quality criterion.

Source:
https://www.opus.pro/blog/opusclip-ranks-1-across-every-ai-reframe-engine

Implication for Vid-go-clip:
- moment discovery and reframing should be separate stages,
- future dynamic 9:16 should track the speaking/important subject rather than blindly center-crop.

### Vizard

Vizard describes modern AI clipping as highlight discovery + subtitles + platform optimization. More importantly, its 2026 explanation of clipping-model changes identifies a common failure: one-pass systems choose the interesting moment and boundaries at the same time, causing clips that end mid-sentence.

Sources:
https://vizard.ai/blog/why-clips-used-to-get-cut-off-mid-sentence-and-how-we-fixed-it
https://docs.vizard.ai/docs/introduction

Implication:
- discover first,
- score/rank second,
- refine start/end boundaries after a candidate is considered strong,
- context completeness deserves its own score.

### Descript

Descript's current clipping product finds good clips from long recordings. Their 2026 engineering write-up makes an especially important point: transcript-only AI cannot find moments whose importance is visual, so the system must inspect video frames as well as speech.

Sources:
https://www.descript.com/ai/find-good-clips
https://www.descript.com/blog/article/underlord-ai-can-watch-videos

Implication:
- transcript-only ranking is insufficient,
- top candidates need visual verification,
- visual description should be exposed to the user rather than hidden.

## Open-source/local building blocks

### faster-whisper

faster-whisper is a CTranslate2 implementation of Whisper. Its project reports substantial speed/memory improvements over the original Whisper implementation and supports 8-bit inference.

Source:
https://github.com/SYSTRAN/faster-whisper

Why selected:
- local transcription,
- no metered API,
- strong timestamped speech foundation,
- practical CPU fallback and GPU acceleration when the runtime supports it.

### PySceneDetect

PySceneDetect provides content, adaptive, threshold, histogram, and perceptual-hash based scene detectors. Adaptive/content detectors use frame changes to locate shot boundaries.

Sources:
https://www.scenedetect.com/docs/head/
https://www.scenedetect.com/docs/head/api/detectors.html

Why selected:
- scene cuts are useful boundary/context signals,
- local and deterministic,
- complements transcript timing.

### Ollama + Qwen3-VL

Ollama currently exposes local Qwen3-VL variants including 2B, 4B, 8B and larger models. The 4B build is listed at roughly 3.3 GB and accepts text + image input.

Source:
https://ollama.com/library/qwen3-vl

Why selected:
- one local model can score transcript meaning and inspect sampled video frames,
- 4B is a practical starting point for consumer hardware,
- provider/model can be replaced later,
- no paid inference API is required.

## Design conclusion

A useful clipping engine should not ask one model one vague question like:

> "Which parts will go viral?"

That collapses too many different editorial judgments into one opaque answer.

Vid-go-clip v0.1 uses distinct stages:

1. **Transcribe** speech with timestamps.
2. **Detect visual scene boundaries.**
3. **Generate candidate windows** aligned to speech rather than arbitrary fixed timestamps.
4. **Cheap local heuristic prefilter** to avoid spending multimodal inference on every overlapping window.
5. **Semantic AI scoring** on:
   - importance
   - controversy/disagreement potential
   - interest
   - emotion
   - context completeness
6. **Visual verification** of the strongest candidates using chronological sampled frames.
7. **Motion signal** from actual frames as a second visual clue.
8. **Weighted ranking** based on user-selected focus.
9. **Overlap de-duplication** after ranking.
10. **Precise FFmpeg export.**

## Why separate category scores

"Important", "controversial", and "interesting" are not synonyms.

A moment can be:
- important but visually boring,
- controversial but low-information,
- fascinating but non-controversial,
- emotionally powerful but context-dependent,
- visually strong with almost no speech.

Separate scores let the user choose the editorial goal instead of accepting a black-box "viral" label.

The controversy score is specifically defined as **likelihood of meaningful disagreement/debate**, not truth, morality, correctness, or political desirability.

## Boundary strategy

Clip boundaries matter almost as much as clip discovery.

v0.1:
- starts/ends on Whisper speech-segment timing,
- favors ending on punctuation/completed speech,
- can prepend a previous segment when a candidate begins with a connective such as "but" or "because",
- can append the next segment when the ending appears incomplete,
- enforces configurable min/target/max durations.

Future:
- use the AI to suggest exact boundary corrections,
- include speaker-turn boundaries,
- include silence/applause/laughter boundaries,
- inspect whether a visual action completes after speech ends.

## Visual strategy

v0.1:
- top semantic candidates get three chronological frame samples,
- Qwen3-VL describes what is visibly happening and assigns a visual-value score,
- OpenCV measures frame-to-frame movement as a supporting signal.

This is intentionally cheaper than trying to send an entire multi-hour video to a vision model.

Future:
- adaptive sampling around high motion,
- face/speaker tracking,
- OCR of charts/captions,
- object/event detectors for sports/gameplay,
- direct short video chunks into models with native temporal video support where practical.

## Reframing

Automatic 9:16 reframing is a separate problem from clipping.

v0.1 offers an optional center crop for convenience, but it is explicitly **not** considered smart reframing.

Future reframing should account for:
- active speaker,
- face position,
- multiple speakers,
- important objects/charts,
- camera cuts,
- shot composition continuity.

## Product principles

- Local/free core by default.
- Explain why every clip was selected.
- Preserve context rather than maximize sensationalism.
- Do not silently discard the full candidate pool when the user changes ranking focus.
- Cache expensive analysis.
- Provider/model interfaces should remain replaceable.
- Never imply an editorial score is factual truth.
