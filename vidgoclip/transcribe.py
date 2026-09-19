from __future__ import annotations

from pathlib import Path
from typing import Callable

from faster_whisper import WhisperModel

from .models import TranscriptSegment, TranscriptWord

Progress = Callable[[str], None] | None


def _runtime() -> tuple[str, str]:
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda", "float16"
    except Exception:
        pass
    return "cpu", "int8"


def transcribe_video(
    video_path: Path,
    *,
    model_name: str = "small",
    progress: Progress = None,
) -> tuple[list[TranscriptSegment], list[TranscriptWord], str]:
    device, compute_type = _runtime()
    if progress:
        progress(
            f"Loading Whisper {model_name} on {device} ({compute_type})..."
        )

    model = WhisperModel(
        model_name,
        device=device,
        compute_type=compute_type,
    )

    if progress:
        progress("Transcribing speech with word-level timing...")

    segments_iter, info = model.transcribe(
        str(video_path),
        beam_size=5,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        condition_on_previous_text=False,
        word_timestamps=True,
    )

    segments: list[TranscriptSegment] = []
    words: list[TranscriptWord] = []

    for index, segment in enumerate(segments_iter, start=1):
        text = segment.text.strip()
        if text:
            segments.append(
                TranscriptSegment(
                    start=float(segment.start),
                    end=float(segment.end),
                    text=text,
                )
            )

        for word in getattr(segment, "words", None) or []:
            token = str(getattr(word, "word", "") or "").strip()
            if not token:
                continue
            start = getattr(word, "start", None)
            end = getattr(word, "end", None)
            if start is None or end is None:
                continue
            words.append(
                TranscriptWord(
                    start=float(start),
                    end=float(end),
                    word=token,
                    probability=float(
                        getattr(word, "probability", 1.0) or 0.0
                    ),
                )
            )

        if progress and index % 25 == 0:
            progress(
                f"Transcribed {index} speech segments • "
                f"{len(words)} timed words..."
            )

    language = str(getattr(info, "language", "") or "")
    if progress:
        progress(
            f"Transcription complete: {len(segments)} segments • "
            f"{len(words)} words"
            + (f" • language {language}" if language else "")
        )
    return segments, words, language
