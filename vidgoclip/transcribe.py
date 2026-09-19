from __future__ import annotations

from pathlib import Path
from typing import Callable

from faster_whisper import WhisperModel

from .models import TranscriptSegment

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
) -> tuple[list[TranscriptSegment], str]:
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
        progress("Transcribing speech locally...")

    segments_iter, info = model.transcribe(
        str(video_path),
        beam_size=5,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        condition_on_previous_text=False,
    )

    segments: list[TranscriptSegment] = []
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
        if progress and index % 25 == 0:
            progress(f"Transcribed {index} speech segments...")

    language = str(getattr(info, "language", "") or "")
    if progress:
        progress(
            f"Transcription complete: {len(segments)} segments"
            + (f" • language {language}" if language else "")
        )
    return segments, language
