from __future__ import annotations

from pathlib import Path
from typing import Callable

from .captions import write_ass_captions
from .config import OUTPUT_DIR, TEMP_DIR
from .media import run_process, safe_slug
from .models import Candidate, TranscriptSegment, TranscriptWord
from .reframe import render_dynamic_vertical

Progress = Callable[[str], None] | None


def _ffmpeg_filter_path(path: Path) -> str:
    value = str(path.resolve()).replace("\\", "/")
    value = value.replace(":", r"\:")
    value = value.replace("'", r"\'")
    return value


def export_clip(
    video_path: Path,
    candidate: Candidate,
    *,
    destination_dir: Path | None = None,
    vertical: bool = False,
    smart_reframe: bool = True,
    burn_captions: bool = True,
    words: list[TranscriptWord] | None = None,
    transcript: list[TranscriptSegment] | None = None,
    caption_words_per_line: int = 7,
    progress: Progress = None,
) -> Path:
    destination_dir = destination_dir or OUTPUT_DIR
    destination_dir.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    slug = safe_slug(candidate.title or candidate.id)
    suffix = "-9x16" if vertical else ""
    output = destination_dir / f"{candidate.id}-{slug}{suffix}.mp4"
    duration = max(0.2, candidate.end - candidate.start)

    ass_file = None
    if burn_captions:
        ass_file = write_ass_captions(
            TEMP_DIR / f"{candidate.id}-{slug}.ass",
            candidate,
            words=words or [],
            transcript=transcript or [],
            words_per_line=caption_words_per_line,
        )

    subtitle_filter = ""
    if ass_file is not None:
        subtitle_filter = f"ass='{_ffmpeg_filter_path(ass_file)}'"

    if vertical and smart_reframe:
        if progress:
            progress("Tracking faces/motion for smart 9:16 reframing...")
        temp_video = TEMP_DIR / f"{candidate.id}-{slug}-tracked.mp4"
        render_dynamic_vertical(
            video_path,
            candidate,
            temp_video,
            progress=progress,
        )

        command = [
            "ffmpeg",
            "-y",
            "-i",
            str(temp_video),
            "-ss",
            f"{candidate.start:.3f}",
            "-i",
            str(video_path),
            "-t",
            f"{duration:.3f}",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0?",
        ]
        if subtitle_filter:
            command += ["-vf", subtitle_filter]
        command += [
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            "-shortest",
            "-movflags",
            "+faststart",
            str(output),
        ]
    else:
        filters: list[str] = []
        if vertical:
            filters.append(
                "scale=1080:1920:force_original_aspect_ratio=increase,"
                "crop=1080:1920,format=yuv420p"
            )
        if subtitle_filter:
            filters.append(subtitle_filter)

        command = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{candidate.start:.3f}",
            "-i",
            str(video_path),
            "-t",
            f"{duration:.3f}",
        ]
        if filters:
            command += ["-vf", ",".join(filters)]
        command += [
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            "-movflags",
            "+faststart",
            str(output),
        ]

    if progress:
        progress("Encoding final clip...")
    result = run_process(command)
    if result.returncode != 0:
        raise RuntimeError(
            result.stderr.strip() or "FFmpeg clip export failed."
        )
    return output
