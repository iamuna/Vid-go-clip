from __future__ import annotations

from pathlib import Path

from .config import OUTPUT_DIR
from .media import run_process, safe_slug
from .models import Candidate


def export_clip(
    video_path: Path,
    candidate: Candidate,
    *,
    destination_dir: Path | None = None,
    vertical: bool = False,
) -> Path:
    destination_dir = destination_dir or OUTPUT_DIR
    destination_dir.mkdir(parents=True, exist_ok=True)

    slug = safe_slug(candidate.title or candidate.id)
    output = destination_dir / (
        f"{candidate.id}-{slug}{'-9x16' if vertical else ''}.mp4"
    )

    duration = max(0.2, candidate.end - candidate.start)
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

    if vertical:
        command += [
            "-vf",
            (
                "scale=1080:1920:force_original_aspect_ratio=increase,"
                "crop=1080:1920,"
                "format=yuv420p"
            ),
        ]

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

    result = run_process(command)
    if result.returncode != 0:
        raise RuntimeError(
            result.stderr.strip() or "FFmpeg clip export failed."
        )
    return output
