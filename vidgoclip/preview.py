from __future__ import annotations

from pathlib import Path

from .config import TEMP_DIR
from .media import run_process, safe_slug, video_cache_key
from .models import Candidate


def prepare_preview(
    video_path: Path,
    candidate: Candidate,
) -> tuple[Path, Path | None]:
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    key = video_cache_key(video_path)
    slug = safe_slug(candidate.id)
    stem = f"preview-{key}-{slug}-{int(candidate.start*100)}-{int(candidate.end*100)}"
    video_out = TEMP_DIR / f"{stem}.mp4"
    audio_out = TEMP_DIR / f"{stem}.wav"
    duration = max(0.2, candidate.duration)

    if not video_out.exists() or video_out.stat().st_size < 4096:
        result = run_process(
            [
                "ffmpeg",
                "-y",
                "-ss",
                f"{candidate.start:.3f}",
                "-i",
                str(video_path),
                "-t",
                f"{duration:.3f}",
                "-vf",
                "scale=640:-2",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-crf",
                "26",
                "-movflags",
                "+faststart",
                str(video_out),
            ]
        )
        if result.returncode != 0:
            raise RuntimeError(
                result.stderr.strip() or "Could not prepare preview video."
            )

    audio_path: Path | None = audio_out
    if not audio_out.exists() or audio_out.stat().st_size < 1024:
        audio = run_process(
            [
                "ffmpeg",
                "-y",
                "-ss",
                f"{candidate.start:.3f}",
                "-i",
                str(video_path),
                "-t",
                f"{duration:.3f}",
                "-vn",
                "-ac",
                "2",
                "-ar",
                "44100",
                "-c:a",
                "pcm_s16le",
                str(audio_out),
            ]
        )
        if audio.returncode != 0:
            audio_path = None

    return video_out, audio_path
