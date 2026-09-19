from __future__ import annotations

from pathlib import Path
from typing import Callable

from scenedetect import AdaptiveDetector, SceneManager, open_video

from .models import Scene

Progress = Callable[[str], None] | None


def detect_scenes(
    video_path: Path,
    *,
    progress: Progress = None,
) -> list[Scene]:
    if progress:
        progress("Detecting shot and scene boundaries...")

    try:
        video = open_video(str(video_path))
        manager = SceneManager()
        manager.add_detector(
            AdaptiveDetector(
                adaptive_threshold=3.0,
                min_scene_len=12,
            )
        )
        manager.detect_scenes(video=video, show_progress=False)
        pairs = manager.get_scene_list()
    except Exception:
        return []

    scenes = [
        Scene(
            start=float(start.get_seconds()),
            end=float(end.get_seconds()),
        )
        for start, end in pairs
        if end.get_seconds() > start.get_seconds()
    ]
    if progress:
        progress(f"Detected {len(scenes)} visual scenes/shots.")
    return scenes
