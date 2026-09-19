from __future__ import annotations

from pathlib import Path

import cv2

from .media import run_process


def extract_candidate_frames(
    video_path: Path,
    *,
    start: float,
    end: float,
    destination: Path,
    count: int = 3,
) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    duration = max(0.5, end - start)

    if count <= 1:
        positions = [start + duration * 0.5]
    else:
        positions = [
            start + duration * (0.15 + 0.70 * index / (count - 1))
            for index in range(count)
        ]

    paths: list[Path] = []
    for index, timestamp in enumerate(positions):
        output = destination / f"frame-{index:02d}.jpg"
        result = run_process(
            [
                "ffmpeg",
                "-y",
                "-ss",
                f"{timestamp:.3f}",
                "-i",
                str(video_path),
                "-frames:v",
                "1",
                "-vf",
                "scale=640:-2",
                "-q:v",
                "3",
                str(output),
            ]
        )
        if result.returncode == 0 and output.exists():
            paths.append(output)
    return paths


def motion_score(
    video_path: Path,
    *,
    start: float,
    end: float,
    samples: int = 10,
) -> float:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return 0.0

    try:
        duration = max(0.1, end - start)
        frames = []
        for index in range(samples):
            timestamp = start + duration * (index + 0.5) / samples
            capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
            ok, frame = capture.read()
            if not ok:
                continue
            frame = cv2.resize(frame, (256, 144))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(gray)

        if len(frames) < 2:
            return 0.0

        diffs = []
        for left, right in zip(frames, frames[1:]):
            diff = cv2.absdiff(left, right)
            diffs.append(float(diff.mean()))

        mean_diff = sum(diffs) / len(diffs)
        # Typical conversational footage lands low; large movement/cuts climb.
        return round(min(100.0, mean_diff * 4.2), 2)
    finally:
        capture.release()
