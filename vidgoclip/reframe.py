from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .models import Candidate

Progress = Callable[[str], None] | None


@dataclass
class TrackPoint:
    time: float
    x_ratio: float
    source: str


def smooth_tracking_points(
    points: list[TrackPoint],
    *,
    alpha: float = 0.28,
) -> list[TrackPoint]:
    if not points:
        return []
    smoothed = [
        TrackPoint(points[0].time, points[0].x_ratio, points[0].source)
    ]
    current = points[0].x_ratio
    for point in points[1:]:
        current = current * (1.0 - alpha) + point.x_ratio * alpha
        smoothed.append(
            TrackPoint(point.time, float(current), point.source)
        )
    return smoothed


def _face_detector():
    path = (
        Path(cv2.data.haarcascades)
        / "haarcascade_frontalface_default.xml"
    )
    if not path.exists():
        return None
    detector = cv2.CascadeClassifier(str(path))
    if detector.empty():
        return None
    return detector


def analyze_tracking_path(
    video_path: Path,
    candidate: Candidate,
    *,
    sample_fps: float = 4.0,
) -> list[TrackPoint]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return [TrackPoint(0.0, 0.5, "center")]

    detector = _face_detector()
    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    interval_frames = max(1, int(round(source_fps / sample_fps)))
    start_frame = max(0, int(candidate.start * source_fps))
    end_frame = max(start_frame + 1, int(candidate.end * source_fps))

    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    points: list[TrackPoint] = []
    previous_gray = None
    previous_ratio = 0.5
    frame_index = start_frame

    try:
        while frame_index <= end_frame:
            ok, frame = capture.read()
            if not ok:
                break

            if (frame_index - start_frame) % interval_frames != 0:
                frame_index += 1
                continue

            height, width = frame.shape[:2]
            small_width = min(640, width)
            scale = small_width / max(1, width)
            small = cv2.resize(
                frame,
                (
                    small_width,
                    max(1, int(height * scale)),
                ),
            )
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)

            chosen_ratio = None
            chosen_source = "center"

            if detector is not None:
                faces = detector.detectMultiScale(
                    gray,
                    scaleFactor=1.12,
                    minNeighbors=5,
                    minSize=(34, 34),
                )
                if len(faces):
                    faces = sorted(
                        faces,
                        key=lambda box: box[2] * box[3],
                        reverse=True,
                    )
                    # Prefer a large face that is not an extreme jump from the
                    # previous crop center.
                    top = faces[: min(4, len(faces))]
                    best = min(
                        top,
                        key=lambda box: abs(
                            ((box[0] + box[2] / 2) / small.shape[1])
                            - previous_ratio
                        )
                        - (box[2] * box[3])
                        / max(1.0, small.shape[0] * small.shape[1])
                        * 0.45,
                    )
                    chosen_ratio = (
                        best[0] + best[2] / 2
                    ) / small.shape[1]
                    chosen_source = "face"

            if chosen_ratio is None and previous_gray is not None:
                delta = cv2.absdiff(gray, previous_gray)
                delta = cv2.GaussianBlur(delta, (9, 9), 0)
                _, mask = cv2.threshold(delta, 18, 255, cv2.THRESH_BINARY)
                mask = cv2.morphologyEx(
                    mask,
                    cv2.MORPH_OPEN,
                    np.ones((3, 3), np.uint8),
                )
                mass = float(mask.sum())
                if mass > mask.size * 255 * 0.008:
                    moments = cv2.moments(mask)
                    if moments["m00"] > 0:
                        chosen_ratio = (
                            moments["m10"] / moments["m00"]
                        ) / small.shape[1]
                        chosen_source = "motion"

            if chosen_ratio is None:
                chosen_ratio = previous_ratio * 0.92 + 0.5 * 0.08

            chosen_ratio = float(
                max(0.08, min(0.92, chosen_ratio))
            )
            relative_time = (
                frame_index - start_frame
            ) / max(1.0, source_fps)
            points.append(
                TrackPoint(
                    relative_time,
                    chosen_ratio,
                    chosen_source,
                )
            )
            previous_ratio = chosen_ratio
            previous_gray = gray
            frame_index += 1
    finally:
        capture.release()

    if not points:
        points = [TrackPoint(0.0, 0.5, "center")]

    return smooth_tracking_points(points)


def _interpolated_ratio(points: list[TrackPoint], time_value: float) -> float:
    if not points:
        return 0.5
    if len(points) == 1:
        return points[0].x_ratio

    times = [point.time for point in points]
    ratios = [point.x_ratio for point in points]
    return float(np.interp(time_value, times, ratios))


def render_dynamic_vertical(
    video_path: Path,
    candidate: Candidate,
    destination: Path,
    *,
    progress: Progress = None,
) -> tuple[Path, list[TrackPoint]]:
    points = analyze_tracking_path(video_path, candidate)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError("OpenCV could not open the video for smart reframing.")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError("Could not read source video dimensions.")

    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(destination),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (1080, 1920),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError("Could not create temporary smart-reframe video.")

    start_frame = max(0, int(candidate.start * fps))
    total_frames = max(1, int(candidate.duration * fps))
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    target_ratio = 9.0 / 16.0
    source_ratio = width / max(1.0, height)

    try:
        for local_index in range(total_frames):
            ok, frame = capture.read()
            if not ok:
                break

            time_value = local_index / max(1.0, fps)
            center_ratio = _interpolated_ratio(points, time_value)

            if source_ratio >= target_ratio:
                crop_h = height
                crop_w = max(2, int(round(height * target_ratio)))
                center_x = int(round(center_ratio * width))
                x0 = max(0, min(width - crop_w, center_x - crop_w // 2))
                crop = frame[0:crop_h, x0 : x0 + crop_w]
            else:
                crop_w = width
                crop_h = max(2, int(round(width / target_ratio)))
                crop_h = min(height, crop_h)
                y0 = max(0, (height - crop_h) // 2)
                crop = frame[y0 : y0 + crop_h, 0:crop_w]

            vertical = cv2.resize(
                crop,
                (1080, 1920),
                interpolation=cv2.INTER_AREA,
            )
            writer.write(vertical)

            if progress and local_index % max(1, int(fps * 3)) == 0:
                percent = int(local_index / total_frames * 100)
                progress(f"Smart reframing... {percent}%")
    finally:
        writer.release()
        capture.release()

    return destination, points
