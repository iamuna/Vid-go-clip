from __future__ import annotations

from pathlib import Path
from typing import Callable

from .ai import (
    ollama_ready,
    score_semantic_candidates,
    score_visual_candidate,
)
from .candidates import (
    apply_final_score,
    build_candidates,
    deduplicate_ranked,
    expand_context,
    prefilter_candidates,
)
from .config import CACHE_DIR
from .frames import extract_candidate_frames, motion_score
from .media import ffprobe_duration, video_cache_key
from .models import AnalysisResult, Candidate, Scene, TranscriptSegment
from .scenes import detect_scenes
from .storage import analysis_key, load_analysis, save_analysis
from .transcribe import transcribe_video

Progress = Callable[[str], None] | None


def _progress(callback: Progress, message: str) -> None:
    if callback:
        callback(message)


def _visual_only_candidates(
    scenes: list[Scene],
    duration: float,
    *,
    target_seconds: float,
    max_seconds: float,
) -> list[Candidate]:
    if not scenes:
        scenes = [Scene(start=0.0, end=duration)]

    result: list[Candidate] = []
    index = 0
    while index < len(scenes):
        start = scenes[index].start
        end = scenes[index].end
        cursor = index + 1

        while cursor < len(scenes) and end - start < target_seconds:
            proposed = scenes[cursor].end
            if proposed - start > max_seconds:
                break
            end = proposed
            cursor += 1

        candidate = Candidate(
            id=f"v{len(result)+1:04d}",
            start=start,
            end=min(duration, end),
            text="[No speech detected in this visual segment]",
            heuristic_score=42.0,
        )
        result.append(candidate)
        index = max(index + 1, cursor)
    return result


def analyze_video(
    video_path: Path,
    *,
    settings: dict,
    progress: Progress = None,
    force: bool = False,
) -> AnalysisResult:
    if not video_path.is_file():
        raise RuntimeError("Selected video file does not exist.")

    duration = ffprobe_duration(video_path)
    video_key = video_cache_key(video_path)
    key = analysis_key(
        video_key,
        whisper_model=str(settings["whisper_model"]),
        ollama_model=str(settings["ollama_model"]),
    )

    if not force:
        cached = load_analysis(key)
        if cached is not None:
            _progress(progress, "Loaded cached analysis.")
            for candidate in cached.candidates:
                apply_final_score(
                    candidate,
                    str(settings.get("analysis_focus", "Balanced")),
                )
            cached.candidates = deduplicate_ranked(cached.candidates)
            return cached

    transcript, language = transcribe_video(
        video_path,
        model_name=str(settings["whisper_model"]),
        progress=progress,
    )
    scenes = detect_scenes(video_path, progress=progress)

    _progress(progress, "Building candidate clip windows...")
    candidates = build_candidates(
        transcript,
        scenes,
        min_seconds=float(settings["min_clip_seconds"]),
        target_seconds=float(settings["target_clip_seconds"]),
        max_seconds=float(settings["max_clip_seconds"]),
    )
    if not candidates:
        candidates = _visual_only_candidates(
            scenes,
            duration,
            target_seconds=float(settings["target_clip_seconds"]),
            max_seconds=float(settings["max_clip_seconds"]),
        )

    candidates = prefilter_candidates(
        candidates,
        limit=int(settings["max_ai_candidates"]),
    )

    ai_available = ollama_ready(str(settings["ollama_base_url"]))
    if ai_available and transcript:
        _progress(
            progress,
            f"Scoring {len(candidates)} candidate moments with local AI...",
        )
        score_semantic_candidates(
            candidates,
            base_url=str(settings["ollama_base_url"]),
            model=str(settings["ollama_model"]),
        )
    else:
        from .ai import fallback_semantic_scores

        _progress(
            progress,
            "Local AI is unavailable; using fallback semantic ranking.",
        )
        for candidate in candidates:
            candidate.scores.update(fallback_semantic_scores(candidate))
            candidate.title = candidate.title or f"Moment {candidate.id}"
            candidate.reason = (
                "Fallback ranking. Start Ollama for deeper semantic analysis."
            )

    for candidate in candidates:
        apply_final_score(candidate, "Balanced")

    visual_limit = min(
        len(candidates),
        int(settings["max_visual_candidates"]),
    )
    visual_targets = sorted(
        candidates,
        key=lambda item: item.final_score,
        reverse=True,
    )[:visual_limit]

    frame_root = CACHE_DIR / f"frames-{key}"
    for index, candidate in enumerate(visual_targets, start=1):
        _progress(
            progress,
            f"Watching candidate {index}/{visual_limit} visually...",
        )
        motion = motion_score(
            video_path,
            start=candidate.start,
            end=candidate.end,
        )
        frames = extract_candidate_frames(
            video_path,
            start=candidate.start,
            end=candidate.end,
            destination=frame_root / candidate.id,
            count=3,
        )
        if ai_available:
            score_visual_candidate(
                candidate,
                frame_paths=frames,
                motion=motion,
                base_url=str(settings["ollama_base_url"]),
                model=str(settings["ollama_model"]),
            )
        else:
            candidate.scores["visual"] = max(
                20.0,
                min(75.0, 30.0 + motion * 0.55),
            )

    for candidate in candidates:
        expand_context(
            candidate,
            transcript,
            max_seconds=float(settings["max_clip_seconds"]),
        )
        apply_final_score(
            candidate,
            str(settings.get("analysis_focus", "Balanced")),
        )

    # Cache the full scored pool so changing focus later can produce a
    # genuinely different ranking instead of reusing an already-deduplicated list.
    cached_result = AnalysisResult(
        video_path=str(video_path.resolve()),
        duration=duration,
        transcript=transcript,
        scenes=scenes,
        candidates=candidates,
        cache_key=key,
        model_notes={
            "whisper": str(settings["whisper_model"]),
            "ollama": str(settings["ollama_model"]),
            "language": language,
            "visual_ai": "enabled" if ai_available else "fallback",
        },
    )
    save_analysis(cached_result)

    ranked = deduplicate_ranked(candidates)
    result = AnalysisResult(
        video_path=cached_result.video_path,
        duration=cached_result.duration,
        transcript=cached_result.transcript,
        scenes=cached_result.scenes,
        candidates=ranked,
        cache_key=cached_result.cache_key,
        model_notes=cached_result.model_notes,
    )
    _progress(progress, f"Analysis complete: {len(ranked)} ranked moments.")
    return result
