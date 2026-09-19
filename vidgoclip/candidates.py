from __future__ import annotations

import math
import re

from .config import FOCUS_WEIGHTS
from .models import Candidate, Scene, TranscriptSegment

INTEREST_TERMS = {
    "surprising", "crazy", "insane", "secret", "actually", "never",
    "always", "imagine", "weird", "unexpected", "why", "how",
    "mistake", "truth", "real", "story", "problem", "change",
}
CONTROVERSY_TERMS = {
    "wrong", "disagree", "lie", "lying", "false", "controversial",
    "debate", "argument", "critic", "blame", "ban", "illegal",
    "scam", "fraud", "hate", "ridiculous", "shouldn't", "should not",
}
IMPORTANCE_TERMS = {
    "important", "because", "therefore", "result", "means", "reason",
    "evidence", "decision", "million", "billion", "percent", "risk",
    "must", "need", "major", "key", "critical", "conclusion",
}
EMOTION_TERMS = {
    "love", "hate", "angry", "afraid", "fear", "sad", "happy",
    "shocked", "shock", "amazing", "terrible", "horrible", "excited",
    "cry", "laugh", "pain", "sorry", "wow",
}

CONNECTIVE_STARTS = (
    "and ", "but ", "so ", "because ", "then ", "also ", "well ",
    "which ", "that ", "this is why ", "that's why ",
)


def _term_hits(text: str, terms: set[str]) -> int:
    lower = text.lower()
    return sum(1 for term in terms if term in lower)


def _scene_change_count(start: float, end: float, scenes: list[Scene]) -> int:
    return sum(1 for scene in scenes if start < scene.start < end)


def heuristic_score(
    text: str,
    start: float,
    end: float,
    scenes: list[Scene],
) -> float:
    words = text.split()
    if not words:
        return 0.0

    interest = _term_hits(text, INTEREST_TERMS)
    controversy = _term_hits(text, CONTROVERSY_TERMS)
    importance = _term_hits(text, IMPORTANCE_TERMS)
    emotion = _term_hits(text, EMOTION_TERMS)
    punctuation = min(4, text.count("?") + text.count("!"))
    numbers = min(3, len(re.findall(r"\b\d+(?:\.\d+)?%?\b", text)))
    scene_changes = min(4, _scene_change_count(start, end, scenes))
    density = min(1.0, len(words) / max(1.0, end - start) / 3.0)

    raw = (
        interest * 4.0
        + controversy * 4.5
        + importance * 3.5
        + emotion * 3.5
        + punctuation * 2.5
        + numbers * 1.8
        + scene_changes * 1.5
        + density * 8.0
    )
    return min(100.0, 18.0 + raw)


def build_candidates(
    transcript: list[TranscriptSegment],
    scenes: list[Scene],
    *,
    min_seconds: float = 18.0,
    target_seconds: float = 42.0,
    max_seconds: float = 75.0,
) -> list[Candidate]:
    if not transcript:
        return []

    candidates: list[Candidate] = []
    next_anchor_time = -1.0

    for start_index, first in enumerate(transcript):
        if first.start < next_anchor_time:
            continue

        start = first.start
        chosen: list[TranscriptSegment] = []
        end = first.end

        for segment in transcript[start_index:]:
            if segment.start - start > max_seconds:
                break
            proposed_end = segment.end
            if proposed_end - start > max_seconds:
                break

            chosen.append(segment)
            end = proposed_end

            duration = end - start
            sentence_complete = segment.text.rstrip().endswith((".", "!", "?"))
            if duration >= target_seconds and sentence_complete:
                break
            if duration >= max_seconds:
                break

        duration = end - start
        if duration < min_seconds:
            continue

        text = " ".join(x.text for x in chosen).strip()
        if len(text.split()) < 28:
            continue

        candidate = Candidate(
            id=f"c{len(candidates) + 1:04d}",
            start=max(0.0, start - 0.6),
            end=end + 0.7,
            text=text,
        )
        candidate.heuristic_score = heuristic_score(
            candidate.text,
            candidate.start,
            candidate.end,
            scenes,
        )
        candidates.append(candidate)

        # Overlapping windows are intentional, but not every transcript fragment.
        next_anchor_time = first.start + max(10.0, target_seconds * 0.38)

    return candidates


def prefilter_candidates(
    candidates: list[Candidate],
    *,
    limit: int,
) -> list[Candidate]:
    if len(candidates) <= limit:
        return list(candidates)

    ranked = sorted(
        candidates,
        key=lambda item: item.heuristic_score,
        reverse=True,
    )

    # Reserve most slots for signal strength, while keeping some timeline coverage.
    strong_count = max(1, int(limit * 0.75))
    selected = ranked[:strong_count]
    selected_ids = {item.id for item in selected}

    remaining = [x for x in candidates if x.id not in selected_ids]
    remaining.sort(key=lambda x: x.start)

    coverage_slots = limit - len(selected)
    if coverage_slots > 0 and remaining:
        step = max(1, math.ceil(len(remaining) / coverage_slots))
        for index in range(0, len(remaining), step):
            selected.append(remaining[index])
            if len(selected) >= limit:
                break

    return sorted(selected, key=lambda item: item.start)


def expand_context(
    candidate: Candidate,
    transcript: list[TranscriptSegment],
    *,
    max_seconds: float,
) -> Candidate:
    if not transcript:
        return candidate

    indices = [
        index
        for index, segment in enumerate(transcript)
        if segment.end >= candidate.start and segment.start <= candidate.end
    ]
    if not indices:
        return candidate

    first_i, last_i = min(indices), max(indices)
    current_duration = candidate.duration

    first_text = transcript[first_i].text.strip().lower()
    if (
        first_i > 0
        and first_text.startswith(CONNECTIVE_STARTS)
        and current_duration < max_seconds - 4
    ):
        candidate.start = max(0.0, transcript[first_i - 1].start - 0.4)
        candidate.text = (
            transcript[first_i - 1].text + " " + candidate.text
        ).strip()

    last_text = transcript[last_i].text.strip()
    if (
        last_i + 1 < len(transcript)
        and not last_text.endswith((".", "!", "?"))
        and candidate.duration < max_seconds - 4
    ):
        candidate.end = transcript[last_i + 1].end + 0.4
        candidate.text = (
            candidate.text + " " + transcript[last_i + 1].text
        ).strip()

    return candidate


def apply_final_score(candidate: Candidate, focus: str) -> None:
    weights = FOCUS_WEIGHTS.get(focus, FOCUS_WEIGHTS["Balanced"])
    scores = candidate.scores
    candidate.final_score = round(
        sum(
            float(scores.get(name, 0.0)) * weight
            for name, weight in weights.items()
        ),
        2,
    )


def overlap_ratio(a: Candidate, b: Candidate) -> float:
    overlap = max(0.0, min(a.end, b.end) - max(a.start, b.start))
    if overlap <= 0:
        return 0.0
    shorter = max(0.001, min(a.duration, b.duration))
    return overlap / shorter


def deduplicate_ranked(
    candidates: list[Candidate],
    *,
    overlap_threshold: float = 0.68,
) -> list[Candidate]:
    result: list[Candidate] = []
    for candidate in sorted(
        candidates,
        key=lambda item: item.final_score,
        reverse=True,
    ):
        if any(
            overlap_ratio(candidate, kept) >= overlap_threshold
            for kept in result
        ):
            continue
        result.append(candidate)
    return result
