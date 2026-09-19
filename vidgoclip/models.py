from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class TranscriptWord:
    start: float
    end: float
    word: str
    probability: float = 1.0


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str


@dataclass
class Scene:
    start: float
    end: float


@dataclass
class Candidate:
    id: str
    start: float
    end: float
    text: str
    heuristic_score: float = 0.0
    title: str = ""
    reason: str = ""
    visual_description: str = ""
    audio_description: str = ""
    scores: dict[str, float] = field(default_factory=dict)
    final_score: float = 0.0

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Candidate":
        # Backward compatibility with v0.1 caches.
        value = dict(value)
        value.setdefault("audio_description", "")
        return cls(**value)


@dataclass
class AnalysisResult:
    video_path: str
    duration: float
    transcript: list[TranscriptSegment]
    words: list[TranscriptWord]
    scenes: list[Scene]
    candidates: list[Candidate]
    cache_key: str
    model_notes: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_path": self.video_path,
            "duration": self.duration,
            "transcript": [asdict(x) for x in self.transcript],
            "words": [asdict(x) for x in self.words],
            "scenes": [asdict(x) for x in self.scenes],
            "candidates": [x.to_dict() for x in self.candidates],
            "cache_key": self.cache_key,
            "model_notes": self.model_notes,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "AnalysisResult":
        return cls(
            video_path=str(value["video_path"]),
            duration=float(value["duration"]),
            transcript=[TranscriptSegment(**x) for x in value.get("transcript", [])],
            words=[TranscriptWord(**x) for x in value.get("words", [])],
            scenes=[Scene(**x) for x in value.get("scenes", [])],
            candidates=[Candidate.from_dict(x) for x in value.get("candidates", [])],
            cache_key=str(value.get("cache_key", "")),
            model_notes=dict(value.get("model_notes", {})),
        )
