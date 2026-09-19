from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .config import CACHE_DIR
from .models import AnalysisResult


def analysis_key(
    video_key: str,
    *,
    whisper_model: str,
    ollama_model: str,
) -> str:
    payload = f"{video_key}|{whisper_model}|{ollama_model}|v2".encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:24]


def analysis_path(key: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"analysis-{key}.json"


def load_analysis(key: str) -> AnalysisResult | None:
    path = analysis_path(key)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return AnalysisResult.from_dict(data)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None


def save_analysis(result: AnalysisResult) -> Path:
    path = analysis_path(result.cache_key)
    path.write_text(
        json.dumps(result.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path
