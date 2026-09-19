from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CACHE_DIR = ROOT / "cache"
OUTPUT_DIR = ROOT / "output"
TEMP_DIR = ROOT / "temp"
SETTINGS_FILE = DATA_DIR / "settings.json"

DEFAULT_SETTINGS = {
    "whisper_model": "small",
    "ollama_model": "qwen3-vl:4b",
    "ollama_base_url": "http://127.0.0.1:11434",
    "max_ai_candidates": 36,
    "max_visual_candidates": 12,
    "min_clip_seconds": 18,
    "target_clip_seconds": 42,
    "max_clip_seconds": 75,
    "default_export_count": 5,
    "analysis_focus": "Balanced",
    "vertical_export": False,
    "smart_reframe": True,
    "burn_captions": True,
    "caption_words_per_line": 7,
}

FOCUS_WEIGHTS = {
    "Balanced": {
        "importance": 0.24,
        "controversy": 0.13,
        "interest": 0.24,
        "emotion": 0.10,
        "visual": 0.11,
        "audio": 0.08,
        "context": 0.10,
    },
    "Important": {
        "importance": 0.38,
        "controversy": 0.08,
        "interest": 0.18,
        "emotion": 0.07,
        "visual": 0.07,
        "audio": 0.05,
        "context": 0.17,
    },
    "Controversial": {
        "importance": 0.18,
        "controversy": 0.36,
        "interest": 0.18,
        "emotion": 0.09,
        "visual": 0.05,
        "audio": 0.05,
        "context": 0.09,
    },
    "Interesting": {
        "importance": 0.15,
        "controversy": 0.10,
        "interest": 0.38,
        "emotion": 0.11,
        "visual": 0.11,
        "audio": 0.08,
        "context": 0.09,
    },
    "Emotional": {
        "importance": 0.14,
        "controversy": 0.08,
        "interest": 0.20,
        "emotion": 0.31,
        "visual": 0.08,
        "audio": 0.12,
        "context": 0.07,
    },
}


def ensure_directories() -> None:
    for folder in (DATA_DIR, CACHE_DIR, OUTPUT_DIR, TEMP_DIR):
        folder.mkdir(parents=True, exist_ok=True)


def load_settings() -> dict:
    ensure_directories()
    if not SETTINGS_FILE.exists():
        save_settings(DEFAULT_SETTINGS)
        return DEFAULT_SETTINGS.copy()

    try:
        loaded = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        loaded = {}

    result = DEFAULT_SETTINGS.copy()
    result.update(loaded)
    return result


def save_settings(settings: dict) -> None:
    ensure_directories()
    SETTINGS_FILE.write_text(
        json.dumps(settings, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
