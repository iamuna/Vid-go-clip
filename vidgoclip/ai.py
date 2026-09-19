from __future__ import annotations

import base64
import json
from pathlib import Path

import requests

from .models import Candidate


def _clamp_score(value) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return round(max(0.0, min(100.0, number)), 2)


def _extract_json(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].lstrip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end < start:
        raise RuntimeError("Local AI did not return valid JSON.")
    return json.loads(cleaned[start : end + 1])


def ollama_ready(base_url: str) -> bool:
    try:
        response = requests.get(
            f"{base_url.rstrip('/')}/api/tags",
            timeout=2,
        )
        return response.ok
    except requests.RequestException:
        return False


def _chat_json(
    *,
    base_url: str,
    model: str,
    prompt: str,
    images: list[Path] | None = None,
    timeout: int = 300,
) -> dict:
    image_payload: list[str] = []
    for path in images or []:
        image_payload.append(
            base64.b64encode(path.read_bytes()).decode("ascii")
        )

    message = {
        "role": "user",
        "content": prompt,
    }
    if image_payload:
        message["images"] = image_payload

    try:
        response = requests.post(
            f"{base_url.rstrip('/')}/api/chat",
            json={
                "model": model,
                "messages": [message],
                "stream": False,
                "format": "json",
                "options": {
                    "temperature": 0.15,
                },
            },
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise RuntimeError(
            "Local Ollama AI is not reachable. Run setup.bat or start Ollama."
        ) from exc

    if response.status_code == 404:
        raise RuntimeError(
            f"Ollama model '{model}' is unavailable. Run: ollama pull {model}"
        )
    if not response.ok:
        raise RuntimeError(
            f"Ollama request failed ({response.status_code}): "
            f"{response.text[:600]}"
        )

    payload = response.json()
    content = str((payload.get("message") or {}).get("content") or "")
    return _extract_json(content)


def fallback_semantic_scores(candidate: Candidate) -> dict[str, float]:
    base = max(20.0, min(82.0, candidate.heuristic_score))
    lower = candidate.text.lower()

    controversy_markers = (
        "wrong", "disagree", "lie", "false", "ban", "scam",
        "fraud", "ridiculous", "should not", "shouldn't",
    )
    emotion_markers = (
        "love", "hate", "angry", "fear", "shocked", "wow",
        "terrible", "amazing", "sorry", "laugh",
    )
    controversy = min(
        85.0,
        20.0 + 12.0 * sum(x in lower for x in controversy_markers),
    )
    emotion = min(
        85.0,
        20.0 + 11.0 * sum(x in lower for x in emotion_markers),
    )

    return {
        "importance": base,
        "controversy": controversy,
        "interest": min(90.0, base + 5.0),
        "emotion": emotion,
        "visual": 35.0,
        "context": 62.0,
    }


def score_semantic_candidates(
    candidates: list[Candidate],
    *,
    base_url: str,
    model: str,
    batch_size: int = 6,
) -> None:
    for offset in range(0, len(candidates), batch_size):
        batch = candidates[offset : offset + batch_size]

        entries = []
        for candidate in batch:
            entries.append(
                {
                    "id": candidate.id,
                    "start_seconds": round(candidate.start, 2),
                    "end_seconds": round(candidate.end, 2),
                    "transcript": candidate.text[:5000],
                }
            )

        prompt = f"""
You are the editorial analysis engine for a video-clipping application.

Evaluate each candidate moment independently.

Definitions:
- importance: consequence, central insight, major revelation, useful conclusion,
  key decision, strong evidence, or information a viewer would care about.
- controversy: likelihood the statement/conversation would provoke meaningful
  disagreement, argument, challenge, or debate. This is NOT a truth score and
  is NOT a moral judgment.
- interest: curiosity, surprise, novelty, humor, memorable wording, strong
  story/payoff, or standalone shareability.
- emotion: visible in the language as strong feeling, tension, humor, shock,
  vulnerability, excitement, anger, sadness, etc.
- context: whether the excerpt contains enough setup and payoff to make sense
  on its own and does not obviously begin/end in the middle of a thought.

Use 0-100 scores. Avoid inflating every score. Ordinary filler should score low.

Return ONLY a JSON object:
{{
  "candidates": [
    {{
      "id": "c0001",
      "importance": 0,
      "controversy": 0,
      "interest": 0,
      "emotion": 0,
      "context": 0,
      "title": "short factual suggested title",
      "reason": "one concise explanation of why this is or is not clip-worthy"
    }}
  ]
}}

Candidates:
{json.dumps(entries, ensure_ascii=False)}
""".strip()

        try:
            data = _chat_json(
                base_url=base_url,
                model=model,
                prompt=prompt,
                timeout=360,
            )
            returned = {
                str(item.get("id")): item
                for item in data.get("candidates", [])
                if isinstance(item, dict)
            }
        except Exception:
            returned = {}

        for candidate in batch:
            item = returned.get(candidate.id)
            if item is None:
                candidate.scores.update(fallback_semantic_scores(candidate))
                if not candidate.title:
                    candidate.title = f"Moment {candidate.id}"
                if not candidate.reason:
                    candidate.reason = (
                        "Local semantic AI was unavailable; ranking uses "
                        "the fallback signal score."
                    )
                continue

            candidate.scores.update(
                {
                    "importance": _clamp_score(item.get("importance")),
                    "controversy": _clamp_score(item.get("controversy")),
                    "interest": _clamp_score(item.get("interest")),
                    "emotion": _clamp_score(item.get("emotion")),
                    "context": _clamp_score(item.get("context")),
                    "visual": 35.0,
                }
            )
            candidate.title = str(item.get("title") or "").strip()[:100]
            candidate.reason = str(item.get("reason") or "").strip()[:700]


def score_visual_candidate(
    candidate: Candidate,
    *,
    frame_paths: list[Path],
    motion: float,
    base_url: str,
    model: str,
) -> None:
    if not frame_paths:
        candidate.scores["visual"] = max(20.0, min(65.0, motion))
        return

    prompt = f"""
These images are chronological sampled frames from ONE candidate video clip.

Transcript:
{candidate.text[:3500]}

Evaluate only what the frames visibly support. Do not invent unseen actions.

Score visual_value 0-100 based on whether the clip contains useful visible
action, reaction, demonstration, object/event, change, evidence, unusual
setting, or other imagery that improves the clip. A static ordinary talking
head can still be usable but should not automatically score high.

Measured frame-to-frame motion signal: {motion:.1f}/100.
Treat that as one clue, not ground truth.

Return ONLY JSON:
{{
  "visual_value": 0,
  "visual_description": "brief description of what is visibly happening",
  "visual_reason": "brief reason for the score"
}}
""".strip()

    try:
        data = _chat_json(
            base_url=base_url,
            model=model,
            prompt=prompt,
            images=frame_paths,
            timeout=360,
        )
        model_score = _clamp_score(data.get("visual_value"))
        candidate.scores["visual"] = round(
            model_score * 0.78 + motion * 0.22,
            2,
        )
        description = str(data.get("visual_description") or "").strip()
        reason = str(data.get("visual_reason") or "").strip()
        candidate.visual_description = description[:600]
        if reason:
            candidate.reason = (
                f"{candidate.reason} Visual: {reason}"
            ).strip()[:1000]
    except Exception:
        candidate.scores["visual"] = max(
            20.0,
            min(75.0, 30.0 + motion * 0.55),
        )
