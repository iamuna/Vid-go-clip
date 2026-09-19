from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import wave

import numpy as np

from .config import CACHE_DIR
from .media import run_process
from .models import Candidate


@dataclass
class AudioTimeline:
    times: np.ndarray
    rms: np.ndarray
    flux: np.ndarray
    zcr: np.ndarray


def _normalize(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    low = float(np.percentile(values, 20))
    high = float(np.percentile(values, 95))
    if high <= low + 1e-9:
        return np.zeros_like(values, dtype=np.float32)
    normalized = (values - low) / (high - low)
    return np.clip(normalized, 0.0, 1.0).astype(np.float32)


def _extract_wav(video_path: Path, cache_key: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    wav_path = CACHE_DIR / f"audio-{cache_key}.wav"
    if wav_path.exists() and wav_path.stat().st_size > 1024:
        return wav_path

    result = run_process(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(wav_path),
        ]
    )
    if result.returncode != 0:
        raise RuntimeError(
            result.stderr.strip() or "Could not extract audio for analysis."
        )
    return wav_path


def analyze_audio(video_path: Path, cache_key: str) -> AudioTimeline | None:
    try:
        wav_path = _extract_wav(video_path, cache_key)
        with wave.open(str(wav_path), "rb") as handle:
            sample_rate = handle.getframerate()
            channels = handle.getnchannels()
            frames = handle.readframes(handle.getnframes())
    except Exception:
        return None

    samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    samples /= 32768.0

    frame_size = max(256, int(sample_rate * 0.05))
    hop = max(128, int(sample_rate * 0.025))
    if samples.size < frame_size:
        return None

    window = np.hanning(frame_size).astype(np.float32)
    rms_values: list[float] = []
    zcr_values: list[float] = []
    flux_values: list[float] = []
    times: list[float] = []
    previous_spectrum: np.ndarray | None = None

    for start in range(0, samples.size - frame_size + 1, hop):
        frame = samples[start : start + frame_size]
        rms = float(np.sqrt(np.mean(frame * frame) + 1e-12))
        signs = np.signbit(frame)
        zcr = float(np.mean(signs[1:] != signs[:-1]))

        spectrum = np.abs(np.fft.rfft(frame * window))
        spectrum /= float(spectrum.sum() + 1e-9)
        if previous_spectrum is None:
            flux = 0.0
        else:
            delta = spectrum - previous_spectrum
            flux = float(np.maximum(delta, 0.0).sum())
        previous_spectrum = spectrum

        times.append((start + frame_size * 0.5) / sample_rate)
        rms_values.append(rms)
        zcr_values.append(zcr)
        flux_values.append(flux)

    return AudioTimeline(
        times=np.asarray(times, dtype=np.float32),
        rms=_normalize(np.asarray(rms_values, dtype=np.float32)),
        flux=_normalize(np.asarray(flux_values, dtype=np.float32)),
        zcr=_normalize(np.asarray(zcr_values, dtype=np.float32)),
    )


def score_candidate_audio(
    candidate: Candidate,
    timeline: AudioTimeline | None,
) -> None:
    if timeline is None or timeline.times.size == 0:
        candidate.scores.setdefault("audio", 25.0)
        candidate.audio_description = "Audio analysis unavailable."
        return

    mask = (
        (timeline.times >= candidate.start)
        & (timeline.times <= candidate.end)
    )
    if not np.any(mask):
        candidate.scores["audio"] = 20.0
        candidate.audio_description = "No usable audio window."
        return

    rms = timeline.rms[mask]
    flux = timeline.flux[mask]
    zcr = timeline.zcr[mask]

    loud = float(np.percentile(rms, 78)) * 100.0
    transient = float(np.percentile(flux, 75)) * 100.0
    noisiness = float(np.percentile(zcr, 70)) * 100.0
    modulation = float(np.std(rms)) * 180.0

    shout_like = min(100.0, loud * 0.78 + noisiness * 0.22)
    applause_like = min(
        100.0,
        transient * 0.55 + loud * 0.30 + noisiness * 0.15,
    )
    laughter_like = min(
        100.0,
        modulation * 0.40
        + transient * 0.25
        + loud * 0.20
        + noisiness * 0.15,
    )

    audio_score = max(
        loud * 0.65 + transient * 0.35,
        shout_like,
        applause_like * 0.92,
        laughter_like * 0.90,
    )
    candidate.scores["audio"] = round(min(100.0, audio_score), 2)

    hints: list[str] = []
    if shout_like >= 68:
        hints.append("strong/loud speech")
    if applause_like >= 70:
        hints.append("transient applause-like energy")
    if laughter_like >= 68:
        hints.append("rhythmic laughter-like energy")
    if not hints and loud >= 65:
        hints.append("elevated audio energy")
    if not hints:
        hints.append("mostly steady audio")

    candidate.audio_description = (
        ", ".join(hints)
        + f" • loudness {loud:.0f} • transients {transient:.0f}"
    )
