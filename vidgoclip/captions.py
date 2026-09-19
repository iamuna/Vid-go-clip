from __future__ import annotations

from pathlib import Path

from .models import Candidate, TranscriptSegment, TranscriptWord


def _ass_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    whole = int(seconds)
    centiseconds = int(round((seconds - whole) * 100))
    if centiseconds >= 100:
        whole += 1
        centiseconds = 0
    hours, rem = divmod(whole, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}.{centiseconds:02d}"


def _escape(text: str) -> str:
    return (
        text.replace("\\", r"\\")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("\n", r"\N")
    )


def _word_groups(
    words: list[TranscriptWord],
    candidate: Candidate,
    words_per_line: int,
) -> list[tuple[float, float, str]]:
    relevant = [
        word
        for word in words
        if word.end > candidate.start and word.start < candidate.end
    ]
    groups: list[tuple[float, float, str]] = []
    for offset in range(0, len(relevant), max(2, words_per_line)):
        chunk = relevant[offset : offset + max(2, words_per_line)]
        if not chunk:
            continue
        text = " ".join(word.word for word in chunk).strip()
        groups.append(
            (
                max(0.0, chunk[0].start - candidate.start),
                max(0.1, chunk[-1].end - candidate.start + 0.08),
                text,
            )
        )
    return groups


def _segment_groups(
    transcript: list[TranscriptSegment],
    candidate: Candidate,
) -> list[tuple[float, float, str]]:
    groups = []
    for segment in transcript:
        if segment.end <= candidate.start or segment.start >= candidate.end:
            continue
        groups.append(
            (
                max(0.0, segment.start - candidate.start),
                max(0.1, min(candidate.end, segment.end) - candidate.start),
                segment.text,
            )
        )
    return groups


def write_ass_captions(
    destination: Path,
    candidate: Candidate,
    *,
    words: list[TranscriptWord],
    transcript: list[TranscriptSegment],
    words_per_line: int = 7,
) -> Path | None:
    groups = _word_groups(words, candidate, words_per_line)
    if not groups:
        groups = _segment_groups(transcript, candidate)
    if not groups:
        return None

    destination.parent.mkdir(parents=True, exist_ok=True)

    header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
ScaledBorderAndShadow: yes
WrapStyle: 2

[V4+ Styles]
Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding
Style: Caption,Arial,66,&H00FFFFFF,&H0000FFFF,&H00101010,&H80000000,-1,0,0,0,100,100,0,0,1,5,1,2,70,70,165,1

[Events]
Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text
"""

    lines = [header]
    for start, end, text in groups:
        lines.append(
            "Dialogue: 0,"
            f"{_ass_time(start)},{_ass_time(end)},"
            "Caption,,0,0,0,,"
            f"{_escape(text)}\n"
        )

    destination.write_text("".join(lines), encoding="utf-8-sig")
    return destination
