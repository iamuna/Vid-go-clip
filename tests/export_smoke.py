from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vidgoclip.exporter import export_clip
from vidgoclip.models import Candidate, TranscriptSegment, TranscriptWord

TEMP = ROOT / "temp" / "ci-export"
OUTPUT = ROOT / "artifacts"


def run(args: list[str]) -> None:
    result = subprocess.run(
        args,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr)


def main() -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise RuntimeError("FFmpeg is required.")

    TEMP.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    source = TEMP / "source.mp4"

    run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=1280x720:rate=30:duration=5",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=44100:duration=5",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(source),
        ]
    )

    candidate = Candidate(
        id="c0001",
        start=1.0,
        end=4.0,
        text="Smoke test clip.",
        title="Smoke test",
        final_score=90,
    )
    words = [
        TranscriptWord(1.15, 1.45, "Smart"),
        TranscriptWord(1.50, 1.85, "vertical"),
        TranscriptWord(1.90, 2.20, "caption"),
        TranscriptWord(2.25, 2.55, "smoke"),
        TranscriptWord(2.60, 2.95, "test."),
    ]
    transcript = [
        TranscriptSegment(1.15, 2.95, "Smart vertical caption smoke test.")
    ]

    exported = export_clip(
        source,
        candidate,
        destination_dir=OUTPUT,
        vertical=True,
        smart_reframe=True,
        burn_captions=True,
        words=words,
        transcript=transcript,
        caption_words_per_line=3,
    )
    final = OUTPUT / "vid-go-clip-export-smoke.mp4"
    if final.exists():
        final.unlink()
    exported.replace(final)

    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,codec_name,codec_type",
            "-of",
            "default=noprint_wrappers=1",
            str(final),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    text = probe.stdout
    assert "width=1080" in text, text
    assert "height=1920" in text, text
    assert "codec_name=h264" in text, text

    audio_probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_name",
            "-of",
            "default=noprint_wrappers=1",
            str(final),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "codec_name=aac" in audio_probe.stdout, audio_probe.stdout
    print(final)


if __name__ == "__main__":
    main()
