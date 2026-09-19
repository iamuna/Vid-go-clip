from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vidgoclip.exporter import export_clip
from vidgoclip.models import Candidate

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
    exported = export_clip(
        source,
        candidate,
        destination_dir=OUTPUT,
        vertical=True,
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
            "stream=width,height,codec_name",
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
    print(final)


if __name__ == "__main__":
    main()
