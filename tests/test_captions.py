import tempfile
import unittest
from pathlib import Path

from vidgoclip.captions import write_ass_captions
from vidgoclip.models import Candidate, TranscriptSegment, TranscriptWord


class CaptionTests(unittest.TestCase):
    def test_ass_uses_word_timing(self):
        candidate = Candidate("c1", 10.0, 14.0, "Hello world this is timed.")
        words = [
            TranscriptWord(10.2, 10.5, "Hello"),
            TranscriptWord(10.6, 10.9, "world"),
            TranscriptWord(11.0, 11.2, "this"),
            TranscriptWord(11.3, 11.5, "is"),
            TranscriptWord(11.6, 12.0, "timed."),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = write_ass_captions(
                Path(directory) / "captions.ass",
                candidate,
                words=words,
                transcript=[],
                words_per_line=3,
            )
            self.assertIsNotNone(path)
            text = path.read_text(encoding="utf-8-sig")
            self.assertIn("Hello world this", text)
            self.assertIn("Dialogue:", text)


if __name__ == "__main__":
    unittest.main()
