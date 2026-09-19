import unittest

import numpy as np

from vidgoclip.audio_events import AudioTimeline, score_candidate_audio
from vidgoclip.models import Candidate


class AudioEventTests(unittest.TestCase):
    def test_candidate_gets_audio_score_and_description(self):
        timeline = AudioTimeline(
            times=np.linspace(0, 20, 200, dtype=np.float32),
            rms=np.concatenate([
                np.full(100, 0.15, dtype=np.float32),
                np.full(100, 0.90, dtype=np.float32),
            ]),
            flux=np.concatenate([
                np.full(100, 0.10, dtype=np.float32),
                np.full(100, 0.85, dtype=np.float32),
            ]),
            zcr=np.concatenate([
                np.full(100, 0.20, dtype=np.float32),
                np.full(100, 0.70, dtype=np.float32),
            ]),
        )
        quiet = Candidate("q", 0, 8, "quiet")
        loud = Candidate("l", 12, 19, "loud")

        score_candidate_audio(quiet, timeline)
        score_candidate_audio(loud, timeline)

        self.assertGreater(loud.scores["audio"], quiet.scores["audio"])
        self.assertTrue(loud.audio_description)


if __name__ == "__main__":
    unittest.main()
