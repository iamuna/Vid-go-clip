import unittest

from vidgoclip.ai import _extract_json, fallback_semantic_scores
from vidgoclip.models import Candidate


class AiTests(unittest.TestCase):
    def test_extract_json_from_fenced_text(self):
        value = _extract_json('''```json
{"candidates": [{"id": "c1"}]}
```''')
        self.assertEqual(value["candidates"][0]["id"], "c1")

    def test_fallback_scores_have_all_dimensions(self):
        candidate = Candidate(
            id="c1",
            start=0,
            end=30,
            text="This is important because people disagree and say it is wrong!",
            heuristic_score=72,
        )
        scores = fallback_semantic_scores(candidate)
        for key in (
            "importance",
            "controversy",
            "interest",
            "emotion",
            "visual",
            "context",
        ):
            self.assertIn(key, scores)
            self.assertGreaterEqual(scores[key], 0)
            self.assertLessEqual(scores[key], 100)


if __name__ == "__main__":
    unittest.main()
