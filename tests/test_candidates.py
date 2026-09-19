import unittest

from vidgoclip.candidates import (
    apply_final_score,
    build_candidates,
    deduplicate_ranked,
    expand_context,
    refine_word_boundaries,
)
from vidgoclip.models import Candidate, Scene, TranscriptSegment, TranscriptWord


class CandidateTests(unittest.TestCase):
    def _transcript(self):
        return [
            TranscriptSegment(0, 6, "And this is the setup for the story."),
            TranscriptSegment(6, 13, "The important part happens because the system failed."),
            TranscriptSegment(13, 21, "People strongly disagreed about what happened next."),
            TranscriptSegment(21, 30, "That argument changed the decision completely."),
            TranscriptSegment(30, 39, "This is the conclusion and why it matters."),
            TranscriptSegment(39, 47, "A separate topic begins here."),
            TranscriptSegment(47, 56, "It has enough words to remain a valid candidate."),
        ]

    def test_build_candidates_respects_duration(self):
        candidates = build_candidates(
            self._transcript(),
            [Scene(0, 25), Scene(25, 60)],
            min_seconds=18,
            target_seconds=30,
            max_seconds=45,
        )
        self.assertTrue(candidates)
        for candidate in candidates:
            self.assertGreaterEqual(candidate.duration, 18)
            self.assertLessEqual(candidate.duration, 47)

    def test_context_prepend_updates_text(self):
        transcript = self._transcript()
        candidate = Candidate(
            id="c1",
            start=6,
            end=30,
            text="The important part happens because the system failed. "
                 "People strongly disagreed about what happened next. "
                 "That argument changed the decision completely.",
        )
        # Candidate does not start with a connective here, so force a connective.
        transcript[1].text = "But the important part happens because the system failed."
        candidate.text = (
            transcript[1].text + " " + transcript[2].text + " " + transcript[3].text
        )
        expanded = expand_context(candidate, transcript, max_seconds=50)
        self.assertLess(expanded.start, 6)
        self.assertIn("setup for the story", expanded.text)

    def test_focus_changes_score(self):
        candidate = Candidate(
            id="c1",
            start=0,
            end=30,
            text="test",
            scores={
                "importance": 95,
                "controversy": 10,
                "interest": 50,
                "emotion": 20,
                "visual": 30,
                "context": 80,
            },
        )
        apply_final_score(candidate, "Important")
        important_score = candidate.final_score
        apply_final_score(candidate, "Controversial")
        controversial_score = candidate.final_score
        self.assertGreater(important_score, controversial_score)

    def test_deduplicates_heavy_overlap(self):
        a = Candidate(id="a", start=0, end=40, text="a", final_score=90)
        b = Candidate(id="b", start=5, end=39, text="b", final_score=80)
        c = Candidate(id="c", start=60, end=95, text="c", final_score=70)
        result = deduplicate_ranked([a, b, c])
        self.assertEqual([item.id for item in result], ["a", "c"])


    def test_word_boundary_refinement_uses_sentence_and_pause(self):
        words = [
            TranscriptWord(0.0, 0.4, "Earlier"),
            TranscriptWord(0.45, 0.9, "setup."),
            TranscriptWord(2.0, 2.3, "This"),
            TranscriptWord(2.35, 2.7, "is"),
            TranscriptWord(2.75, 3.1, "the"),
            TranscriptWord(3.15, 3.7, "important"),
            TranscriptWord(3.75, 4.2, "part."),
            TranscriptWord(5.4, 5.9, "Next"),
            TranscriptWord(5.95, 6.4, "thought."),
        ]
        candidate = Candidate(
            id="c1",
            start=2.4,
            end=4.0,
            text="is the important part",
        )
        refined = refine_word_boundaries(
            candidate,
            words,
            max_seconds=10,
        )
        self.assertLess(refined.start, 2.1)
        self.assertGreater(refined.end, 4.2)
        self.assertLess(refined.end, 5.4)


if __name__ == "__main__":
    unittest.main()
