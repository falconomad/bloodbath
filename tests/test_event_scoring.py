import unittest

from src.analysis.events import detect_events, score_events


class EventScoringTests(unittest.TestCase):
    def test_score_events_positive_and_negative_mix(self):
        headlines = [
            "Company reports earnings beat and record profit",
            "Company faces lawsuit after product recall",
        ]

        # +0.6 +0.6 -0.8 -0.5 = -0.1
        self.assertEqual(score_events(headlines), -0.1)

    def test_score_events_caps_to_bounds(self):
        bullish = ["earnings beat", "beats earnings", "record profit"]
        bearish = ["bankruptcy", "lawsuit", "investigation"]

        self.assertEqual(score_events(bullish), 1.0)
        self.assertEqual(score_events(bearish), -1.0)

    def test_score_events_handles_empty_partial_and_non_string_items(self):
        headlines = ["", None, 123, "minor product launch"]

        self.assertEqual(score_events(headlines), 0.25)
        self.assertTrue(detect_events(headlines))
        self.assertEqual(score_events([]), 0.0)
        self.assertFalse(detect_events([]))


if __name__ == "__main__":
    unittest.main()
