import unittest
from unittest.mock import patch

from src.analysis.events import detect_events, score_events, score_events_details


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

    def test_score_events_includes_upcoming_earnings_signal(self):
        self.assertEqual(score_events([], has_upcoming_earnings=True), 0.12)
        self.assertTrue(detect_events([], has_upcoming_earnings=True))
        details = score_events_details([], has_upcoming_earnings=True)
        self.assertEqual(details["earnings_bonus"], 0.12)

    def test_score_events_handles_empty_partial_and_non_string_items(self):
        headlines = ["", None, 123, "minor product launch"]

        self.assertEqual(score_events(headlines), 0.25)
        self.assertTrue(detect_events(headlines))
        self.assertEqual(score_events([]), 0.0)
        self.assertFalse(detect_events([]))

    @patch("src.analysis.events._classify_headline_topic")
    @patch("src.analysis.events.zero_shot_model", object())
    def test_topic_classification_adjusts_event_score(self, mock_topic):
        mock_topic.side_effect = [
            ("earnings", 0.92),
            ("lawsuits", 0.95),
        ]
        headlines = [
            "Company beats earnings estimates",
            "Company faces legal complaint",
        ]
        score = score_events(headlines)
        # Should include both lexical and topic-level effects, bounded in range.
        self.assertGreaterEqual(score, -1.0)
        self.assertLessEqual(score, 1.0)
        self.assertNotEqual(score, 0.0)

    def test_score_events_details_contains_breakdown(self):
        details = score_events_details(
            ["Company reports earnings beat and record profit", "Company faces lawsuit"],
            has_upcoming_earnings=True,
        )
        self.assertIn("phrase_hits", details)
        self.assertIn("topic_contributions", details)
        self.assertIn("score", details)
        self.assertGreater(details["lexical_total"], 0)


if __name__ == "__main__":
    unittest.main()
