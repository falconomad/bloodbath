import unittest

from src.analysis.events import calculate_event_score, detect_events


class EventScoringTests(unittest.TestCase):
    def test_positive_event_score(self):
        news = [
            "Company reports earnings beat and guidance raised",
            "Analyst upgrade after strong profit growth",
        ]
        score = calculate_event_score(news)
        self.assertGreater(score, 0)
        self.assertTrue(detect_events(news))

    def test_negative_event_score(self):
        news = [
            "Company faces lawsuit after product recall",
            "Analyst downgrade as guidance cut follows earnings miss",
        ]
        score = calculate_event_score(news)
        self.assertLess(score, 0)
        self.assertTrue(detect_events(news))

    def test_no_signal_returns_zero(self):
        news = ["Company held annual conference and introduced roadmap"]
        self.assertEqual(calculate_event_score(news), 0.0)
        self.assertFalse(detect_events(news))

    def test_score_is_bounded(self):
        noisy_news = ["earnings beat guidance raised upgrade profit" for _ in range(50)]
        score = calculate_event_score(noisy_news)
        self.assertLessEqual(score, 1.0)
        self.assertGreaterEqual(score, -1.0)


if __name__ == "__main__":
    unittest.main()
