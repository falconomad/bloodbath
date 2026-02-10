import unittest
from unittest.mock import patch

from src.analysis import sentiment


class SentimentAnalysisTests(unittest.TestCase):
    def test_prepare_headlines_dedupes_and_filters_empty(self):
        headlines = ["  Strong growth ahead  ", "", None, "strong growth ahead", "Profit jumps"]
        prepared = sentiment._prepare_headlines(headlines)

        self.assertEqual(prepared, ["Strong growth ahead", "Profit jumps"])

    @patch("src.analysis.sentiment.sentiment_model", None)
    def test_lexical_fallback_handles_weighted_phrases(self):
        news = ["Company posts guidance raise after strong profit growth"]
        score = sentiment.analyze_news_sentiment(news)
        self.assertGreater(score, 0)

    @patch("src.analysis.sentiment.sentiment_model")
    def test_model_scoring_centers_low_confidence_predictions(self, mock_model):
        mock_model.return_value = [
            {"label": "positive", "score": 0.55},
            {"label": "negative", "score": 0.54},
            {"label": "neutral", "score": 0.99},
        ]

        score = sentiment.analyze_news_sentiment(["a", "b", "c"])

        # Near-canceling low-confidence labels should stay close to neutral.
        self.assertLess(abs(score), 0.03)

    @patch("src.analysis.sentiment.sentiment_model", None)
    def test_details_support_structured_news_and_source_weights(self):
        news = [
            {"headline": "Strong growth and record profit", "source": "Reuters", "datetime": "2026-02-10T08:00:00Z"},
            {"headline": "Strong growth and record profit", "source": "UnknownBlog", "datetime": "2026-02-10T08:00:00Z"},
        ]
        details = sentiment.analyze_news_sentiment_details(news)
        self.assertGreater(details["score"], 0)
        self.assertEqual(details["article_count"], 1)

    @patch("src.analysis.sentiment.sentiment_model")
    def test_details_detect_mixed_opinions(self, mock_model):
        mock_model.return_value = [
            {"label": "positive", "score": 0.99},
            {"label": "negative", "score": 0.99},
            {"label": "positive", "score": 0.98},
            {"label": "negative", "score": 0.98},
        ]
        details = sentiment.analyze_news_sentiment_details(
            [
                "very bullish",
                "very bearish",
                "bullish upgrade",
                "bearish downgrade",
            ]
        )
        self.assertTrue(details["mixed_opinions"])


if __name__ == "__main__":
    unittest.main()
