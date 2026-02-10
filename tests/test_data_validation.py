import unittest

from src.validation.data_validation import (
    validate_earnings_payload,
    validate_micro_features,
    validate_news_headlines,
    validate_price_history,
)


class DataValidationTests(unittest.TestCase):
    def test_validate_price_history_rejects_gaps(self):
        data = {"Close": [100.0] * 20 + [None] * 25}
        q = validate_price_history(data, min_points=40, max_missing_ratio=0.05)
        self.assertFalse(q.valid)
        self.assertEqual(q.reason, "too_many_gaps")

    def test_validate_price_history_accepts_clean_series(self):
        data = {"Close": [100.0 + i for i in range(60)]}
        q = validate_price_history(data, min_points=40, max_missing_ratio=0.05)
        self.assertTrue(q.valid)

    def test_validate_micro_features(self):
        ok, reason, features = validate_micro_features(
            {"available": True, "rel_volume": 1.2, "intraday_return": 0.01, "quality": 0.7}
        )
        self.assertTrue(ok)
        self.assertEqual(reason, "")
        self.assertGreater(features["quality"], 0)

        ok2, reason2, _ = validate_micro_features({"available": True, "rel_volume": 0, "quality": 0.6})
        self.assertFalse(ok2)
        self.assertEqual(reason2, "invalid_rel_volume")

    def test_validate_news_headlines(self):
        ok, reason, count = validate_news_headlines(["a", " ", None, "b"], min_articles=2)
        self.assertTrue(ok)
        self.assertEqual(count, 2)

        ok2, reason2, count2 = validate_news_headlines(["a"], min_articles=2)
        self.assertFalse(ok2)
        self.assertEqual(reason2, "too_few_articles")
        self.assertEqual(count2, 1)

    def test_validate_earnings_payload(self):
        ok, count = validate_earnings_payload([1, 2])
        self.assertTrue(ok)
        self.assertEqual(count, 2)

        ok2, count2 = validate_earnings_payload(None)
        self.assertFalse(ok2)
        self.assertEqual(count2, 0)


if __name__ == "__main__":
    unittest.main()
