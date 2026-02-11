import json
import tempfile
import unittest
from pathlib import Path

from src.ml.predictive_model import (
    build_feature_row_for_inference,
    build_supervised_examples,
    load_model_artifact,
    predict_from_artifact,
    run_model_search,
    train_and_evaluate,
    train_evaluate_and_save,
)


class PredictiveModelTests(unittest.TestCase):
    def _write_trace(self, rows):
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
        path = Path(tmp.name)
        try:
            for row in rows:
                tmp.write(json.dumps(row) + "\n")
        finally:
            tmp.close()
        return path

    def _base_row(self, ts, ticker, price, trend, sentiment, rel_volume=1.0):
        return {
            "ts": ts,
            "ticker": ticker,
            "price": price,
            "score": trend * 0.7 + sentiment * 0.3,
            "confidence": 0.7,
            "conflict_ratio": 0.2,
            "signals": {
                "trend": {"value": trend, "confidence": 0.9, "quality_ok": True},
                "sentiment": {"value": sentiment, "confidence": 0.8, "quality_ok": True},
                "events": {"value": 0.1, "confidence": 0.6, "quality_ok": True},
                "social": {"value": 0.0, "confidence": 0.4, "quality_ok": True},
                "market": {"value": 0.0, "confidence": 0.4, "quality_ok": True},
                "micro": {"value": 0.0, "confidence": 0.5, "quality_ok": True},
                "dip": {"value": 0.0, "confidence": 0.4, "quality_ok": True},
                "volatility": {"value": -0.1, "confidence": 0.5, "quality_ok": True},
            },
            "risk_context": {
                "rel_volume": rel_volume,
                "atr_pct": 0.02,
                "data_gap_ratio": 0.0,
                "portfolio_drawdown": 0.03,
                "portfolio_avg_correlation": 0.25,
                "ticker_sector_allocation": 0.12,
                "sentiment_variance": 0.15,
                "sentiment_article_count": 8,
                "social_post_count": 7,
                "market_news_count": 9,
            },
        }

    def test_build_supervised_examples_includes_sentiment_as_feature(self):
        rows = [
            self._base_row("2025-01-01T10:00:00+00:00", "AAA", 100, 0.6, 0.2),
            self._base_row("2025-01-02T10:00:00+00:00", "AAA", 101, 0.6, 0.1),
            self._base_row("2025-01-03T10:00:00+00:00", "AAA", 102, 0.6, 0.0),
        ]
        examples = build_supervised_examples(rows, horizon=1)
        self.assertEqual(len(examples), 2)
        self.assertIn("sig_sentiment_value", examples[0].features)
        self.assertIn("sig_trend_value", examples[0].features)
        self.assertIn("risk_rel_volume", examples[0].features)

    def test_train_and_evaluate_reports_required_metrics(self):
        rows = []
        # Create enough rows across two tickers for train/test split.
        for i in range(45):
            day = i + 1
            rows.append(self._base_row(f"2025-02-{day:02d}T10:00:00+00:00", "AAA", 100 + (i * 0.6), 0.7, 0.2))
            rows.append(self._base_row(f"2025-02-{day:02d}T10:00:00+00:00", "BBB", 120 - (i * 0.5), -0.7, -0.2))

        path = self._write_trace(rows)
        try:
            result = train_and_evaluate(str(path), horizon=5, train_ratio=0.8, model_family="random_forest")
            # If sklearn is unavailable in environment, test the graceful status.
            if result.get("status") == "missing_deps":
                self.assertIn("detail", result)
                return

            self.assertEqual(result.get("status"), "ok")
            self.assertIn("precision", result["classification"])
            self.assertIn("recall", result["classification"])
            self.assertIn("roc_auc", result["classification"])
            self.assertIn("profit_factor", result["profit"])
            self.assertIn("avg_trade_expectancy", result["profit"])
            self.assertTrue(any(x["feature"] == "sig_sentiment_value" for x in result.get("top_features", [])))
        finally:
            path.unlink(missing_ok=True)

    def test_train_save_and_infer_artifact(self):
        rows = []
        for i in range(45):
            day = i + 1
            rows.append(self._base_row(f"2025-03-{day:02d}T10:00:00+00:00", "AAA", 100 + (i * 0.5), 0.6, 0.1))
            rows.append(self._base_row(f"2025-03-{day:02d}T10:00:00+00:00", "BBB", 130 - (i * 0.4), -0.5, -0.1))

        trace_path = self._write_trace(rows)
        model_file = Path(tempfile.NamedTemporaryFile(suffix=".pkl", delete=False).name)
        try:
            result = train_evaluate_and_save(str(trace_path), str(model_file), horizon=5, train_ratio=0.8)
            if result.get("status") == "missing_deps":
                self.assertIn("detail", result)
                return
            self.assertEqual(result.get("status"), "ok")
            self.assertTrue(model_file.exists())

            artifact = load_model_artifact(str(model_file))
            self.assertIsNotNone(artifact)
            feat = build_feature_row_for_inference(
                signals=rows[0]["signals"],
                risk_context=rows[0]["risk_context"],
                score=rows[0]["score"],
                conflict_ratio=rows[0]["conflict_ratio"],
                confidence=rows[0]["confidence"],
            )
            pred = predict_from_artifact(artifact, feat)
            self.assertIsNotNone(pred)
            self.assertIn("prob_up", pred)
            self.assertIn("expected_return", pred)
        finally:
            trace_path.unlink(missing_ok=True)
            model_file.unlink(missing_ok=True)

    def test_model_search_returns_best_candidate(self):
        rows = []
        for i in range(45):
            day = i + 1
            rows.append(self._base_row(f"2025-04-{day:02d}T10:00:00+00:00", "AAA", 100 + (i * 0.4), 0.6, 0.1))
            rows.append(self._base_row(f"2025-04-{day:02d}T10:00:00+00:00", "BBB", 120 - (i * 0.4), -0.6, -0.1))
        path = self._write_trace(rows)
        try:
            result = run_model_search(
                trace_path=str(path),
                horizons=[5, 10],
                model_families=["random_forest", "gradient_boosting"],
                train_ratio=0.8,
                save_best_artifact="",
            )
            if result.get("status") == "missing_deps":
                self.assertIn("detail", result)
                return
            self.assertEqual(result.get("status"), "ok")
            self.assertIn("leaderboard", result)
            self.assertIsNotNone(result.get("best"))
            self.assertGreaterEqual(len(result.get("leaderboard", [])), 2)
        finally:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
