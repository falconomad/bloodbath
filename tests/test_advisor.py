import unittest
from unittest.mock import patch

import pandas as pd

from src import advisor


class AdvisorCycleTests(unittest.TestCase):
    def test_run_top20_cycle_uses_prefetched_dataframe_without_truthiness_eval(self):
        data = pd.DataFrame({"Close": [100.0, 101.0, 102.5]})
        candidates = {
            "AAA": {
                "data": data,
                "dip_score": 0.0,
                "drawdown": 0.25,
                "stabilized": True,
                "volatility_penalty": 0.0,
            }
        }

        with patch("src.advisor._build_candidate_list", return_value=candidates), patch(
            "src.advisor.get_price_data"
        ) as mock_get_price_data, patch("src.advisor.get_company_news", return_value=[]), patch(
            "src.advisor.generate_recommendation", return_value={"composite_score": 0.0}
        ), patch.object(advisor.top20_manager, "step") as mock_step, patch.object(
            advisor.top20_manager, "history_df", return_value=pd.DataFrame()
        ), patch.object(
            advisor.top20_manager, "transactions_df", return_value=pd.DataFrame()
        ), patch.object(
            advisor.top20_manager, "position_snapshot_df", return_value=pd.DataFrame()
        ):
            advisor.run_top20_cycle()

        mock_get_price_data.assert_not_called()
        mock_step.assert_called_once()


class AdvisorSignalTests(unittest.TestCase):
    def test_run_top20_cycle_with_signals_returns_analyses(self):
        data = pd.DataFrame({"Close": [100.0, 101.0, 102.5]})
        candidates = {
            "AAA": {
                "data": data,
                "dip_score": 0.2,
                "drawdown": 0.25,
                "stabilized": True,
                "volatility_penalty": 0.0,
            }
        }

        with patch("src.advisor._build_candidate_list", return_value=candidates), patch(
            "src.advisor.get_company_news", return_value=[]
        ), patch(
            "src.advisor.generate_recommendation",
            return_value={
                "composite_score": 0.9,
                "signal_confidence": 0.8,
                "sentiment": 0.0,
                "decision": "BUY",
                "position_size": 0.1,
                "decision_reasons": [],
            },
        ), patch.object(
            advisor.top20_manager, "step"
        ), patch.object(advisor.top20_manager, "history_df", return_value=pd.DataFrame()), patch.object(
            advisor.top20_manager, "transactions_df", return_value=pd.DataFrame()
        ), patch.object(advisor.top20_manager, "position_snapshot_df", return_value=pd.DataFrame()):
            history, transactions, positions, analyses = advisor.run_top20_cycle_with_signals()

        self.assertTrue(isinstance(history, pd.DataFrame))
        self.assertTrue(isinstance(transactions, pd.DataFrame))
        self.assertTrue(isinstance(positions, pd.DataFrame))
        self.assertEqual(len(analyses), 1)
        self.assertEqual(analyses[0]["ticker"], "AAA")
        self.assertEqual(analyses[0]["decision"], "BUY")

    def test_build_candidate_list_selects_top_gainers_and_losers(self):
        symbols = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
        price_map = {
            "AAA": pd.DataFrame({"Close": [100.0, 90.0]}),   # -10%
            "BBB": pd.DataFrame({"Close": [100.0, 80.0]}),   # -20%
            "CCC": pd.DataFrame({"Close": [100.0, 95.0]}),   # -5%
            "DDD": pd.DataFrame({"Close": [100.0, 120.0]}),  # +20%
            "EEE": pd.DataFrame({"Close": [100.0, 110.0]}),  # +10%
            "FFF": pd.DataFrame({"Close": [100.0, 105.0]}),  # +5%
        }
        with patch("src.advisor.get_sp500_universe", return_value=symbols), patch(
            "src.advisor.get_bulk_price_data", return_value=price_map
        ):
            candidates = advisor._build_candidate_list(universe_size=6, top_losers=2, top_gainers=2)

        self.assertEqual(set(candidates.keys()), {"BBB", "AAA", "DDD", "EEE"})
        self.assertEqual(candidates["BBB"]["mover_bucket"], "LOSER")
        self.assertEqual(candidates["AAA"]["mover_bucket"], "LOSER")
        self.assertEqual(candidates["DDD"]["mover_bucket"], "GAINER")
        self.assertEqual(candidates["EEE"]["mover_bucket"], "GAINER")


if __name__ == "__main__":
    unittest.main()
