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


if __name__ == "__main__":
    unittest.main()
