from pathlib import Path
from datetime import datetime
import sys
from zoneinfo import ZoneInfo

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.api.data_fetcher import get_bulk_price_data
from src.backtest.sweep import run_parameter_sweep
from src.core.sp500_list import TOP20
from src.db import init_db, save_backtest_sweep_results


def main():
    run_id = datetime.now(ZoneInfo("Europe/Paris")).strftime("%Y%m%d-%H%M%S")
    price_map = get_bulk_price_data(TOP20, period="2y", interval="1d")
    table = run_parameter_sweep(
        price_map,
        buy_thresholds=[0.55, 0.7, 0.85, 1.0],
        sell_thresholds=[-0.7, -0.85, -1.0],
        min_buy_scores=[0.45, 0.6, 0.75],
        slippage_bps_values=[5.0],
        fee_bps_values=[1.0],
        max_drawdown_limit_pct=-35.0,
        top_k=10,
    )

    if table.empty:
        print("[sweep] no parameter sets passed constraints")
        return

    table = table.copy()
    table.insert(0, "rank", range(1, len(table) + 1))

    print("[sweep] top parameter sets")
    print(table.to_string(index=False))

    try:
        init_db()
        save_backtest_sweep_results(run_id, table.to_dict(orient="records"))
        print(f"[sweep] persisted rows={len(table)} run_id={run_id}")
    except Exception as exc:
        print(f"[sweep] persistence skipped ({exc})")


if __name__ == "__main__":
    main()
