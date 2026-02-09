from pathlib import Path
import sys

# Ensure repo root is on sys.path when running as `python worker/backtest_runner.py`.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.backtest.walk_forward import run_backtest_for_top20


def main():
    result = run_backtest_for_top20(period="2y", interval="1d")
    print("[backtest] metrics")
    for k, v in result.metrics.items():
        print(f"  - {k}: {v}")
    print(f"[backtest] history_rows={len(result.history)} transactions_rows={len(result.transactions)}")


if __name__ == "__main__":
    main()
