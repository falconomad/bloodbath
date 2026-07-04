Bloodbath Trading Bot

This repository is a single-purpose headless trading bot that:
- uses Alpaca market data and Alpaca paper trading
- scores candidates with local technical and sentiment logic
- optionally asks Gemini for a final decision on a small shortlist
- validates every trade through a risk gate
- submits paper-trading orders through Alpaca
- writes run telemetry so each run can be reviewed afterward

Current Workflow

The main entrypoint is `python main.py`.

The live automation path is GitHub Actions:
- workflow file: `.github/workflows/trading.yml`
- workflow name in GitHub: `Trading Bot Execution`
- trigger: every 10 minutes on weekdays, plus manual runs

Required GitHub Secrets

- `ALPACA_API_KEY`
- `ALPACA_API_SECRET`
- `GEMINI_API_KEY`

How The Bot Works

1. Connects to Alpaca paper trading and loads account state, clock, and open positions.
2. Forces exits first when an open position crosses the profit-take or stop-loss threshold.
3. Pulls top movers and recent daily bars from Alpaca.
4. Pulls symbol news from Alpaca, with `yfinance` fallback if needed.
5. Pulls earnings context from `yfinance` and macro headlines from Google News RSS.
6. Builds local technical and sentiment scores.
7. Ranks candidates with liquidity, regime, and confidence filters.
8. Sends only the top shortlisted names to Gemini when the cadence gate allows it.
9. Runs the risk manager, which can reject trades or cap buy allocations.
10. Sends buy and sell orders to Alpaca paper trading.
11. Writes telemetry under `logs/` so the run can be inspected later.

Core Files

- `main.py`: orchestrates the full run
- `config.py`: runtime settings and environment variables
- `data_ingestion/`: Alpaca, news, and event fetchers
- `ai_engines/`: technical, sentiment, ranking, and Gemini decision logic
- `execution/`: risk checks, trade execution, and telemetry output
- `.github/workflows/trading.yml`: scheduled GitHub Actions runner

Generated Outputs

These are runtime artifacts and should not be committed:
- `logs/engine_events.jsonl`
- `logs/profit_summary.json`
- `logs/gemini_quota_state.json`

GitHub Actions Notes

- Keep `.github/workflows/trading.yml`.
- The workflow now uploads the `logs/` directory as a GitHub Actions artifact so each run can be reviewed.
- If you see other workflows in GitHub but not in this repo, they are likely from another branch or an older state and should be compared against this file before keeping them.
