# Architecture

## Overview
This project is organized as a goal-driven trading agent with clear stage boundaries:

1. Data ingestion
2. Validation and normalization
3. Signal generation
4. Goal-aware scoring and decisioning
5. Portfolio execution
6. Trace logging and learning memory

## Core Runtime Flow
- `src/advisor.py`
  - Orchestrates cycle execution (`run_top20_cycle_with_signals`) as compatibility facade
  - Fetches candidate universes and market/news inputs
  - Delegates recommendation logic to `src/pipeline/recommendation_service.py`
  - Applies goal policy overlays from `src/agent/goal_policy.py`
  - Passes final analyses to `src/core/top20_manager.py`

- `src/agent/goal_policy.py`
  - Tracks objective pace (`start_capital`, `target_capital`, deadline horizon)
  - Computes progress, required daily return, and pacing multiplier

- `src/agent/decision_engine.py`
  - Applies goal-based threshold/position-size adjustments to the base scoring config
  - Keeps `config/weights.yaml` as the stable base policy

- `src/pipeline/recommendation_service.py`
  - Builds normalized module signals (trend, sentiment, events, micro, dip, volatility)
  - Applies centralized data validation
  - Calls decision engine for score/confidence/decision
  - Emits structured trace entries

- `src/pipeline/decision_engine.py`
  - Config-driven scoring and thresholds from `config/weights.yaml`
  - Conflict handling, hard guardrails, veto layer, stability/cooldown logic
  - Volatility-based confidence and position-size adjustment

## Data and Validation
- `src/api/data_fetcher.py`
  - Market/news/earnings/micro-feature fetchers
  - Supports structured news payloads (`structured=True`)

- `src/validation/data_validation.py`
  - Reusable validation checks (price history, micro payload, headlines, earnings payload)
  - All modules should use this layer for sanity checks instead of ad-hoc validation

## Execution and Portfolio Controls
- `src/core/top20_manager.py`
  - Executes BUY/SELL/HOLD with risk-aware rules
  - Handles stops, take-profit, cooldowns, confirmation streaks
  - Enforces sector-level constraints:
    - max sector allocation
    - max new sector exposure per step
    - max positions per sector

- `src/core/sp500_list.py`
  - Universe symbols and top-level sector mapping (`TOP20_SECTOR`)

## Persistence
- `src/db.py`
  - DB initialization and worker dedupe key
  - Tables for portfolio, transactions, snapshots, signals
  - Goal and learning tables:
    - `agent_goal_snapshots`
    - `trade_decisions`
  - Decision memory persistence tables:
    - `decision_memory`
    - `decision_engine_meta`

- `worker/auto_worker.py`
  - Restores portfolio + decision memory state
  - Runs cycle and persists updated state

## Analytics, Backtesting, Experiments
- `src/backtesting/simple_backtester.py`
  - Trace-based backtesting and weight tuning
  - Realism model: fees, spread, slippage, partial fills

- `src/analytics/performance_reports.py`
  - Signal accuracy, win rate, profit factor, drawdown, expectancy, module contributions

- `src/analytics/explainability_report.py`
  - Human-readable decision/veto/guardrail summaries from trace logs

- `src/experiments/runner.py`
  - Multi-variant strategy experiments
  - Walk-forward train/test splits
  - Saves JSON and CSV results under `artifacts/experiments/`

## Shared Utilities
- `src/common/trace_utils.py`
  - Shared JSONL loading and safe float parsing for analytics/backtesting tooling

## Configuration
- `config/weights.yaml`
  - Single source for scoring weights, thresholds, quality checks, risk knobs, stability, veto settings
  - Loaded at process import/start (not hot-reloaded in a running process)

## Testing Strategy
- `tests/`
  - Focused deterministic unit tests for:
    - decision engine behavior
    - validation layer
    - sentiment processing
    - backtesting realism
    - analytics/explainability
    - experiment runner

## Extension Guidelines
- New data quality rules: add to `src/validation/data_validation.py` first, then consume in pipeline.
- New signal modules: return normalized signal semantics (`value`, `confidence`, `quality_ok`, `reason`).
- New decision constraints: add to `src/pipeline/decision_engine.py` and expose config knobs in `config/weights.yaml`.
- New analysis/reporting: consume `logs/recommendation_trace.jsonl` using `src/common/trace_utils.py`.
- Avoid adding API calls into scoring logic; keep fetchers in `src/api/` and policy in `src/pipeline/`.
