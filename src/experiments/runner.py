from __future__ import annotations

import argparse
import copy
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.backtesting.simple_backtester import (
    ExecutionModel,
    build_examples,
    evaluate_candidate,
    load_trace_entries,
    sample_weight_candidates,
)
from src.pipeline.decision_engine import load_config


RESULTS_DIR = Path("artifacts/experiments")


def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    total = sum(max(float(v), 0.0) for v in weights.values())
    if total <= 0:
        n = len(weights) if weights else 1
        return {k: 1.0 / n for k in weights} if weights else {}
    return {k: max(float(v), 0.0) / total for k, v in weights.items()}


def _deep_update(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def build_walk_forward_splits(n: int, windows: int = 4, train_ratio: float = 0.7) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    if n <= 2:
        return []
    windows = max(int(windows), 1)
    block = max(n // windows, 2)

    splits: list[tuple[tuple[int, int], tuple[int, int]]] = []
    start = 0
    while start + block <= n:
        end = start + block
        train_end = start + max(int(block * train_ratio), 1)
        if train_end >= end:
            break
        splits.append(((start, train_end), (train_end, end)))
        start += block
    return splits


def run_experiment(
    trace_path: str,
    experiment_config: dict[str, Any],
    output_path: str | None = None,
) -> dict[str, Any]:
    base_cfg = load_config()
    entries = load_trace_entries(trace_path)
    horizon = int(experiment_config.get("horizon", 1))
    examples = build_examples(entries, horizon=horizon)
    if not examples:
        return {"status": "no_data", "examples": 0}
    oos_cfg = experiment_config.get("out_of_sample", {}) or {}
    oos_enabled = bool(oos_cfg.get("enabled", True))
    oos_ratio = float(oos_cfg.get("ratio", 0.2))
    oos_ratio = min(max(oos_ratio, 0.0), 0.5)
    oos_cut = int(len(examples) * (1.0 - oos_ratio)) if oos_enabled else len(examples)
    if oos_enabled and oos_cut > 1 and oos_cut < len(examples):
        dev_examples = examples[:oos_cut]
        oos_examples = examples[oos_cut:]
    else:
        dev_examples = examples
        oos_examples = []

    variants = list(experiment_config.get("variants", []))
    if not variants:
        variants = [{"name": "baseline", "weights": base_cfg.get("weights", {})}]
    auto_tune = experiment_config.get("auto_tune", {}) or {}
    if bool(auto_tune.get("enabled", False)):
        trials = int(auto_tune.get("trials", 24))
        seed = int(auto_tune.get("seed", 11))
        base = _normalize_weights(base_cfg.get("weights", {}))
        sampled = sample_weight_candidates(base, trials=trials, seed=seed)
        auto_variants = [{"name": f"auto_{idx:02d}", "weights": w} for idx, w in enumerate(sampled, start=1)]
        variants = variants + auto_variants

    exec_cfg = experiment_config.get("execution", {})
    execution_model = ExecutionModel(
        fee_bps=float(exec_cfg.get("fee_bps", 1.0)),
        spread_bps=float(exec_cfg.get("spread_bps", 5.0)),
        slippage_bps=float(exec_cfg.get("slippage_bps", 3.0)),
        fill_ratio=float(exec_cfg.get("fill_ratio", 1.0)),
    )

    split_cfg = experiment_config.get("walk_forward", {})
    use_walk_forward = bool(split_cfg.get("enabled", True))
    if use_walk_forward:
        splits = build_walk_forward_splits(
            n=len(dev_examples),
            windows=int(split_cfg.get("windows", 4)),
            train_ratio=float(split_cfg.get("train_ratio", 0.7)),
        )
        if not splits:
            splits = [((0, int(len(dev_examples) * 0.7)), (int(len(dev_examples) * 0.7), len(dev_examples)))]
    else:
        cut = max(int(len(dev_examples) * 0.7), 1)
        splits = [((0, cut), (cut, len(dev_examples)))]

    rows: list[dict[str, Any]] = []
    for variant in variants:
        name = str(variant.get("name", "variant")).strip() or "variant"
        variant_patch = {k: v for k, v in variant.items() if k != "name"}
        variant_cfg = _deep_update(base_cfg, variant_patch)
        variant_cfg["weights"] = _normalize_weights(variant_cfg.get("weights", {}))

        train_scores = []
        test_scores = []
        variant_split_rows: list[dict[str, Any]] = []
        for idx, ((tr0, tr1), (te0, te1)) in enumerate(splits):
            train_ex = dev_examples[tr0:tr1]
            test_ex = dev_examples[te0:te1]
            if not train_ex or not test_ex:
                continue

            train_metrics = evaluate_candidate(train_ex, variant_cfg, variant_cfg["weights"], execution_model=execution_model)
            test_metrics = evaluate_candidate(test_ex, variant_cfg, variant_cfg["weights"], execution_model=execution_model)
            train_scores.append(float(train_metrics.get("objective", 0.0)))
            test_scores.append(float(test_metrics.get("objective", 0.0)))
            split_row = {
                "variant": name,
                "split": idx,
                "train_objective": train_metrics.get("objective", 0.0),
                "test_objective": test_metrics.get("objective", 0.0),
                "test_total_return": test_metrics.get("total_return", 0.0),
                "test_win_rate": test_metrics.get("win_rate", 0.0),
                "test_profit_factor": test_metrics.get("profit_factor", 0.0),
                "test_trades": test_metrics.get("trades", 0.0),
                "test_sharpe_ratio": test_metrics.get("sharpe_ratio", 0.0),
                "test_sortino_ratio": test_metrics.get("sortino_ratio", 0.0),
                "test_max_drawdown": test_metrics.get("max_drawdown", 0.0),
                "test_cagr": test_metrics.get("cagr", 0.0),
                "test_calmar_ratio": test_metrics.get("calmar_ratio", 0.0),
                "test_expectancy_per_trade": test_metrics.get("expectancy_per_trade", 0.0),
            }
            variant_split_rows.append(split_row)
            rows.append(split_row)

        mean_train = (sum(train_scores) / len(train_scores)) if train_scores else -1e9
        mean_test = (sum(test_scores) / len(test_scores)) if test_scores else -1e9
        denom = max(len(variant_split_rows), 1)
        rows.append(
            {
                "variant": name,
                "split": "aggregate",
                "train_objective": mean_train,
                "test_objective": mean_test,
                "test_total_return": sum(r["test_total_return"] for r in variant_split_rows) / denom,
                "test_win_rate": sum(r["test_win_rate"] for r in variant_split_rows) / denom,
                "test_profit_factor": sum(r["test_profit_factor"] for r in variant_split_rows) / denom,
                "test_trades": sum(r["test_trades"] for r in variant_split_rows) / denom,
                "test_sharpe_ratio": sum(r["test_sharpe_ratio"] for r in variant_split_rows) / denom,
                "test_sortino_ratio": sum(r["test_sortino_ratio"] for r in variant_split_rows) / denom,
                "test_max_drawdown": sum(r["test_max_drawdown"] for r in variant_split_rows) / denom,
                "test_cagr": sum(r["test_cagr"] for r in variant_split_rows) / denom,
                "test_calmar_ratio": sum(r["test_calmar_ratio"] for r in variant_split_rows) / denom,
                "test_expectancy_per_trade": sum(r["test_expectancy_per_trade"] for r in variant_split_rows) / denom,
            }
        )

    aggregates = [r for r in rows if r["split"] == "aggregate"]
    aggregates.sort(key=lambda r: float(r.get("test_objective", -1e9)), reverse=True)
    best = aggregates[0] if aggregates else None
    best_oos = None
    if best is not None and oos_examples:
        best_name = str(best.get("variant", "")).strip()
        best_variant = next(
            (v for v in variants if str(v.get("name", "")).strip() == best_name),
            None,
        )
        if best_variant is not None:
            variant_patch = {k: v for k, v in best_variant.items() if k != "name"}
            variant_cfg = _deep_update(base_cfg, variant_patch)
            variant_cfg["weights"] = _normalize_weights(variant_cfg.get("weights", {}))
            best_oos = evaluate_candidate(oos_examples, variant_cfg, variant_cfg["weights"], execution_model=execution_model)

    result = {
        "status": "ok",
        "examples": len(examples),
        "dev_examples": len(dev_examples),
        "out_of_sample_examples": len(oos_examples),
        "variants": [v.get("name", "variant") for v in variants],
        "best_variant": best,
        "best_variant_out_of_sample": best_oos,
        "rows": rows,
    }

    out = output_path or str(RESULTS_DIR / f"experiment_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json")
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    csv_path = path.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "variant",
                "split",
                "train_objective",
                "test_objective",
                "test_total_return",
                "test_win_rate",
                "test_profit_factor",
                "test_trades",
                "test_sharpe_ratio",
                "test_sortino_ratio",
                "test_max_drawdown",
                "test_cagr",
                "test_calmar_ratio",
                "test_expectancy_per_trade",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    result["output_json"] = str(path)
    result["output_csv"] = str(csv_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run strategy experiments with walk-forward testing")
    parser.add_argument("--trace", default="logs/recommendation_trace.jsonl")
    parser.add_argument("--config", required=True, help="Experiment config json path")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    result = run_experiment(args.trace, cfg, output_path=(args.output or None))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
