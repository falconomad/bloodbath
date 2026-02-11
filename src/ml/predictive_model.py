from __future__ import annotations

import argparse
import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.common.trace_utils import load_jsonl_dict_rows, safe_float

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

try:
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
    from sklearn.metrics import mean_squared_error, precision_score, recall_score, roc_auc_score
except Exception:  # pragma: no cover
    RandomForestClassifier = None
    RandomForestRegressor = None
    GradientBoostingClassifier = None
    GradientBoostingRegressor = None
    mean_squared_error = None
    precision_score = None
    recall_score = None
    roc_auc_score = None


@dataclass
class Example:
    ts: str
    ticker: str
    features: dict[str, float]
    target_return: float


def _sorted_trace_by_ticker(entries: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_ticker: dict[str, list[dict[str, Any]]] = {}
    for row in entries:
        ticker = str(row.get("ticker", "")).upper().strip()
        ts = str(row.get("ts", "")).strip()
        price = safe_float(row.get("price", float("nan")), default=float("nan"))
        if not ticker or not ts or not math.isfinite(price) or price <= 0:
            continue
        by_ticker.setdefault(ticker, []).append(row)

    for ticker in by_ticker:
        by_ticker[ticker].sort(key=lambda r: str(r.get("ts", "")))
    return by_ticker


def _build_feature_row(row: dict[str, Any]) -> dict[str, float]:
    signals = dict(row.get("signals", {}) or {})
    risk = dict(row.get("risk_context", {}) or {})

    feature_map: dict[str, float] = {}
    # Signal block: sentiment is one feature among many.
    for name in ["trend", "sentiment", "social", "market", "events", "micro", "dip", "volatility"]:
        payload = dict(signals.get(name, {}) or {})
        feature_map[f"sig_{name}_value"] = safe_float(payload.get("value", 0.0))
        feature_map[f"sig_{name}_confidence"] = safe_float(payload.get("confidence", 0.0))
        feature_map[f"sig_{name}_quality"] = 1.0 if bool(payload.get("quality_ok", False)) else 0.0

    feature_map["score"] = safe_float(row.get("score", 0.0))
    feature_map["conflict_ratio"] = safe_float(row.get("conflict_ratio", 0.0))
    feature_map["decision_confidence"] = safe_float(row.get("confidence", 0.0))

    for key in [
        "rel_volume",
        "atr_pct",
        "data_gap_ratio",
        "portfolio_drawdown",
        "portfolio_avg_correlation",
        "ticker_sector_allocation",
        "sentiment_variance",
        "sentiment_article_count",
        "social_post_count",
        "market_news_count",
    ]:
        feature_map[f"risk_{key}"] = safe_float(risk.get(key, 0.0))

    return feature_map


def build_feature_row_for_inference(
    signals: dict[str, dict[str, Any]],
    risk_context: dict[str, Any] | None = None,
    score: float = 0.0,
    conflict_ratio: float = 0.0,
    confidence: float = 0.0,
) -> dict[str, float]:
    payload = {
        "signals": signals or {},
        "risk_context": risk_context or {},
        "score": float(score),
        "conflict_ratio": float(conflict_ratio),
        "confidence": float(confidence),
    }
    return _build_feature_row(payload)


def build_supervised_examples(entries: list[dict[str, Any]], horizon: int = 5) -> list[Example]:
    horizon = max(int(horizon), 1)
    by_ticker = _sorted_trace_by_ticker(entries)
    out: list[Example] = []

    for ticker, rows in by_ticker.items():
        for idx, row in enumerate(rows):
            j = idx + horizon
            if j >= len(rows):
                continue
            p0 = safe_float(row.get("price", 0.0))
            p1 = safe_float(rows[j].get("price", 0.0))
            if p0 <= 0 or p1 <= 0:
                continue
            target_return = (p1 - p0) / p0
            out.append(
                Example(
                    ts=str(row.get("ts", "")),
                    ticker=ticker,
                    features=_build_feature_row(row),
                    target_return=float(target_return),
                )
            )

    out.sort(key=lambda e: e.ts)
    return out


def chronological_split(examples: list[Example], train_ratio: float = 0.8) -> tuple[list[Example], list[Example]]:
    if not examples:
        return [], []
    train_ratio = min(max(float(train_ratio), 0.5), 0.95)
    cut = max(int(len(examples) * train_ratio), 1)
    cut = min(cut, len(examples) - 1)
    return examples[:cut], examples[cut:]


def _matrix(examples: list[Example]) -> tuple[list[str], Any, Any]:
    if np is None:
        raise RuntimeError("numpy is required for ML training")
    if not examples:
        return [], np.zeros((0, 0)), np.zeros((0,))

    feature_names = sorted(examples[0].features.keys())
    x = np.array([[safe_float(e.features.get(f, 0.0)) for f in feature_names] for e in examples], dtype=float)
    y = np.array([float(e.target_return) for e in examples], dtype=float)
    return feature_names, x, y


def _profit_metrics(y_true: Any, y_prob: Any, threshold: float = 0.55, trade_cost_bps: float = 8.0) -> dict[str, float]:
    if np is None:
        return {"trades": 0.0, "strategy_total_return": 0.0, "buy_hold_total_return": 0.0, "profit_factor": 0.0}

    probs = np.asarray(y_prob, dtype=float)
    rets = np.asarray(y_true, dtype=float)
    signal = (probs >= float(threshold)).astype(float)
    costs = (float(trade_cost_bps) / 10_000.0) * signal
    pnl = (signal * rets) - costs

    gross_profit = float(np.sum(pnl[pnl > 0])) if pnl.size else 0.0
    gross_loss = float(abs(np.sum(pnl[pnl < 0]))) if pnl.size else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 1e-12 else (999.0 if gross_profit > 0 else 0.0)

    return {
        "trades": float(np.sum(signal)),
        "strategy_total_return": float(np.sum(pnl)),
        "buy_hold_total_return": float(np.sum(rets)),
        "profit_factor": float(profit_factor),
        "avg_trade_expectancy": float(np.mean(pnl[signal > 0])) if np.any(signal > 0) else 0.0,
    }


def train_and_evaluate(
    trace_path: str,
    horizon: int = 5,
    train_ratio: float = 0.8,
    model_family: str = "random_forest",
    random_state: int = 11,
) -> dict[str, Any]:
    if RandomForestClassifier is None or np is None:
        return {"status": "missing_deps", "detail": "Install scikit-learn and numpy"}

    entries = load_jsonl_dict_rows(trace_path)
    examples = build_supervised_examples(entries, horizon=horizon)
    train, test = chronological_split(examples, train_ratio=train_ratio)
    if len(train) < 20 or len(test) < 10:
        return {
            "status": "insufficient_data",
            "examples": len(examples),
            "train_examples": len(train),
            "test_examples": len(test),
        }

    feature_names, x_train, y_train = _matrix(train)
    _, x_test, y_test = _matrix(test)

    y_train_cls = (y_train > 0).astype(int)
    y_test_cls = (y_test > 0).astype(int)

    family = str(model_family).strip().lower()
    if family == "gradient_boosting":
        clf = GradientBoostingClassifier(random_state=random_state)
        reg = GradientBoostingRegressor(random_state=random_state)
    else:
        clf = RandomForestClassifier(n_estimators=250, max_depth=8, min_samples_leaf=8, random_state=random_state, n_jobs=-1)
        reg = RandomForestRegressor(n_estimators=250, max_depth=8, min_samples_leaf=8, random_state=random_state, n_jobs=-1)

    clf.fit(x_train, y_train_cls)
    reg.fit(x_train, y_train)

    y_pred_cls = clf.predict(x_test)
    y_prob = clf.predict_proba(x_test)[:, 1]
    y_pred_ret = reg.predict(x_test)

    precision = float(precision_score(y_test_cls, y_pred_cls, zero_division=0))
    recall = float(recall_score(y_test_cls, y_pred_cls, zero_division=0))
    roc_auc = float(roc_auc_score(y_test_cls, y_prob)) if len(set(y_test_cls.tolist())) > 1 else 0.5
    rmse = math.sqrt(float(mean_squared_error(y_test, y_pred_ret)))

    importances = list(getattr(clf, "feature_importances_", []))
    ranked = sorted(zip(feature_names, importances), key=lambda t: t[1], reverse=True)
    top_features = [{"feature": n, "importance": float(v)} for n, v in ranked[:20]]

    profit = _profit_metrics(y_true=y_test, y_prob=y_prob)

    return {
        "status": "ok",
        "model_family": family,
        "horizon": int(horizon),
        "examples": len(examples),
        "train_examples": len(train),
        "test_examples": len(test),
        "classification": {
            "precision": precision,
            "recall": recall,
            "roc_auc": roc_auc,
        },
        "regression": {
            "rmse": rmse,
        },
        "profit": profit,
        "top_features": top_features,
    }


def train_evaluate_and_save(
    trace_path: str,
    output_path: str,
    horizon: int = 5,
    train_ratio: float = 0.8,
    model_family: str = "random_forest",
    random_state: int = 11,
) -> dict[str, Any]:
    if RandomForestClassifier is None or np is None:
        return {"status": "missing_deps", "detail": "Install scikit-learn and numpy"}

    entries = load_jsonl_dict_rows(trace_path)
    examples = build_supervised_examples(entries, horizon=horizon)
    train, test = chronological_split(examples, train_ratio=train_ratio)
    if len(train) < 20 or len(test) < 10:
        return {
            "status": "insufficient_data",
            "examples": len(examples),
            "train_examples": len(train),
            "test_examples": len(test),
        }

    feature_names, x_train, y_train = _matrix(train)
    _, x_test, y_test = _matrix(test)
    y_train_cls = (y_train > 0).astype(int)
    y_test_cls = (y_test > 0).astype(int)

    family = str(model_family).strip().lower()
    if family == "gradient_boosting":
        clf = GradientBoostingClassifier(random_state=random_state)
        reg = GradientBoostingRegressor(random_state=random_state)
    else:
        clf = RandomForestClassifier(n_estimators=250, max_depth=8, min_samples_leaf=8, random_state=random_state, n_jobs=-1)
        reg = RandomForestRegressor(n_estimators=250, max_depth=8, min_samples_leaf=8, random_state=random_state, n_jobs=-1)

    clf.fit(x_train, y_train_cls)
    reg.fit(x_train, y_train)

    y_pred_cls = clf.predict(x_test)
    y_prob = clf.predict_proba(x_test)[:, 1]
    y_pred_ret = reg.predict(x_test)
    precision = float(precision_score(y_test_cls, y_pred_cls, zero_division=0))
    recall = float(recall_score(y_test_cls, y_pred_cls, zero_division=0))
    roc_auc = float(roc_auc_score(y_test_cls, y_prob)) if len(set(y_test_cls.tolist())) > 1 else 0.5
    rmse = math.sqrt(float(mean_squared_error(y_test, y_pred_ret)))
    profit = _profit_metrics(y_true=y_test, y_prob=y_prob)

    artifact = {
        "version": 1,
        "model_family": family,
        "horizon": int(horizon),
        "feature_names": feature_names,
        "classifier": clf,
        "regressor": reg,
        "metrics": {
            "classification": {"precision": precision, "recall": recall, "roc_auc": roc_auc},
            "regression": {"rmse": rmse},
            "profit": profit,
        },
    }
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        pickle.dump(artifact, f)

    return {
        "status": "ok",
        "artifact_path": str(out),
        "model_family": family,
        "horizon": int(horizon),
        "examples": len(examples),
        "train_examples": len(train),
        "test_examples": len(test),
        "classification": artifact["metrics"]["classification"],
        "regression": artifact["metrics"]["regression"],
        "profit": artifact["metrics"]["profit"],
    }


def load_model_artifact(path: str) -> dict[str, Any] | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        with p.open("rb") as f:
            payload = pickle.load(f)
        if not isinstance(payload, dict):
            return None
        return payload
    except Exception:
        return None


def predict_from_artifact(artifact: dict[str, Any], feature_row: dict[str, float]) -> dict[str, float] | None:
    if np is None or not artifact:
        return None
    clf = artifact.get("classifier")
    reg = artifact.get("regressor")
    feature_names = list(artifact.get("feature_names", []) or [])
    if clf is None or reg is None or not feature_names:
        return None
    x = np.array([[safe_float(feature_row.get(name, 0.0)) for name in feature_names]], dtype=float)
    prob_up = float(clf.predict_proba(x)[0][1])
    expected_return = float(reg.predict(x)[0])
    return {"prob_up": prob_up, "expected_return": expected_return}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train predictive return models from trace history")
    parser.add_argument("--trace", default="logs/recommendation_trace.jsonl")
    parser.add_argument("--horizon", type=int, default=5, help="Forward horizon in steps (e.g. 5 or 10)")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--model", default="random_forest", choices=["random_forest", "gradient_boosting"])
    parser.add_argument("--output", default="")
    parser.add_argument("--save-artifact", default="", help="Optional .pkl path to save trained models")
    args = parser.parse_args()

    if args.save_artifact:
        result = train_evaluate_and_save(
            trace_path=args.trace,
            output_path=args.save_artifact,
            horizon=args.horizon,
            train_ratio=args.train_ratio,
            model_family=args.model,
        )
    else:
        result = train_and_evaluate(
            trace_path=args.trace,
            horizon=args.horizon,
            train_ratio=args.train_ratio,
            model_family=args.model,
        )
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
