#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.ml.predictive_model import build_supervised_examples


def summarize(trace_path: str, horizons: list[int], train_ratio: float = 0.8) -> dict:
    rows = []
    p = Path(trace_path)
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                rows.append(obj)

    out = {"trace_path": trace_path, "rows": len(rows), "horizons": {}}
    for h in horizons:
        examples = build_supervised_examples(rows, horizon=h)
        returns = np.array([e.target_return for e in examples], dtype=float) if examples else np.array([])
        if returns.size == 0:
            out["horizons"][str(h)] = {"examples": 0}
            continue
        cut = max(min(int(len(returns) * train_ratio), len(returns) - 1), 1)
        train = returns[:cut]
        test = returns[cut:]
        out["horizons"][str(h)] = {
            "examples": int(len(returns)),
            "train_examples": int(len(train)),
            "test_examples": int(len(test)),
            "train_positive_gt0": float(np.mean(train > 0.0)),
            "test_positive_gt0": float(np.mean(test > 0.0)) if len(test) else 0.0,
            "train_mean": float(np.mean(train)),
            "test_mean": float(np.mean(test)) if len(test) else 0.0,
            "p01": float(np.quantile(returns, 0.01)),
            "p50": float(np.quantile(returns, 0.50)),
            "p99": float(np.quantile(returns, 0.99)),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect trace return-label quality by horizon")
    parser.add_argument("--trace", default="logs/recommendation_trace.jsonl")
    parser.add_argument("--horizons", default="5,10,20")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    args = parser.parse_args()

    horizons = [int(x.strip()) for x in args.horizons.split(",") if x.strip()]
    result = summarize(args.trace, horizons=horizons, train_ratio=args.train_ratio)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
