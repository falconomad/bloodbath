#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _line_count(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def main() -> None:
    p = argparse.ArgumentParser(description="Write version manifest for a training dataset")
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--source", action="append", default=[])
    p.add_argument("--meta", action="append", default=[])
    args = p.parse_args()

    dataset = Path(args.dataset)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    sources = [Path(x) for x in (args.source or [])]
    source_items = []
    for src in sources:
        item = {"path": str(src), "exists": src.exists()}
        if src.exists():
            item["sha256"] = _sha256(src)
            item["size_bytes"] = src.stat().st_size
            if src.suffix.lower() in {".jsonl", ".csv"}:
                item["rows"] = _line_count(src)
        source_items.append(item)

    meta = {}
    for kv in (args.meta or []):
        if "=" not in kv:
            continue
        k, v = kv.split("=", 1)
        meta[k.strip()] = v.strip()

    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "path": str(dataset),
            "exists": dataset.exists(),
            "sha256": _sha256(dataset) if dataset.exists() else "",
            "size_bytes": dataset.stat().st_size if dataset.exists() else 0,
            "rows": _line_count(dataset) if dataset.exists() else 0,
        },
        "sources": source_items,
        "meta": meta,
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(str(out))


if __name__ == "__main__":
    main()
