import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def ensure_parent(path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def write_json(path: str | Path, payload: Any) -> Path:
    target = ensure_parent(path)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return target


def write_json_atomic(path: str | Path, payload: Any) -> Path:
    target = ensure_parent(path)
    temp_path = target.with_suffix(f"{target.suffix}.tmp")
    temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temp_path, target)
    return target


def append_jsonl(path: str | Path, payload: Any) -> Path:
    target = ensure_parent(path)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
    return target


def write_csv_row(path: str | Path, row: dict[str, Any]) -> Path:
    target = ensure_parent(path)
    with target.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    return target


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
