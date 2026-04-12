from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cos435_citylearn.io import append_jsonl


class JobEventStore:
    def __init__(self, jobs_root: Path):
        self.jobs_root = jobs_root

    def _path(self, job_id: str) -> Path:
        return self.jobs_root / job_id / "events.jsonl"

    def append(self, job_id: str, payload: dict[str, Any]) -> Path:
        path = self._path(job_id)
        seq = 1
        if path.exists():
            with path.open(encoding="utf-8") as handle:
                seq = sum(1 for _ in handle) + 1
        return append_jsonl(path, {"seq": seq, **payload})

    def list_after(self, job_id: str, after_seq: int = 0) -> list[dict[str, Any]]:
        path = self._path(job_id)
        if not path.exists():
            return []

        events = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                event = json.loads(line)
                if int(event.get("seq", 0)) > after_seq:
                    events.append(event)
        return events
