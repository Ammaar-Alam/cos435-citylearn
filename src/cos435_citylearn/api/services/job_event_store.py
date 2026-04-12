from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
from typing import Any

from cos435_citylearn.io import ensure_parent


class JobEventStore:
    def __init__(self, jobs_root: Path):
        self.jobs_root = jobs_root

    def _path(self, job_id: str) -> Path:
        return self.jobs_root / job_id / "events.jsonl"

    def append(self, job_id: str, payload: dict[str, Any]) -> Path:
        path = ensure_parent(self._path(job_id))
        with path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                handle.seek(0)
                seq = sum(1 for _ in handle if _.strip()) + 1
                handle.seek(0, os.SEEK_END)
                handle.write(json.dumps({"seq": seq, **payload}, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return path

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
