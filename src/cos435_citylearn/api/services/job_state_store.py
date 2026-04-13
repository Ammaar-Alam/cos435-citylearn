from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cos435_citylearn.io import write_json_atomic


class JobStateStore:
    def __init__(self, jobs_root: Path):
        self.jobs_root = jobs_root

    def _path(self, job_id: str) -> Path:
        return self.jobs_root / job_id / "state.json"

    def get(self, job_id: str) -> dict[str, Any] | None:
        path = self._path(job_id)
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def write(self, job_id: str, payload: dict[str, Any]) -> Path:
        return write_json_atomic(self._path(job_id), payload)
