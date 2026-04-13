from __future__ import annotations

import json
import threading
from pathlib import Path

import cos435_citylearn.io as io_module


def test_write_json_atomic_uses_unique_temp_files_for_concurrent_writes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    target = tmp_path / "state.json"
    payloads = [{"value": 1}, {"value": 2}]
    barrier = threading.Barrier(2)
    errors: list[Exception] = []
    original_replace = io_module.os.replace

    def gated_replace(src, dst):
        barrier.wait(timeout=2)
        return original_replace(src, dst)

    monkeypatch.setattr(io_module.os, "replace", gated_replace)

    def writer(payload: dict[str, int]) -> None:
        try:
            io_module.write_json_atomic(target, payload)
        except Exception as exc:  # pragma: no cover - exercised on failure
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(payload,), daemon=True) for payload in payloads]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)

    assert not errors
    assert json.loads(target.read_text()) in payloads
