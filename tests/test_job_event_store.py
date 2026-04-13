from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from cos435_citylearn.api.services.job_event_store import JobEventStore


def test_job_event_store_assigns_unique_monotonic_sequences(tmp_path: Path) -> None:
    store = JobEventStore(tmp_path)
    job_id = "job_concurrent"

    def append_event(index: int) -> None:
        store.append(
            job_id,
            {
                "job_id": job_id,
                "event_type": "progress",
                "created_at": f"2026-04-12T00:00:{index:02d}Z",
                "payload": {"index": index},
            },
        )

    with ThreadPoolExecutor(max_workers=12) as executor:
        list(executor.map(append_event, range(48)))

    events = store.list_after(job_id)
    seqs = [event["seq"] for event in events]

    assert len(seqs) == 48
    assert seqs == list(range(1, 49))
