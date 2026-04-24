from datetime import datetime, timezone

import pytest

from cos435_citylearn.run_id import build_run_id


@pytest.fixture(autouse=True)
def _clear_slurm_env(monkeypatch):
    # Any SLURM env leaking in from the surrounding shell (pytest running inside
    # a SLURM allocation, or another test leaving state behind) would make the
    # uuid fallback stop firing and turn `same_params -> distinct run_id` tests
    # flaky. Scrub the full set in every test; individual cases re-set what
    # they actually care about via monkeypatch.setenv.
    for var in (
        "SLURM_ARRAY_JOB_ID",
        "SLURM_ARRAY_TASK_ID",
        "SLURM_JOB_ID",
        "SLURM_RESTART_COUNT",
    ):
        monkeypatch.delenv(var, raising=False)


def test_build_run_id_deterministic_with_frozen_now_and_job_id() -> None:
    run_id = build_run_id(
        algo="sac",
        variant="central_baseline",
        split="public_dev",
        seed=3,
        now=datetime(2026, 4, 11, 23, 10, 0, tzinfo=timezone.utc),
        job_id="deadbeef",
    )

    assert run_id == "sac__central_baseline__public_dev__seed3__20260411T231000Z__deadbeef"


def test_build_run_id_without_lr_omits_lr_segment() -> None:
    run_id = build_run_id(
        algo="sac",
        variant="central_baseline",
        split="public_dev",
        seed=3,
        now=datetime(2026, 4, 11, 23, 10, 0, tzinfo=timezone.utc),
        lr=None,
        job_id="deadbeef",
    )

    assert "__lr" not in run_id
    assert run_id == "sac__central_baseline__public_dev__seed3__20260411T231000Z__deadbeef"


def test_build_run_id_includes_lr_when_provided() -> None:
    run_id = build_run_id(
        algo="ppo",
        variant="shared_dtde_reward_v2",
        split="public_dev",
        seed=0,
        now=datetime(2026, 4, 11, 23, 10, 0, tzinfo=timezone.utc),
        lr=1e-4,
        job_id="deadbeef",
    )

    assert (
        run_id
        == "ppo__shared_dtde_reward_v2__public_dev__seed0__lr0p0001__20260411T231000Z__deadbeef"
    )


def test_build_run_id_distinguishes_sweep_lrs() -> None:
    kwargs = dict(
        algo="ppo",
        variant="shared_dtde_reward_v2",
        split="public_dev",
        seed=0,
        now=datetime(2026, 4, 11, 23, 10, 0, tzinfo=timezone.utc),
        job_id="deadbeef",
    )
    a = build_run_id(lr=1e-4, **kwargs)
    b = build_run_id(lr=3e-4, **kwargs)

    assert a != b


def test_build_run_id_same_params_same_second_do_not_collide() -> None:
    # Two dashboard double-submits / SLURM requeues with identical params in the same
    # second must produce distinct run_ids (the whole point of the uuid/job_id suffix).
    kwargs = dict(
        algo="ppo",
        variant="shared_dtde_reward_v2",
        split="public_dev",
        seed=0,
        now=datetime(2026, 4, 11, 23, 10, 0, tzinfo=timezone.utc),
        lr=1e-4,
    )
    a = build_run_id(**kwargs)
    b = build_run_id(**kwargs)

    assert a != b


def test_build_run_id_prefers_slurm_array_env(monkeypatch) -> None:
    monkeypatch.setenv("SLURM_ARRAY_JOB_ID", "123456")
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "7")
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)

    run_id = build_run_id(
        algo="ppo",
        variant="shared_dtde_reward_v2",
        split="public_dev",
        seed=0,
        now=datetime(2026, 4, 11, 23, 10, 0, tzinfo=timezone.utc),
        lr=1e-4,
    )

    assert run_id.endswith("__123456.7")


def test_build_run_id_falls_back_to_slurm_job_id(monkeypatch) -> None:
    monkeypatch.delenv("SLURM_ARRAY_JOB_ID", raising=False)
    monkeypatch.delenv("SLURM_ARRAY_TASK_ID", raising=False)
    monkeypatch.setenv("SLURM_JOB_ID", "987654")

    run_id = build_run_id(
        algo="ppo",
        variant="shared_dtde_reward_v2",
        split="public_dev",
        seed=0,
        now=datetime(2026, 4, 11, 23, 10, 0, tzinfo=timezone.utc),
        lr=1e-4,
    )

    assert run_id.endswith("__987654")


def test_build_run_id_naive_now_is_treated_as_utc() -> None:
    aware = build_run_id(
        algo="sac",
        variant="central_baseline",
        split="public_dev",
        seed=3,
        now=datetime(2026, 4, 11, 23, 10, 0, tzinfo=timezone.utc),
        job_id="deadbeef",
    )
    naive = build_run_id(
        algo="sac",
        variant="central_baseline",
        split="public_dev",
        seed=3,
        now=datetime(2026, 4, 11, 23, 10, 0),
        job_id="deadbeef",
    )

    assert aware == naive


def test_build_run_id_slurm_requeue_disambiguates_restart(monkeypatch) -> None:
    # --requeue keeps the same ARRAY_JOB_ID/ARRAY_TASK_ID across attempts;
    # without SLURM_RESTART_COUNT the retry collides with the failed attempt.
    monkeypatch.setenv("SLURM_ARRAY_JOB_ID", "123456")
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "7")

    kwargs = dict(
        algo="ppo",
        variant="shared_dtde_reward_v2",
        split="public_dev",
        seed=0,
        now=datetime(2026, 4, 11, 23, 10, 0, tzinfo=timezone.utc),
        lr=1e-4,
    )

    monkeypatch.delenv("SLURM_RESTART_COUNT", raising=False)
    first_attempt = build_run_id(**kwargs)

    monkeypatch.setenv("SLURM_RESTART_COUNT", "1")
    retry = build_run_id(**kwargs)

    assert first_attempt != retry
    assert first_attempt.endswith("__123456.7")
    assert retry.endswith("__123456.7.r1")


def test_build_run_id_slurm_restart_count_zero_is_not_appended(monkeypatch) -> None:
    # SLURM sets SLURM_RESTART_COUNT=0 on the first attempt; don't pollute the
    # happy-path run_id with a dangling .r0 suffix.
    monkeypatch.setenv("SLURM_JOB_ID", "987654")
    monkeypatch.setenv("SLURM_RESTART_COUNT", "0")

    run_id = build_run_id(
        algo="ppo",
        variant="shared_dtde_reward_v2",
        split="public_dev",
        seed=0,
        now=datetime(2026, 4, 11, 23, 10, 0, tzinfo=timezone.utc),
        lr=1e-4,
    )

    assert run_id.endswith("__987654")
    assert ".r0" not in run_id
