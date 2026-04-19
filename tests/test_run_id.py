from datetime import datetime

from cos435_citylearn.run_id import build_run_id


def test_build_run_id_matches_repo_contract() -> None:
    run_id = build_run_id(
        algo="sac",
        variant="central_baseline",
        split="public_dev",
        seed=3,
        now=datetime(2026, 4, 11, 23, 10, 0),
    )

    assert run_id == "sac__central_baseline__public_dev__seed3__20260411_231000"


def test_build_run_id_without_lr_keeps_legacy_format() -> None:
    # explicit lr=None must match pre-sweep format so old SAC checkpoints resolve
    run_id = build_run_id(
        algo="sac",
        variant="central_baseline",
        split="public_dev",
        seed=3,
        now=datetime(2026, 4, 11, 23, 10, 0),
        lr=None,
    )

    assert "__lr" not in run_id
    assert run_id == "sac__central_baseline__public_dev__seed3__20260411_231000"


def test_build_run_id_includes_lr_when_provided() -> None:
    run_id = build_run_id(
        algo="ppo",
        variant="shared_dtde_reward_v2",
        split="public_dev",
        seed=0,
        now=datetime(2026, 4, 11, 23, 10, 0),
        lr=1e-4,
    )

    assert run_id == "ppo__shared_dtde_reward_v2__public_dev__seed0__lr0p0001__20260411_231000"


def test_build_run_id_distinguishes_sweep_lrs() -> None:
    kwargs = dict(
        algo="ppo",
        variant="shared_dtde_reward_v2",
        split="public_dev",
        seed=0,
        now=datetime(2026, 4, 11, 23, 10, 0),
    )
    a = build_run_id(lr=1e-4, **kwargs)
    b = build_run_id(lr=3e-4, **kwargs)

    assert a != b
