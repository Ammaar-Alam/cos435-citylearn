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
