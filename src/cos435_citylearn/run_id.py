from datetime import datetime


def build_run_id(
    algo: str,
    variant: str,
    split: str,
    seed: int,
    now: datetime | None = None,
) -> str:
    stamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    return f"{algo}__{variant}__{split}__seed{seed}__{stamp}"
