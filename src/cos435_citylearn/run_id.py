from datetime import datetime


def _format_lr(lr: float) -> str:
    # keep path-safe (no dots), readable for humans scanning dir listings
    return f"{lr:.6g}".replace(".", "p").replace("+", "").replace("-", "m")


def build_run_id(
    algo: str,
    variant: str,
    split: str,
    seed: int,
    now: datetime | None = None,
    lr: float | None = None,
) -> str:
    stamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    lr_part = f"__lr{_format_lr(lr)}" if lr is not None else ""
    return f"{algo}__{variant}__{split}__seed{seed}{lr_part}__{stamp}"
