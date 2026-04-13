from cos435_citylearn.baselines.rbc import run_rbc


def run_sac(*args, **kwargs):
    from cos435_citylearn.baselines.sac import run_sac as _run_sac

    return _run_sac(*args, **kwargs)

__all__ = ["run_rbc", "run_sac"]
