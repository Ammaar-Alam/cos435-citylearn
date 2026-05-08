from cos435_citylearn.baselines.rbc import run_rbc


def run_sac(*args, **kwargs):
    from cos435_citylearn.baselines.sac import run_sac as _run_sac

    return _run_sac(*args, **kwargs)


def run_ppo(*args, **kwargs):
    from cos435_citylearn.baselines.ppo import run_ppo as _run_ppo

    return _run_ppo(*args, **kwargs)


def run_td3(*args, **kwargs):
    from cos435_citylearn.baselines.td3 import run_td3 as _run_td3

    return _run_td3(*args, **kwargs)


__all__ = ["run_rbc", "run_sac", "run_ppo", "run_td3"]
