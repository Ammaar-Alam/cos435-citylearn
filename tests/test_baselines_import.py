from __future__ import annotations

import builtins
import importlib


def test_baselines_import_does_not_require_sac_module(monkeypatch) -> None:
    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "cos435_citylearn.baselines.sac":
            raise ImportError(f"blocked import: {name}")
        return real_import(name, globals, locals, fromlist, level)

    import cos435_citylearn.baselines as baselines_module

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    importlib.reload(baselines_module)

    try:
        assert callable(baselines_module.run_rbc)
    finally:
        monkeypatch.setattr(builtins, "__import__", real_import)
        importlib.reload(baselines_module)
