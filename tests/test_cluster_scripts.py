from __future__ import annotations

from cos435_citylearn.paths import REPO_ROOT


def test_mappo_sweep_exports_artifact_root_for_direct_sbatch() -> None:
    script = (REPO_ROOT / "scripts" / "cluster" / "mappo_sweep.slurm").read_text()

    assert 'SWEEP_ROOT="${SWEEP_ROOT:-$ROOT_DIR/results/mappo_sweep}"' in script
    export_idx = script.index("export ROOT_DIR SWEEP_ROOT")
    call_idx = script.index('bash "$ROOT_DIR/scripts/cluster/run_final_cell.sh"')
    assert export_idx < call_idx


def test_mappo_cluster_entrypoints_prefer_root_source_tree() -> None:
    export_line = 'export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"'
    for rel_path in [
        "scripts/cluster/run_final_cell.sh",
        "scripts/cluster/mappo_smoke.slurm",
    ]:
        script = (REPO_ROOT / rel_path).read_text()
        assert export_line in script
        assert script.index(export_line) < script.index("python ")
