from __future__ import annotations

from cos435_citylearn.paths import REPO_ROOT


def test_mappo_sweep_exports_artifact_root_for_direct_sbatch() -> None:
    script = (REPO_ROOT / "scripts" / "cluster" / "mappo_sweep.slurm").read_text()

    assert 'SWEEP_ROOT="${SWEEP_ROOT:-$ROOT_DIR/results/mappo_sweep}"' in script
    export_idx = script.index("export ROOT_DIR SWEEP_ROOT")
    call_idx = script.index('bash "$ROOT_DIR/scripts/cluster/run_final_cell.sh"')
    assert export_idx < call_idx
