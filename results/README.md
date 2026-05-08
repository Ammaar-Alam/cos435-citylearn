# results layout

Actual run outputs stay out of git, but the directory structure is tracked so everyone writes results to the same place.

- `runs/` run-local artifacts and checkpoints
- `metrics/` flat metric exports
- `tables/` report tables
- `figures/` generated figures
- `manifests/` environment and experiment manifests
- `ui_exports/` simulation bundles, playback payloads, and render media for the local dashboard
- `dashboard/` backend job requests, logs, and result files

Downloaded shared-drive bundles should be treated as local intake only. Normalize
their contents into the directories above, then commit only clean summaries and
report assets under `submission/`.

## git policy

The tracked files in this directory should normally be only `.gitkeep` files and
documentation. The ignored local-output boundaries are:

- `results/runs/` checkpoints, rollout traces, and run-local metadata
- `results/metrics/` flat per-run metric rows consumed by the submission exporter
- `results/tables/` and `results/figures/` scratch report outputs
- `results/manifests/` regenerated environment/schema manifests
- `results/ui_exports/` CityLearn UI upload bundles, playback payloads, and media
- `results/dashboard/` local dashboard job state
- `results/sweep/` Princeton/Neuronic sweep JSONs, summary CSVs, and Slurm logs
- `results/sweep/summary.csv` is the raw shared PPO/SAC/TD3 sweep manifest consumed by the submission exporter; commit only normalized copies under `submission/results/`
- `results/*_tmp/` temporary staging directories

Commit final report artifacts only after they are normalized under
`submission/results/`, `submission/figures/`, or `submission/presentation.pptx`.
