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
