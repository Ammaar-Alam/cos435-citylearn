# analysis scripts

Use the Make targets from the repo root as the canonical report refresh path:

```bash
make submission-results
make figures
make cross-figures
make cross-table
```

## canonical scripts

- `export_submission_results.py` reads normalized run metrics and sweep summaries
  from ignored `results/` directories and writes the tracked CSV snapshot under
  `submission/results/`. Its shared sweep path is generic for PPO, SAC, and
  TD3; the older `ppo_shared_sweep_*` outputs are preserved for report sections
  that still depend on the PPO-specific tables.
- `make_figures.py` writes the main report figures under `submission/figures/`.
- `make_cross_split_figures.py` writes cross-split comparison figures under
  `submission/figures/`.
- `make_cross_split_table.py` refreshes the compact cross-split table.
- `build_deck.py` builds `submission/presentation.pptx` from tracked result
  tables and figures.

## specialized script

`aggregate_ppo_dtde_phase2.py` is a specialized backfill and consistency script
for the PPO shared-DTDE phase-2 released-eval rows. It parses historical
`results/eval_phase2_ppo_dtde_*` logs and aligns those rows with the same
best-public-dev learning-rate selection used for phase-3 reporting. Treat it as
a reproducibility/repair tool for that one PPO data source, not as the primary
submission exporter.
