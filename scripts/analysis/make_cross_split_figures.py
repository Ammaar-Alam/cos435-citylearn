"""Generate released-evaluation figures from tracked submission CSVs."""

from __future__ import annotations

from make_figures import (
    _load_submission_rows,
    plot_cross_split_comparison,
    plot_generalization_gap,
    plot_per_split_scores,
)


def main() -> None:
    rows = _load_submission_rows()

    print("Per-split scores (held-out)...")
    plot_per_split_scores()

    print("Cross-split comparison (released phase_2 bars)...")
    plot_cross_split_comparison()

    print("Generalization gap (public_dev vs phase_2)...")
    plot_generalization_gap(rows)

    print("Done.")


if __name__ == "__main__":
    main()
