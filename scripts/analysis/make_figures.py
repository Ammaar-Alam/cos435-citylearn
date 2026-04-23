"""Generate results figures and tables."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = REPO_ROOT / "results" / "grace" / "runs"
FIGURES_OUT = REPO_ROOT / "results" / "grace" / "figures"
TABLES_OUT = REPO_ROOT / "results" / "grace" / "tables"
SUBMISSION_RESULTS = REPO_ROOT / "submission" / "results" / "local_main_results.csv"

STYLE = {
    "figure.dpi": 150,
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
}

METHOD_ORDER = [
    "local_rbc",
    "ppo_central_baseline_public_dev",
    "ppo_shared_dtde_reward_v2_public_dev",
    "sac_central_baseline_public_dev",
    "sac_central_reward_v1_public_dev",
    "sac_central_reward_v2_public_dev",
    "sac_shared_dtde_reward_v2_public_dev",
]

SHORT_LABELS = {
    "local_rbc": "RBC",
    "ppo_central_baseline_public_dev": "PPO\nCentral",
    "ppo_shared_dtde_reward_v2_public_dev": "PPO\nDTDE",
    "sac_central_baseline_public_dev": "SAC\nCentral",
    "sac_central_reward_v1_public_dev": "SAC\nrv1",
    "sac_central_reward_v2_public_dev": "SAC\nrv2",
    "sac_shared_dtde_reward_v2_public_dev": "SAC\nDTDE",
}

COLORS = {
    "local_rbc": "#aaaaaa",
    "ppo_central_baseline_public_dev": "#4C72B0",
    "ppo_shared_dtde_reward_v2_public_dev": "#2196F3",
    "sac_central_baseline_public_dev": "#DD8452",
    "sac_central_reward_v1_public_dev": "#55A868",
    "sac_central_reward_v2_public_dev": "#C44E52",
    "sac_shared_dtde_reward_v2_public_dev": "#8172B2",
}

CHESCA_SCORE = 0.562  # official public leaderboard winner


def _load_submission_rows() -> dict[str, dict]:
    rows = {}
    with SUBMISSION_RESULTS.open(newline="") as f:
        for row in csv.DictReader(f):
            rows[row["method_id"]] = row
    return rows


def _load_training_curve(run_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    path = run_dir / "training_curve.csv"
    steps, rewards = [], []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            steps.append(int(row["step"]))
            rewards.append(float(row["mean_reward"]))
    return np.array(steps), np.array(rewards)


def _find_ppo_run() -> Path | None:
    runs = sorted(RUNS_ROOT.glob("ppo__ppo_central_baseline__public_dev__*"))
    return runs[-1] if runs else None


def plot_training_curve(run_dir: Path) -> None:
    steps, rewards = _load_training_curve(run_dir)

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 4.5))

        ax.plot(steps / 1_000, rewards / 1_000, color=COLORS["ppo_central_baseline_public_dev"],
                linewidth=1.8, label="PPO centralized baseline")

        ax.set_xlabel("Training steps (thousands)")
        ax.set_ylabel("Mean episode reward (thousands)")
        ax.set_title("PPO Centralized Baseline — Training Curve")
        ax.legend(framealpha=0.9)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.0f}k"))

        fig.tight_layout()
        out = FIGURES_OUT / "ppo_training_curve.png"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
    print(f"  wrote {out}")


def plot_method_comparison(rows: dict[str, dict]) -> None:
    available = [m for m in METHOD_ORDER if m in rows]
    labels = [SHORT_LABELS[m] for m in available]
    means = [float(rows[m]["average_score_mean"]) for m in available]
    stds = [
        float(rows[m]["average_score_std"]) if rows[m]["average_score_std"] else 0.0
        for m in available
    ]
    colors = [COLORS[m] for m in available]

    x = np.arange(len(available))

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 5))

        bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors,
                      error_kw={"elinewidth": 1.2, "ecolor": "#444"}, width=0.6, zorder=3)

        ax.axhline(CHESCA_SCORE, color="#e53935", linewidth=1.5, linestyle="--",
                   label=f"CHESCA winner ({CHESCA_SCORE:.3f})", zorder=4)

        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, mean + 0.01,
                    f"{mean:.3f}", ha="center", va="bottom", fontsize=9)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylabel("Average score (lower is better)")
        ax.set_title("Method Comparison — Local Phase-2 Evaluation")
        ax.legend(framealpha=0.9)
        ax.set_ylim(0, max(means) * 1.15)

        fig.tight_layout()
        out = FIGURES_OUT / "method_comparison.png"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
    print(f"  wrote {out}")


def plot_kpi_breakdown(rows: dict[str, dict]) -> None:
    kpi_fields = [
        ("district_cost_total_mean", "Cost"),
        ("district_carbon_emissions_total_mean", "Carbon"),
        ("district_daily_peak_average_mean", "Daily peak"),
        ("district_discomfort_proportion_mean", "Discomfort"),
        ("district_one_minus_thermal_resilience_proportion_mean", "Thermal\nresilience"),
    ]

    methods = [
        ("local_rbc", "RBC", COLORS["local_rbc"]),
        ("ppo_central_baseline_public_dev", "PPO Central", COLORS["ppo_central_baseline_public_dev"]),
        ("ppo_shared_dtde_reward_v2_public_dev", "PPO DTDE", COLORS["ppo_shared_dtde_reward_v2_public_dev"]),
        ("sac_central_reward_v1_public_dev", "SAC rv1 (best)", COLORS["sac_central_reward_v1_public_dev"]),
    ]
    methods = [(m, lbl, c) for m, lbl, c in methods if m in rows]

    n_kpis = len(kpi_fields)
    n_methods = len(methods)
    x = np.arange(n_kpis)
    width = 0.8 / n_methods
    offsets = np.linspace(-(n_methods - 1) / 2, (n_methods - 1) / 2, n_methods) * width

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(10, 5))

        for (method_id, label, color), offset in zip(methods, offsets):
            values = [float(rows[method_id][field]) for field, _ in kpi_fields]
            ax.bar(x + offset, values, width=width * 0.9, color=color, label=label, zorder=3)

        ax.axhline(1.0, color="#888", linewidth=1.0, linestyle=":", zorder=2)
        ax.set_xticks(x)
        ax.set_xticklabels([lbl for _, lbl in kpi_fields], fontsize=10)
        ax.set_ylabel("Normalized KPI (lower is better, 1.0 = RBC-equivalent)")
        ax.set_title("KPI Breakdown by Method")
        ax.legend(framealpha=0.9, fontsize=9)

        fig.tight_layout()
        out = FIGURES_OUT / "kpi_breakdown.png"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
    print(f"  wrote {out}")


def write_comparison_table(rows: dict[str, dict]) -> None:
    available = [m for m in METHOD_ORDER if m in rows]
    fieldnames = [
        "method_label", "algorithm", "seed_count", "evidence_level",
        "average_score_mean", "average_score_std", "average_score_ci95",
        "pct_improvement_vs_local_rbc_mean",
    ]
    out_rows = []
    for m in available:
        r = rows[m]
        out_rows.append({f: r.get(f, "") for f in fieldnames})

    out = TABLES_OUT / "method_comparison.csv"
    TABLES_OUT.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"  wrote {out}")


def main() -> None:
    FIGURES_OUT.mkdir(parents=True, exist_ok=True)
    TABLES_OUT.mkdir(parents=True, exist_ok=True)

    rows = _load_submission_rows()

    run_dir = _find_ppo_run()
    if run_dir:
        print("Training curve...")
        plot_training_curve(run_dir)
    else:
        print("Warning: no PPO run found, skipping training curve")

    print("Method comparison bar chart...")
    plot_method_comparison(rows)

    print("KPI breakdown chart...")
    plot_kpi_breakdown(rows)

    print("Comparison table...")
    write_comparison_table(rows)

    print("Done.")


if __name__ == "__main__":
    main()
