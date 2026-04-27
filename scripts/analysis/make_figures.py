"""Generate results figures and tables."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO_ROOT / "results"
FIGURES_OUT = REPO_ROOT / "submission" / "figures"
TABLES_OUT = REPO_ROOT / "submission" / "tables"
SUBMISSION_RESULTS = REPO_ROOT / "submission" / "results" / "local_main_results.csv"
CROSS_SPLIT_CSV = REPO_ROOT / "submission" / "results" / "cross_split_scores.csv"
RELEASED_MAIN_CSV = REPO_ROOT / "submission" / "results" / "released_eval_main_results.csv"

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
    runs = sorted(RESULTS_ROOT.glob("*/runs/ppo__ppo_central_baseline__public_dev__*"))
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


PER_SPLIT_METHOD_ORDER = [
    ("RBC",          "#aaaaaa"),
    ("PPO Central",  "#4C72B0"),
    ("PPO DTDE",     "#2196F3"),
    ("SAC Central",  "#DD8452"),
    ("SAC rv1",      "#55A868"),
    ("SAC rv2",      "#C44E52"),
    ("SAC DTDE",     "#8172B2"),
]

# Columns in cross_split_scores.csv that drive the per-split figure.
PHASE2_SPLITS = ["p2_eval1", "p2_eval2", "p2_eval3"]
PHASE3_SPLITS = ["p3_1", "p3_2", "p3_3"]
PER_SPLIT_X_LABELS = ["p2-eval1", "p2-eval2", "p2-eval3", "p3-1", "p3-2", "p3-3"]


def _load_cross_split_rows() -> dict[str, dict[str, float | None]]:
    """Return {method_label: {split_col: score_or_None}} from cross_split_scores.csv."""
    if not CROSS_SPLIT_CSV.exists():
        return {}
    out: dict[str, dict[str, float | None]] = {}
    with CROSS_SPLIT_CSV.open(newline="") as f:
        for row in csv.DictReader(f):
            name = row["method"].strip()
            out[name] = {
                k: (float(v) if v not in (None, "", "nan") else None)
                for k, v in row.items()
                if k != "method"
            }
    return out


def plot_per_split_scores() -> None:
    """Held-out per-split comparison: phase_2 (3-building) on the left,
    phase_3 (6-building) on the right with shaded background and a
    'centralized models not portable' annotation. CHESCA's 0.562 public
    leaderboard score is overlaid as a dashed reference line.
    Reads cross_split_scores.csv and writes to submission/figures/.
    """
    rows = _load_cross_split_rows()
    if not rows:
        print("  skip per_split_scores: cross_split_scores.csv not found")
        return

    split_cols = PHASE2_SPLITS + PHASE3_SPLITS
    x = np.arange(len(split_cols))

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(11.5, 5.6))

        # Phase-3 shaded region
        phase2_end = len(PHASE2_SPLITS) - 0.5
        ax.axvspan(phase2_end, x[-1] + 0.5, color="#000000", alpha=0.04, zorder=0)
        ax.axvline(phase2_end, color="#999999", linestyle=":", linewidth=1.2, zorder=1)
        ax.text(
            phase2_end - 0.05, 1.075, "phase_2 ←",
            ha="right", va="bottom", fontsize=9, color="#666666",
            transform=ax.get_xaxis_transform(),
        )
        ax.text(
            phase2_end + 0.05, 1.075, "→ phase_3",
            ha="left", va="bottom", fontsize=9, color="#666666",
            transform=ax.get_xaxis_transform(),
        )

        for method, color in PER_SPLIT_METHOD_ORDER:
            if method not in rows:
                continue
            ys = [rows[method].get(c) for c in split_cols]
            xs_drawn = [xi for xi, y in zip(x, ys) if y is not None]
            ys_drawn = [y for y in ys if y is not None]
            if not ys_drawn:
                continue
            ax.plot(
                xs_drawn, ys_drawn, marker="o", color=color,
                linewidth=2.0, markersize=6, label=method, zorder=4,
            )

        ax.axhline(
            CHESCA_SCORE, color="#e53935", linewidth=1.6, linestyle="--",
            label=f"CHESCA 2023 public ref ({CHESCA_SCORE:.3f}, diff. eval server)",
            zorder=3,
        )

        ax.text(
            (phase2_end + x[-1] + 0.5) / 2, 0.82,
            "centralized models\nnot portable",
            ha="center", va="center", fontsize=10, color="#888888",
            style="italic", transform=ax.get_xaxis_transform(), zorder=2,
        )

        ax.set_xticks(x)
        ax.set_xticklabels(PER_SPLIT_X_LABELS, fontsize=10)
        ax.set_ylabel("Average score (lower is better)")
        ax.set_title("Per-Split Scores — Released Evaluation Datasets")
        ax.set_xlim(-0.5, x[-1] + 0.5)

        ax.legend(loc="upper left", framealpha=0.95, fontsize=9, ncol=1)
        fig.tight_layout()

        FIGURES_OUT.mkdir(parents=True, exist_ok=True)
        out = FIGURES_OUT / "per_split_scores.png"
        fig.savefig(out, bbox_inches="tight", dpi=200)
        print(f"  wrote {out}")
        plt.close(fig)


# Canonical method display order shared between cross_split_comparison and
# generalization_gap. (method_label in released_eval_main_results.csv) -> (short label, color, local_main_results method_id)
CROSS_SPLIT_METHODS = [
    ("RBC baseline",                "RBC",         COLORS["local_rbc"],                            "local_rbc"),
    ("PPO centralized baseline",    "PPO\nCentral",COLORS["ppo_central_baseline_public_dev"],      "ppo_central_baseline_public_dev"),
    ("PPO shared DTDE reward_v2",   "PPO\nDTDE",   COLORS["ppo_shared_dtde_reward_v2_public_dev"], "ppo_shared_dtde_reward_v2_public_dev"),
    ("Centralized SAC baseline",    "SAC\nCentral",COLORS["sac_central_baseline_public_dev"],      "sac_central_baseline_public_dev"),
    ("Centralized SAC reward_v1",   "SAC\nrv1",    COLORS["sac_central_reward_v1_public_dev"],     "sac_central_reward_v1_public_dev"),
    ("Centralized SAC reward_v2",   "SAC\nrv2",    COLORS["sac_central_reward_v2_public_dev"],     "sac_central_reward_v2_public_dev"),
    ("Shared DTDE SAC reward_v2",   "SAC\nDTDE",   COLORS["sac_shared_dtde_reward_v2_public_dev"], "sac_shared_dtde_reward_v2_public_dev"),
]


def _load_released_phase2_rows() -> dict[str, dict[str, str]]:
    """Return {method_label: row_dict} from released_eval_main_results.csv,
    filtered to eval_group == released_phase_2_online_eval."""
    out: dict[str, dict[str, str]] = {}
    if not RELEASED_MAIN_CSV.exists():
        return out
    with RELEASED_MAIN_CSV.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("eval_group") == "released_phase_2_online_eval":
                out[row["method_label"].strip()] = row
    return out


def plot_cross_split_comparison() -> None:
    """Bar chart of released phase_2 online-eval means with 95% CI error bars,
    one bar per method. Adds a dashed CHESCA reference line as benchmark context.
    Writes to submission/figures/ (dpi=200).
    """
    released = _load_released_phase2_rows()
    if not released:
        print("  skip cross_split_comparison: released_eval_main_results.csv not found")
        return

    methods = [m for m in CROSS_SPLIT_METHODS if m[0] in released]
    labels = [m[1] for m in methods]
    colors = [m[2] for m in methods]
    means = [float(released[m[0]]["average_score_mean"]) for m in methods]
    ci95s = [float(released[m[0]]["average_score_ci95"]) for m in methods]

    x = np.arange(len(methods))
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(
            x, means, yerr=ci95s, capsize=4, color=colors,
            error_kw={"elinewidth": 1.2, "ecolor": "#444"}, width=0.65, zorder=3,
        )
        ax.axhline(
            CHESCA_SCORE, color="#e53935", linewidth=1.5, linestyle="--",
            label="CHESCA public reference (not same eval)", zorder=4,
        )
        for bar, mean in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2, mean + 0.015,
                f"{mean:.3f}", ha="center", va="bottom", fontsize=9,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylabel("Average score (lower is better)")
        ax.set_title("Released Phase-2 Online Evaluation — 3 splits")
        ax.set_ylim(0, max(means) * 1.18)
        ax.legend(loc="upper right", framealpha=0.95, fontsize=9)
        fig.text(
            0.01, -0.01,
            "Published CHESCA is benchmark context; compare local/released runs within their own split group.",
            ha="left", fontsize=8, color="#666666",
        )
        fig.tight_layout()

        FIGURES_OUT.mkdir(parents=True, exist_ok=True)
        out = FIGURES_OUT / "cross_split_comparison.png"
        fig.savefig(out, bbox_inches="tight", dpi=200)
        print(f"  wrote {out}")
        plt.close(fig)


def plot_generalization_gap(rows: dict[str, dict]) -> None:
    """Paired bars showing local public_dev (light) vs released phase_2
    (solid, with std error bars) for each method. Exposes the train→held-out
    generalization gap in a single panel.
    """
    released = _load_released_phase2_rows()
    if not released:
        print("  skip generalization_gap: released_eval_main_results.csv not found")
        return

    methods = [
        m for m in CROSS_SPLIT_METHODS
        if m[0] in released and m[3] in rows
    ]
    labels = [m[1] for m in methods]
    colors = [m[2] for m in methods]
    public_means = [float(rows[m[3]]["average_score_mean"]) for m in methods]
    released_means = [float(released[m[0]]["average_score_mean"]) for m in methods]
    released_stds = [float(released[m[0]]["average_score_std"]) for m in methods]

    x = np.arange(len(methods))
    width = 0.38

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(11, 5))
        # Lighter shade for public_dev bar (alpha=0.45).
        ax.bar(
            x - width / 2, public_means, width=width, color=colors,
            alpha=0.45, edgecolor="none", zorder=3,
            label="public_dev (tuning split)",
        )
        ax.bar(
            x + width / 2, released_means, width=width, color=colors,
            yerr=released_stds, capsize=3, edgecolor="none",
            error_kw={"elinewidth": 1.0, "ecolor": "#333"}, zorder=3,
            label="phase_2_online_eval (mean ± std, 3 splits)",
        )
        ax.axhline(
            CHESCA_SCORE, color="#e53935", linewidth=1.5, linestyle="--",
            label=f"CHESCA 2023 public ref ({CHESCA_SCORE:.3f}, diff. eval server)",
            zorder=4,
        )

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylabel("Average score (lower is better)")
        ax.set_title("Generalization Gap — public_dev vs Released Phase-2 Online Evaluation")
        ymax = max(max(public_means), max(released_means)) * 1.15
        ax.set_ylim(0, ymax)
        ax.legend(loc="upper right", framealpha=0.95, fontsize=9)
        fig.tight_layout()

        FIGURES_OUT.mkdir(parents=True, exist_ok=True)
        out = FIGURES_OUT / "generalization_gap.png"
        fig.savefig(out, bbox_inches="tight", dpi=200)
        print(f"  wrote {out}")
        plt.close(fig)


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

    print("Per-split scores (held-out)...")
    plot_per_split_scores()

    print("Cross-split comparison (released phase_2 bars)...")
    plot_cross_split_comparison()

    print("Generalization gap (public_dev vs phase_2)...")
    plot_generalization_gap(rows)

    print("Comparison table...")
    write_comparison_table(rows)

    print("Done.")


if __name__ == "__main__":
    main()
