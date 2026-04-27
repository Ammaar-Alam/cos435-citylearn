"""Generate cross-split evaluation figures, matching the style of make_figures.py."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
RELEASED_MAIN = REPO_ROOT / "submission" / "results" / "released_eval_main_results.csv"
RELEASED_SEEDS = REPO_ROOT / "submission" / "results" / "released_eval_seed_inventory.csv"
LOCAL_MAIN = REPO_ROOT / "submission" / "results" / "local_main_results.csv"
FIGURES_OUT = REPO_ROOT / "submission" / "figures"

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

CHESCA_SCORE = 0.562

COLORS = {
    "rbc": "#aaaaaa",
    "ppo_central_baseline": "#4C72B0",
    "ppo_shared_dtde": "#2196F3",
    "sac_central_baseline": "#DD8452",
    "sac_central_rv1": "#55A868",
    "sac_central_rv2": "#C44E52",
    "sac_shared_dtde": "#8172B2",
}

SHORT_LABELS = {
    "rbc": "RBC",
    "ppo_central_baseline": "PPO\nCentral",
    "ppo_shared_dtde": "PPO\nDTDE",
    "sac_central_baseline": "SAC\nCentral",
    "sac_central_rv1": "SAC\nrv1",
    "sac_central_rv2": "SAC\nrv2",
    "sac_shared_dtde": "SAC\nDTDE",
}

# local_main method_id → COLORS key
LOCAL_TO_COLORS = {
    "local_rbc": "rbc",
    "ppo_central_baseline_public_dev": "ppo_central_baseline",
    "ppo_shared_dtde_reward_v2_public_dev": "ppo_shared_dtde",
    "sac_central_baseline_public_dev": "sac_central_baseline",
    "sac_central_reward_v1_public_dev": "sac_central_rv1",
    "sac_central_reward_v2_public_dev": "sac_central_rv2",
    "sac_shared_dtde_reward_v2_public_dev": "sac_shared_dtde",
}

SPLIT_LABELS = {
    "phase_2_online_eval_1": "p2-eval1",
    "phase_2_online_eval_2": "p2-eval2",
    "phase_2_online_eval_3": "p2-eval3",
    "phase_3_1": "p3-1",
    "phase_3_2": "p3-2",
    "phase_3_3": "p3-3",
}

SPLIT_ORDER = [
    "phase_2_online_eval_1", "phase_2_online_eval_2", "phase_2_online_eval_3",
    "phase_3_1", "phase_3_2", "phase_3_3",
]


def _load_released_main() -> dict[tuple[str, str], dict]:
    """Returns {(method_id, eval_group): row}"""
    rows = {}
    with RELEASED_MAIN.open(newline="") as f:
        for row in csv.DictReader(f):
            rows[(row["method_id"], row["eval_group"])] = row
    return rows


def _load_released_seeds() -> dict[tuple[str, str], list[dict]]:
    """Returns {(method_id_ck, split): [row, ...]} using (algorithm, variant) key."""
    rows: dict[tuple[str, str], list[dict]] = {}
    # build variant→method_id map
    variant_map = {
        ("rbc", "basic_rbc"): "rbc",
        ("ppo", "ppo_central_baseline"): "ppo_central_baseline",
        ("ppo", "ppo_shared_dtde_reward_v2"): "ppo_shared_dtde",
        ("sac", "central_baseline"): "sac_central_baseline",
        ("sac", "central_reward_v1"): "sac_central_rv1",
        ("sac", "central_reward_v2"): "sac_central_rv2",
        ("sac", "shared_dtde_reward_v2"): "sac_shared_dtde",
    }
    with RELEASED_SEEDS.open(newline="") as f:
        for row in csv.DictReader(f):
            key = (row["algorithm"], row["variant"])
            mid = variant_map.get(key)
            if mid is None:
                continue
            split = row["split"]
            rows.setdefault((mid, split), []).append(row)
    return rows


def _load_local_main() -> dict[str, dict]:
    rows = {}
    with LOCAL_MAIN.open(newline="") as f:
        for row in csv.DictReader(f):
            rows[row["method_id"]] = row
    return rows


# ── Figure 1: cross-split method comparison (phase_2_online_eval) ────────────

def plot_cross_split_comparison(released: dict) -> None:
    method_order = [
        "rbc", "ppo_central_baseline",
        "sac_central_baseline", "sac_central_rv1", "sac_central_rv2", "sac_shared_dtde",
    ]
    available = [(m, released[(m, "phase_2_online_eval")]) for m in method_order
                 if (m, "phase_2_online_eval") in released]

    labels = [SHORT_LABELS[m] for m, _ in available]
    means  = [float(r["average_score_mean"]) for _, r in available]
    stds   = [float(r["average_score_std"]) for _, r in available]
    colors = [COLORS[m] for m, _ in available]
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
        ax.set_title("Method Comparison — Released Phase-2 Online Evaluation (3 splits)")
        ax.legend(framealpha=0.9)
        ax.set_ylim(0, max(means) * 1.15)
        fig.tight_layout()
        out = FIGURES_OUT / "cross_split_comparison.png"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
    print(f"  wrote {out}")


# ── Figure 2: generalization gap (public_dev vs phase_2_online_eval) ─────────

def plot_generalization_gap(released: dict, local: dict) -> None:
    method_order = [
        "rbc", "ppo_central_baseline",
        "sac_central_baseline", "sac_central_rv1", "sac_central_rv2", "sac_shared_dtde",
    ]
    local_key_map = {
        "rbc": "local_rbc",
        "ppo_central_baseline": "ppo_central_baseline_public_dev",
        "sac_central_baseline": "sac_central_baseline_public_dev",
        "sac_central_rv1": "sac_central_reward_v1_public_dev",
        "sac_central_rv2": "sac_central_reward_v2_public_dev",
        "sac_shared_dtde": "sac_shared_dtde_reward_v2_public_dev",
    }

    available = [m for m in method_order
                 if (m, "phase_2_online_eval") in released and local_key_map[m] in local]

    labels = [SHORT_LABELS[m] for m in available]
    dev_scores  = [float(local[local_key_map[m]]["average_score_mean"]) for m in available]
    eval_scores = [float(released[(m, "phase_2_online_eval")]["average_score_mean"]) for m in available]
    eval_stds   = [float(released[(m, "phase_2_online_eval")]["average_score_std"]) for m in available]
    colors = [COLORS[m] for m in available]

    x = np.arange(len(available))
    width = 0.35

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(10, 5))
        bars_dev = ax.bar(x - width / 2, dev_scores, width, color=colors, alpha=0.5,
                          label="public_dev (tuning split)", zorder=3)
        bars_eval = ax.bar(x + width / 2, eval_scores, width, yerr=eval_stds, capsize=4,
                           color=colors, error_kw={"elinewidth": 1.2, "ecolor": "#444"},
                           label="phase_2_online_eval (mean ± std, 3 splits)", zorder=3)

        ax.axhline(CHESCA_SCORE, color="#e53935", linewidth=1.5, linestyle="--",
                   label=f"CHESCA winner ({CHESCA_SCORE:.3f})", zorder=4)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylabel("Average score (lower is better)")
        ax.set_title("Generalization Gap — public_dev vs Released Phase-2 Online Evaluation")
        ax.legend(framealpha=0.9, fontsize=9)
        ax.set_ylim(0, max(max(dev_scores), max(eval_scores)) * 1.18)
        fig.tight_layout()
        out = FIGURES_OUT / "generalization_gap.png"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
    print(f"  wrote {out}")


# centralized methods cannot run on phase_3 (obs/action schema is 3-building specific)
PHASE3_PORTABLE = {"rbc", "sac_shared_dtde"}


# ── Figure 3: per-split score lines ──────────────────────────────────────────

def plot_per_split_lines(seeds: dict) -> None:
    method_order = [
        "rbc", "ppo_central_baseline",
        "sac_central_baseline", "sac_central_rv1", "sac_central_rv2", "sac_shared_dtde",
    ]
    present_splits = list(SPLIT_ORDER)
    x = np.arange(len(present_splits))
    x_labels = [SPLIT_LABELS[s] for s in present_splits]
    p2_count = sum(1 for s in present_splits if s.startswith("phase_2"))

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(10, 5))

        for method in method_order:
            ys, xs_used = [], []
            for i, split in enumerate(present_splits):
                rows = seeds.get((method, split), [])
                if rows:
                    ys.append(float(rows[0]["average_score"]))
                    xs_used.append(i)
            if not ys:
                continue

            portable = method in PHASE3_PORTABLE
            ax.plot(xs_used, ys, marker="o", linewidth=1.8, markersize=6,
                    color=COLORS[method], linestyle="-",
                    label=SHORT_LABELS[method].replace("\n", " "))

        # shade phase_3 region for centralized-only methods and add one label
        p3_x_start = p2_count - 0.5
        p3_x_end = len(present_splits) - 0.5
        ax.axvspan(p3_x_start, p3_x_end, color="#f0f0f0", zorder=0)
        ax.text((p3_x_start + p3_x_end) / 2, 0.62,
                "centralized models\nnot portable",
                ha="center", va="bottom", fontsize=8, color="#999",
                style="italic", linespacing=1.4)

        if 0 < p2_count < len(present_splits):
            ax.axvline(p3_x_start, color="#888", linewidth=1.0, linestyle=":")
            ax.text(p3_x_start - 0.05, ax.get_ylim()[1] * 0.985,
                    "phase_2 ←", ha="right", va="top", fontsize=8, color="#888")
            ax.text(p3_x_start + 0.05, ax.get_ylim()[1] * 0.985,
                    "→ phase_3", ha="left", va="top", fontsize=8, color="#888")

        ax.axhline(CHESCA_SCORE, color="#e53935", linewidth=1.3, linestyle="--",
                   label=f"CHESCA ({CHESCA_SCORE:.3f})", zorder=4)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=10)
        ax.set_ylabel("Average score (lower is better)")
        ax.set_title("Per-Split Scores — Released Evaluation Datasets")
        ax.legend(framealpha=0.9, fontsize=9, loc="upper left")
        fig.tight_layout()
        out = FIGURES_OUT / "per_split_scores.png"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
    print(f"  wrote {out}")


# ── Figure 4: KPI breakdown for phase_2_online_eval ──────────────────────────

def plot_cross_split_kpi(released: dict) -> None:
    kpi_fields = [
        ("district_cost_total_mean", "Cost"),
        ("district_carbon_emissions_total_mean", "Carbon"),
        ("district_daily_peak_average_mean", "Daily peak"),
        ("district_discomfort_proportion_mean", "Discomfort"),
        ("district_one_minus_thermal_resilience_proportion_mean", "Thermal\nresilience"),
    ]

    methods = [
        ("rbc", "RBC"),
        ("ppo_central_baseline", "PPO Central"),
        ("sac_central_rv1", "SAC rv1"),
        ("sac_central_rv2", "SAC rv2"),
        ("sac_shared_dtde", "SAC DTDE"),
    ]
    methods = [(m, lbl) for m, lbl in methods if (m, "phase_2_online_eval") in released]

    # normalise to RBC released-eval values (not public_dev RBC)
    rbc_row = released.get(("rbc", "phase_2_online_eval"), {})
    rbc_vals = {field: float(rbc_row[field]) for field, _ in kpi_fields if rbc_row.get(field)}

    n_kpis = len(kpi_fields)
    n_methods = len(methods)
    x = np.arange(n_kpis)
    width = 0.8 / n_methods
    offsets = np.linspace(-(n_methods - 1) / 2, (n_methods - 1) / 2, n_methods) * width

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(10, 5))

        for (method_id, label), offset in zip(methods, offsets):
            row = released[(method_id, "phase_2_online_eval")]
            values = []
            for field, _ in kpi_fields:
                raw = float(row[field]) if row.get(field) else 0.0
                rbc = rbc_vals.get(field, 1.0)
                values.append(raw / rbc if rbc else 0.0)
            ax.bar(x + offset, values, width=width * 0.9,
                   color=COLORS[method_id], label=label, zorder=3)

        ax.axhline(1.0, color="#888", linewidth=1.0, linestyle=":", zorder=2)
        ax.set_xticks(x)
        ax.set_xticklabels([lbl for _, lbl in kpi_fields], fontsize=10)
        ax.set_ylabel("Normalized KPI (lower is better, 1.0 = RBC-equivalent)")
        ax.set_title("KPI Breakdown — Released Phase-2 Online Evaluation (mean across 3 splits)")
        ax.legend(framealpha=0.9, fontsize=9)
        fig.tight_layout()
        out = FIGURES_OUT / "cross_split_kpi_breakdown.png"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
    print(f"  wrote {out}")


def main() -> None:
    FIGURES_OUT.mkdir(parents=True, exist_ok=True)

    released = _load_released_main()
    seeds    = _load_released_seeds()
    local    = _load_local_main()

    print("Cross-split comparison bar chart...")
    plot_cross_split_comparison(released)

    print("Generalization gap chart...")
    plot_generalization_gap(released, local)

    print("Per-split score lines...")
    plot_per_split_lines(seeds)

    print("Cross-split KPI breakdown...")
    plot_cross_split_kpi(released)

    print("Done.")


if __name__ == "__main__":
    main()
