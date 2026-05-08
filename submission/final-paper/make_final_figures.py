"""Generate paper-facing figures from tracked submission result summaries."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "final-paper" / "figures"

COLORS = {
    "RBC": "#5B6770",
    "PPO": "#6F55B5",
    "SAC": "#2A9D8F",
    "TD3": "#D77A2D",
    "CHESCA": "#111827",
}

LABELS = {
    "public_dev": "Public dev",
    "released_phase2": "Released phase 2",
    "released_phase3": "Released phase 3",
}

RBC = {
    "public_dev": (1.022619, 0.0),
    "released_phase2": (1.087092, 0.051225),
    "released_phase3": (1.113710, 0.030161),
}


def read_rows(name: str) -> list[dict[str, str]]:
    with (RESULTS / name).open(newline="") as fh:
        return list(csv.DictReader(fh))


def as_float(value: str) -> float:
    return float(value) if value else float("nan")


def plot_final_matrix() -> None:
    rows = read_rows("final_sweep_best_by_algorithm.csv")
    context = read_rows("context_experiment_summary.csv")
    groups = ["public_dev", "released_phase2", "released_phase3"]
    algos = ["RBC", "PPO", "SAC", "TD3"]
    learned = {
        (row["eval_group"], row["algorithm"].upper()): row
        for row in rows
        if row["sweep_key"] == "final_hp" and row["eval_group"] in groups
    }

    chesca_phase2 = next(
        row for row in context
        if row["experiment"] == "chesca_upstream_reproduction"
        and row["eval_group"] == "released_phase2"
    )

    plt.rcParams.update({
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,
        "figure.dpi": 180,
    })

    fig, ax = plt.subplots(figsize=(7.2, 3.55))
    width = 0.18
    centers = list(range(len(groups)))
    offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]

    for offset, algo in zip(offsets, algos):
        means: list[float] = []
        ci: list[float] = []
        for group in groups:
            if algo == "RBC":
                mean, err = RBC[group]
            else:
                row = learned[(group, algo)]
                mean = as_float(row["average_score_mean"])
                err = as_float(row["average_score_ci95"])
            means.append(mean)
            ci.append(err)
        xs = [x + offset for x in centers]
        ax.bar(
            xs,
            means,
            width=width,
            color=COLORS[algo],
            edgecolor="#1F2933",
            linewidth=0.55,
            label=algo,
            zorder=3,
        )
        ax.errorbar(
            xs,
            means,
            yerr=ci,
            fmt="none",
            ecolor="#1F2933",
            elinewidth=0.9,
            capsize=2.5,
            capthick=0.9,
            zorder=4,
        )

    chesca_y = as_float(chesca_phase2["average_score_mean"])
    chesca_ci = as_float(chesca_phase2["average_score_ci95"])
    ax.errorbar(
        [centers[1]],
        [chesca_y],
        yerr=[chesca_ci],
        fmt="D",
        markersize=5.5,
        markerfacecolor="white",
        markeredgecolor=COLORS["CHESCA"],
        markeredgewidth=1.0,
        ecolor=COLORS["CHESCA"],
        capsize=3,
        label="CHESCA reproduction",
        zorder=5,
    )
    ax.set_xticks(centers)
    ax.set_xticklabels([LABELS[group] for group in groups])
    ax.set_ylabel("Average score (lower is better)")
    ax.set_ylim(0.45, 1.17)
    ax.grid(axis="y", color="#D8DDE3", linewidth=0.55, alpha=0.85, zorder=0)
    ax.legend(
        ncol=5,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        frameon=False,
        columnspacing=1.25,
        handlelength=1.35,
    )
    fig.tight_layout(pad=0.6)
    fig.savefig(FIGURES / "final_result_matrix.png", bbox_inches="tight")
    plt.close(fig)


def plot_lr_screen_phase3() -> None:
    rows = [
        row for row in read_rows("final_sweep_summary.csv")
        if row["sweep_key"] == "lr_screen" and row["eval_group"] == "released_phase3"
    ]
    algos = ["ppo", "sac", "td3"]

    fig, ax = plt.subplots(figsize=(6.8, 3.1))
    markers = {"ppo": "o", "sac": "s", "td3": "^"}
    for algo in algos:
        series = sorted(
            [row for row in rows if row["algorithm"] == algo],
            key=lambda row: as_float(row["lr"]),
        )
        xs = [as_float(row["lr"]) for row in series]
        ys = [as_float(row["average_score_mean"]) for row in series]
        ci = [as_float(row["average_score_ci95"]) for row in series]
        label = algo.upper()
        ax.errorbar(
            xs,
            ys,
            yerr=ci,
            marker=markers[algo],
            markersize=4.2,
            linewidth=1.7,
            capsize=2.4,
            color=COLORS[label],
            label=label,
            zorder=3,
        )
        best_index = min(range(len(ys)), key=ys.__getitem__)
        ax.scatter(
            [xs[best_index]],
            [ys[best_index]],
            s=48,
            facecolor="white",
            edgecolor=COLORS[label],
            linewidth=1.2,
            zorder=4,
        )

    ax.annotate(
        "TD3 best: 0.736",
        xy=(3e-4, 0.736141),
        xytext=(1.5e-4, 0.717),
        ha="right",
        fontsize=8,
        color=COLORS["TD3"],
        arrowprops={"arrowstyle": "->", "color": COLORS["TD3"], "lw": 0.8},
    )
    ax.set_xscale("log")
    ax.set_xlabel("Learning rate")
    ax.set_ylabel("Phase-3 average score")
    ax.set_ylim(0.70, 0.98)
    ax.grid(axis="y", color="#D8DDE3", linewidth=0.55, alpha=0.85)
    ax.grid(axis="x", color="#E7EAEE", linewidth=0.45, alpha=0.45)
    ax.legend(frameon=False, ncol=3, loc="upper right")
    fig.tight_layout(pad=0.65)
    fig.savefig(FIGURES / "lr_screen_phase3.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    plot_final_matrix()
    plot_lr_screen_phase3()


if __name__ == "__main__":
    main()
