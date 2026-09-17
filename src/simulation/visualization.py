from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.simulation.logger import ExperimentTracker


def _load_condition_runs(
    tracker: ExperimentTracker,
    buffer_type: str,
    memory_regime: str,
    temporal_mode: str,
) -> list[list[dict[str, float]]]:
    index = tracker.load_index()
    matching_runs = [
        row
        for row in index
        if row.get("buffer_type") == buffer_type
        and row.get("memory_regime") == memory_regime
        and row.get("temporal_mode") == temporal_mode
    ]

    all_metrics: list[list[dict[str, float]]] = []
    for row in matching_runs:
        run_id = row.get("run_id")
        if not run_id:
            continue
        metrics = tracker.load_run_metrics(run_id)
        if metrics:
            all_metrics.append(metrics)
    return all_metrics


def _aggregate_curves(
    runs_metrics: list[list[dict[str, float]]],
    cummax: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not runs_metrics:
        return np.array([]), np.array([]), np.array([])

    min_len = min(len(run) for run in runs_metrics)
    generations = np.array([runs_metrics[0][i]["generation"] for i in range(min_len)])

    matrix = np.zeros((len(runs_metrics), min_len), dtype=np.float64)
    for r_idx, run in enumerate(runs_metrics):
        for g_idx in range(min_len):
            matrix[r_idx, g_idx] = run[g_idx]["top_mean_fitness"]
        if cummax:
            matrix[r_idx] = np.maximum.accumulate(matrix[r_idx])

    mean_curve = np.mean(matrix, axis=0)
    std_curve = np.std(matrix, axis=0)
    return generations, mean_curve, std_curve


def _resolve_condition_activations(
    tracker: ExperimentTracker,
    temporal_mode: str,
    default_a: int,
) -> int:
    index = tracker.load_index()
    for row in index:
        if row.get("temporal_mode") == temporal_mode and row.get("n_activations"):
            try:
                return int(float(row["n_activations"]))
            except (ValueError, TypeError):
                pass
    return default_a


def plot_paper_memory_regime(
    tracker: ExperimentTracker,
    memory_regime: str,
    title: str,
    output_file: Path,
    cummax: bool = True,
) -> Path:
    buffer_types = ["average", "peek_end", "decay"]
    buffer_titles = ["(a) Average", "(b) Peek-End Rule", "(c) Decay"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)

    # Greyscale/B&W-accessible palette:
    # High-contrast luminance: dark tone (#08519c) vs mid-light tone (#d95f02)
    color_hetero = "#08519c"
    color_homo = "#d95f02"

    hetero_a = _resolve_condition_activations(tracker, "heterogeneous", 3)
    homo_a = _resolve_condition_activations(tracker, "homogeneous", 1)

    y_label = (
        "Top 10 Cumulative Max Fitness (Coins)"
        if cummax
        else "Top 10 Mean Fitness (Coins)"
    )

    for idx, (b_type, b_title) in enumerate(zip(buffer_types, buffer_titles)):
        ax = axes[idx]
        ax.set_title(b_title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Generations", fontsize=11)
        if idx == 0:
            ax.set_ylabel(y_label, fontsize=11)

        ax.grid(True, linestyle=":", alpha=0.5, color="#bbbbbb")

        hetero_runs = _load_condition_runs(
            tracker, b_type, memory_regime, "heterogeneous"
        )
        gens_h, mean_h, std_h = _aggregate_curves(hetero_runs, cummax=cummax)
        if len(gens_h) > 0:
            ax.plot(
                gens_h,
                mean_h,
                label=f"Heterogeneity (a={hetero_a}, solid)",
                color=color_hetero,
                linestyle="-",
                linewidth=2.2,
            )
            if len(hetero_runs) > 1:
                ax.fill_between(
                    gens_h,
                    mean_h - std_h,
                    mean_h + std_h,
                    color=color_hetero,
                    alpha=0.15,
                )

        homo_runs = _load_condition_runs(tracker, b_type, memory_regime, "homogeneous")
        gens_o, mean_o, std_o = _aggregate_curves(homo_runs, cummax=cummax)
        if len(gens_o) > 0:
            ax.plot(
                gens_o,
                mean_o,
                label=f"Homogeneity (a={homo_a}, dashed)",
                color=color_homo,
                linestyle="--",
                dashes=(6, 3),
                linewidth=2.2,
            )
            if len(homo_runs) > 1:
                ax.fill_between(
                    gens_o,
                    mean_o - std_o,
                    mean_o + std_o,
                    color=color_homo,
                    alpha=0.15,
                    hatch="//",
                )

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(
                handles,
                labels,
                loc="lower right" if cummax else "upper left",
                frameon=True,
                fontsize=9,
                framealpha=0.9,
            )

    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_file


def plot_paper_figure_6(
    tracker: ExperimentTracker,
    output_dir: Path,
    cummax: bool = True,
    output_file: Path | None = None,
) -> Path:
    if output_file is None:
        filename = (
            "figure_6_short_term_memory_cummax.png"
            if cummax
            else "figure_6_short_term_memory_raw.png"
        )
        output_file = output_dir / filename

    title = (
        "Figure 6: Short-Term Memory Model (Cumulative Maximum)"
        if cummax
        else "Figure 6: Short-Term Memory Model (Raw Generational Fitness)"
    )
    return plot_paper_memory_regime(
        tracker=tracker,
        memory_regime="short_term",
        title=title,
        output_file=output_file,
        cummax=cummax,
    )


def plot_paper_figure_7(
    tracker: ExperimentTracker,
    output_dir: Path,
    cummax: bool = True,
    output_file: Path | None = None,
) -> Path:
    if output_file is None:
        filename = (
            "figure_7_long_term_memory_cummax.png"
            if cummax
            else "figure_7_long_term_memory_raw.png"
        )
        output_file = output_dir / filename

    title = (
        "Figure 7: Long-Term Memory Model (Cumulative Maximum)"
        if cummax
        else "Figure 7: Long-Term Memory Model (Raw Generational Fitness)"
    )
    return plot_paper_memory_regime(
        tracker=tracker,
        memory_regime="long_term",
        title=title,
        output_file=output_file,
        cummax=cummax,
    )


def plot_single_run(
    metrics: list[dict[str, float]], title: str, output_path: Path
) -> Path:
    if not metrics:
        return output_path

    generations = [m["generation"] for m in metrics]
    top_mean = [m["top_mean_fitness"] for m in metrics]
    max_fit = [m["max_fitness"] for m in metrics]
    min_fit = [m["min_fitness"] for m in metrics]
    top_cummax = np.maximum.accumulate(top_mean)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        generations,
        top_mean,
        label="Top 10 Mean Fitness (Raw, solid)",
        color="#08519c",
        linestyle="-",
        linewidth=1.8,
        alpha=0.75,
    )
    ax.plot(
        generations,
        top_cummax,
        label="Top 10 Best-so-far (CumMax, dash-dot)",
        color="#000000",
        linestyle="-.",
        linewidth=2.0,
    )
    ax.plot(
        generations,
        max_fit,
        label="Max Fitness (Dashed)",
        color="#2ca02c",
        linestyle="--",
        linewidth=1.5,
    )
    ax.plot(
        generations,
        min_fit,
        label="Min Fitness (Dotted)",
        color="#d62728",
        linestyle=":",
        linewidth=1.5,
    )

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Generation", fontsize=11)
    ax.set_ylabel("Fitness (Coins)", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.5, color="#bbbbbb")
    ax.legend(loc="lower right", frameon=True, framealpha=0.9)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_paper_figures(
    results_dir: Path | str = "results",
    output_dir: Path | str | None = None,
    generate_both: bool = True,
) -> list[Path]:
    tracker = ExperimentTracker(base_dir=results_dir)
    out = Path(output_dir) if output_dir else tracker.figures_dir
    out.mkdir(parents=True, exist_ok=True)

    # Clean up legacy un-suffixed duplicate files if present
    for legacy_name in [
        "figure_6_short_term_memory.png",
        "figure_7_long_term_memory.png",
    ]:
        legacy_file = out / legacy_name
        if legacy_file.exists():
            legacy_file.unlink()

    # 1. Cumulative maximum (reproducing paper figures 6 and 7 step plateaus)
    fig6_cummax = plot_paper_figure_6(tracker, out, cummax=True)
    fig7_cummax = plot_paper_figure_7(tracker, out, cummax=True)

    generated = [fig6_cummax, fig7_cummax]

    # 2. Raw generational fitness (showing underlying population exploration & noise)
    if generate_both:
        fig6_raw = plot_paper_figure_6(tracker, out, cummax=False)
        fig7_raw = plot_paper_figure_7(tracker, out, cummax=False)
        generated.extend([fig6_raw, fig7_raw])

    return generated
