import argparse
import sys
from pathlib import Path

from src.config.schema import AppConfig
from src.simulation.runner import SimulationRunner
from src.simulation.visualization import plot_paper_figures


def main() -> None:
    parser = argparse.ArgumentParser(description="Evolutionary Simulation Runner")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML configuration file (default: configs/config.yaml)",
    )
    parser.add_argument(
        "--plot",
        "--plot-only",
        dest="plot_only",
        action="store_true",
        help=(
            "Generate paper figures from existing logs without running new simulations"
        ),
    )

    args = parser.parse_args()

    try:
        config_path = Path(args.config)
        config = AppConfig.from_yaml(config_path)
        results_dir = Path(config.simulation.results_dir)

        if args.plot_only:
            print(f"Generating paper figures from results in '{results_dir}'...")
            figures = plot_paper_figures(results_dir=results_dir)
            for fig in figures:
                print(f"Saved figure: {fig}")
            return

        print(
            "Starting simulations across all buffer types, memory regimes, "
            "and temporal activation modes configured in YAML..."
        )
        runner = SimulationRunner(
            config=config,
            results_dir=results_dir,
        )
        runner.run_all(verbose=True, generate_plots=True)
        print(f"Completed all simulations. Results saved to '{results_dir}'.")

    except Exception as e:
        print(f"Error executing simulation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
