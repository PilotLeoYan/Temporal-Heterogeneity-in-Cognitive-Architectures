import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.config.schema import AppConfig
from src.simulation.evolution import EvolutionarySimulation
from src.simulation.logger import ExperimentTracker
from src.simulation.reproducibility import generate_seeds, get_environment_metadata
from src.simulation.visualization import plot_paper_figures, plot_single_run


def _run_single_worker(
    config: AppConfig,
    results_dir: Path,
    buffer_type: str,
    memory_regime: str,
    temporal_mode: str,
    n_activations: int,
    seed: int,
    verbose: bool = True,
) -> dict[str, Any]:
    """Isolated worker function suitable for multiprocessing/ProcessPoolExecutor."""
    tracker = ExperimentTracker(base_dir=results_dir)
    short_term_mem = memory_regime == "short_term"
    run_id = f"{buffer_type}_{memory_regime}_{temporal_mode}_seed{seed}"

    config_payload = {
        "buffer_type": buffer_type,
        "memory_regime": memory_regime,
        "short_term_mem": short_term_mem,
        "temporal_mode": temporal_mode,
        "n_activations": n_activations,
        "seed": seed,
        "n_generations": config.evolution.n_generations,
        "pop_size": config.evolution.pop_size,
    }
    metadata = get_environment_metadata()

    run_logger = tracker.create_run(
        run_id=run_id,
        config_payload=config_payload,
        metadata=metadata,
    )

    sim = EvolutionarySimulation(
        config=config,
        buffer_type=buffer_type,
        short_term_mem=short_term_mem,
        n_activations=n_activations,
        seed=seed,
    )

    start_time = time.time()
    status = "completed"
    try:
        if verbose:
            print(
                f"\n--- Starting: {buffer_type} | {memory_regime} | "
                f"{temporal_mode} (a={n_activations}) | seed={seed} ---"
            )
        sim.one_evolution(verbose=verbose, logger=run_logger)
    except Exception as e:
        status = "failed"
        print(f"Run {run_id} failed with error: {e}")
        raise e
    finally:
        duration = time.time() - start_time
        run_logger.finish(status=status)

        final_top = (
            run_logger.history[-1]["top_mean_fitness"] if run_logger.history else 0.0
        )
        best_top = (
            max(r["top_mean_fitness"] for r in run_logger.history)
            if run_logger.history
            else 0.0
        )

        if run_logger.history:
            plot_path = run_logger.run_dir / "run_plot.png"
            plot_single_run(
                metrics=run_logger.history,
                title=f"{buffer_type} | {memory_regime} | {temporal_mode}",
                output_path=plot_path,
            )

    return {
        "run_id": run_id,
        "buffer_type": buffer_type,
        "memory_regime": memory_regime,
        "temporal_mode": temporal_mode,
        "n_activations": n_activations,
        "seed": seed,
        "generations": config.evolution.n_generations,
        "pop_size": config.evolution.pop_size,
        "status": status,
        "final_top_mean_fitness": final_top,
        "best_top_mean_fitness": best_top,
        "duration_sec": duration,
        "timestamp": datetime.now(UTC).isoformat(),
    }


class SimulationRunner:
    def __init__(
        self,
        config: AppConfig,
        results_dir: Path | str | None = None,
        temporal_modes: list[str] | None = None,
        max_workers: int | None = None,
    ) -> None:
        self.config = config
        self.results_dir = (
            Path(results_dir)
            if results_dir is not None
            else Path(self.config.simulation.results_dir)
        )
        self.tracker = ExperimentTracker(base_dir=self.results_dir)
        self.temporal_modes = temporal_modes or list(
            self.config.simulation.activations.keys()
        )
        self.max_workers = (
            max_workers
            if max_workers is not None
            else self.config.simulation.max_workers
        )
        if (
            self.config.simulation.generations != 100
            and self.config.evolution.n_generations == 100
        ):
            self.config.evolution.n_generations = self.config.simulation.generations
        elif (
            self.config.evolution.n_generations != 100
            and self.config.simulation.generations == 100
        ):
            self.config.simulation.generations = self.config.evolution.n_generations

    def _resolve_n_activations(self, temporal_mode: str) -> int:
        if temporal_mode in self.config.simulation.activations:
            return self.config.simulation.activations[temporal_mode]
        if temporal_mode == "homogeneous":
            return 1
        return self.config.architecture.n_activations

    def _resolve_short_term_mem(self, memory_regime: str) -> bool:
        return memory_regime == "short_term"

    def run_single(
        self,
        buffer_type: str,
        memory_regime: str,
        temporal_mode: str,
        seed: int,
        verbose: bool = True,
    ) -> dict[str, Any]:
        n_activations = self._resolve_n_activations(temporal_mode)
        result = _run_single_worker(
            config=self.config,
            results_dir=self.results_dir,
            buffer_type=buffer_type,
            memory_regime=memory_regime,
            temporal_mode=temporal_mode,
            n_activations=n_activations,
            seed=seed,
            verbose=verbose,
        )
        self.tracker.record_run_completion(
            run_id=result["run_id"],
            buffer_type=result["buffer_type"],
            memory_regime=result["memory_regime"],
            temporal_mode=result["temporal_mode"],
            n_activations=result["n_activations"],
            seed=result["seed"],
            generations=result["generations"],
            pop_size=result["pop_size"],
            status=result["status"],
            final_top_mean_fitness=result["final_top_mean_fitness"],
            best_top_mean_fitness=result["best_top_mean_fitness"],
            duration_sec=result["duration_sec"],
            timestamp=result["timestamp"],
        )
        return {
            "run_id": result["run_id"],
            "status": result["status"],
            "duration_sec": result["duration_sec"],
            "final_top_mean_fitness": result["final_top_mean_fitness"],
            "best_top_mean_fitness": result["best_top_mean_fitness"],
        }

    def run_all(
        self,
        verbose: bool = True,
        generate_plots: bool = True,
        max_workers: int | None = None,
    ) -> list[dict[str, Any]]:
        buffer_types = self.config.simulation.buffer_types
        memory_regimes = self.config.simulation.memory_regimes
        num_runs = self.config.simulation.num_runs

        total_runs_count = (
            len(buffer_types)
            * len(memory_regimes)
            * len(self.temporal_modes)
            * num_runs
        )
        seeds = generate_seeds(self.config.evolution.seed, total_runs_count)

        seed_idx = 0
        tasks = []
        for b_type in buffer_types:
            for m_regime in memory_regimes:
                for t_mode in self.temporal_modes:
                    for _ in range(num_runs):
                        current_seed = seeds[seed_idx]
                        seed_idx += 1
                        n_act = self._resolve_n_activations(t_mode)
                        tasks.append((b_type, m_regime, t_mode, n_act, current_seed))

        # Resolve number of parallel workers
        workers = max_workers if max_workers is not None else self.max_workers
        if workers is None:
            cpu_count = os.cpu_count() or 1
            # Conservative default so desktop UI remains responsive
            workers = max(1, min(4, cpu_count // 2))

        all_results: list[dict[str, Any]] = []

        if workers <= 1:
            # Sequential execution
            for idx, (b_type, m_regime, t_mode, _n_act, current_seed) in enumerate(
                tasks, start=1
            ):
                result = self.run_single(
                    buffer_type=b_type,
                    memory_regime=m_regime,
                    temporal_mode=t_mode,
                    seed=current_seed,
                    verbose=verbose,
                )
                all_results.append(result)
        else:
            # Multi-process parallel execution
            if verbose:
                print(
                    f"\nStarting {total_runs_count} runs in parallel "
                    f"using {workers} worker processes..."
                )
            start_parallel = time.time()
            completed_count = 0

            with ProcessPoolExecutor(max_workers=workers) as executor:
                future_to_task = {
                    executor.submit(
                        _run_single_worker,
                        self.config,
                        self.results_dir,
                        b_type,
                        m_regime,
                        t_mode,
                        n_act,
                        current_seed,
                        False,  # Suppress per-worker verbose output
                    ): (b_type, m_regime, t_mode, current_seed)
                    for (b_type, m_regime, t_mode, n_act, current_seed) in tasks
                }

                for future in as_completed(future_to_task):
                    completed_count += 1
                    res = future.result()
                    self.tracker.record_run_completion(
                        run_id=res["run_id"],
                        buffer_type=res["buffer_type"],
                        memory_regime=res["memory_regime"],
                        temporal_mode=res["temporal_mode"],
                        n_activations=res["n_activations"],
                        seed=res["seed"],
                        generations=res["generations"],
                        pop_size=res["pop_size"],
                        status=res["status"],
                        final_top_mean_fitness=res["final_top_mean_fitness"],
                        best_top_mean_fitness=res["best_top_mean_fitness"],
                        duration_sec=res["duration_sec"],
                        timestamp=res["timestamp"],
                    )
                    all_results.append(
                        {
                            "run_id": res["run_id"],
                            "status": res["status"],
                            "duration_sec": res["duration_sec"],
                            "final_top_mean_fitness": res["final_top_mean_fitness"],
                            "best_top_mean_fitness": res["best_top_mean_fitness"],
                        }
                    )
                    if verbose:
                        print(
                            f"[{completed_count:02d}/{total_runs_count:02d}] "
                            f"Completed: {res['buffer_type']} | "
                            f"{res['memory_regime']} | "
                            f"{res['temporal_mode']} | seed={res['seed']} "
                            f"in {res['duration_sec']:.1f}s "
                            f"(Best fitness: {res['best_top_mean_fitness']:.2f})"
                        )

            if verbose:
                total_elapsed = time.time() - start_parallel
                print(
                    f"\nParallel execution of {total_runs_count} runs finished "
                    f"in {total_elapsed:.1f}s."
                )

        if generate_plots:
            if verbose:
                print("\nGenerating paper figures (Figures 6 & 7)...")
            plot_paper_figures(self.results_dir)

        return all_results
