import copy
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.config.schema import AppConfig, SimulationConfig
from src.main import main
from src.simulation.evolution import EvolutionarySimulation
from src.simulation.logger import ExperimentTracker
from src.simulation.reproducibility import generate_seeds
from src.simulation.runner import SimulationRunner
from src.simulation.visualization import plot_paper_figures


class TestSimulationComponents(unittest.TestCase):
    def setUp(self) -> None:
        self.test_dir = Path("tests_scratch_results")
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
        self.test_dir.mkdir(parents=True, exist_ok=True)

        config_path = Path("configs/config.yaml")
        self.config = AppConfig.from_yaml(config_path)

        self.fast_config = copy.deepcopy(self.config)
        self.fast_config.evolution.n_generations = 1
        self.fast_config.evolution.pop_size = 4
        self.fast_config.agent.max_agent_steps = 20
        self.fast_config.simulation.max_workers = 2

    def tearDown(self) -> None:
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_reproducibility_seed_generation(self) -> None:
        seeds_a = generate_seeds(42, 5)
        seeds_b = generate_seeds(42, 5)
        self.assertEqual(seeds_a, seeds_b)
        self.assertEqual(len(seeds_a), 5)
        self.assertEqual(len(set(seeds_a)), 5)

    def test_simulation_determinism(self) -> None:
        sim_1 = EvolutionarySimulation(
            config=self.fast_config,
            buffer_type="average",
            short_term_mem=True,
            n_activations=3,
            seed=1234,
        )
        history_1 = sim_1.one_evolution(verbose=False)

        sim_2 = EvolutionarySimulation(
            config=self.fast_config,
            buffer_type="average",
            short_term_mem=True,
            n_activations=3,
            seed=1234,
        )
        history_2 = sim_2.one_evolution(verbose=False)

        np.testing.assert_allclose(history_1, history_2)

    def test_logger_and_tracker(self) -> None:
        tracker = ExperimentTracker(base_dir=self.test_dir)
        run_logger = tracker.create_run(
            run_id="test_run_1",
            config_payload={"buffer_type": "average"},
        )
        run_logger.log_generation(1, 5.0, 10.0, 1.0, 2.0, 0.1)
        run_logger.log_generation(2, 7.0, 12.0, 2.0, 2.5, 0.1)
        run_logger.finish(status="completed")

        metrics_csv = self.test_dir / "runs" / "test_run_1" / "metrics.csv"
        manifest_json = self.test_dir / "runs" / "test_run_1" / "manifest.json"

        self.assertTrue(metrics_csv.exists())
        self.assertTrue(manifest_json.exists())

        loaded = tracker.load_run_metrics("test_run_1")
        self.assertEqual(len(loaded), 2)
        self.assertEqual(loaded[0]["generation"], 1.0)
        self.assertEqual(loaded[1]["top_mean_fitness"], 7.0)

    def test_simulation_runner_single_and_plots(self) -> None:
        runner = SimulationRunner(
            config=self.fast_config,
            results_dir=self.test_dir,
            temporal_modes=["heterogeneous", "homogeneous"],
        )

        res_h = runner.run_single(
            buffer_type="average",
            memory_regime="short_term",
            temporal_mode="heterogeneous",
            seed=101,
            verbose=False,
        )
        self.assertEqual(res_h["status"], "completed")

        res_o = runner.run_single(
            buffer_type="average",
            memory_regime="short_term",
            temporal_mode="homogeneous",
            seed=102,
            verbose=False,
        )
        self.assertEqual(res_o["status"], "completed")

        figures = plot_paper_figures(results_dir=self.test_dir)
        self.assertEqual(len(figures), 4)
        for fig_path in figures:
            self.assertTrue(fig_path.exists())
            self.assertGreater(fig_path.stat().st_size, 1000)

    def test_simulation_runner_full_matrix(self) -> None:
        grid_config = copy.deepcopy(self.fast_config)
        grid_config.evolution.n_generations = 1
        grid_config.evolution.pop_size = 4
        grid_config.agent.max_agent_steps = 15
        grid_config.simulation.num_runs = 1

        runner = SimulationRunner(
            config=grid_config,
            results_dir=self.test_dir,
            temporal_modes=["heterogeneous", "homogeneous"],
        )

        results = runner.run_all(verbose=False, generate_plots=True)
        # 3 buffer types * 2 memory regimes * 2 temporal modes = 12 runs
        self.assertEqual(len(results), 12)
        for r in results:
            self.assertEqual(r["status"], "completed")

        fig6_cummax = (
            self.test_dir / "figures" / "figure_6_short_term_memory_cummax.png"
        )
        fig7_cummax = self.test_dir / "figures" / "figure_7_long_term_memory_cummax.png"
        fig6_raw = self.test_dir / "figures" / "figure_6_short_term_memory_raw.png"
        fig7_raw = self.test_dir / "figures" / "figure_7_long_term_memory_raw.png"
        index_csv = self.test_dir / "experiments_index.csv"

        self.assertTrue(fig6_cummax.exists())
        self.assertTrue(fig7_cummax.exists())
        self.assertTrue(fig6_raw.exists())
        self.assertTrue(fig7_raw.exists())
        self.assertTrue(index_csv.exists())

    def test_config_simulation_activations_and_results_dir(self) -> None:
        self.assertEqual(self.config.simulation.results_dir, "results")
        self.assertEqual(
            self.config.simulation.activations,
            {"heterogeneous": 3, "homogeneous": 1},
        )

        with self.assertRaises(ValueError):
            SimulationConfig(activations={})

        with self.assertRaises(ValueError):
            SimulationConfig(activations={"heterogeneous": 0})

        with self.assertRaises(ValueError):
            SimulationConfig(results_dir="   ")

    def test_runner_defaults_from_config(self) -> None:
        cfg = copy.deepcopy(self.fast_config)
        cfg.simulation.results_dir = str(self.test_dir)
        cfg.simulation.activations = {"heterogeneous": 4, "homogeneous": 2}

        runner = SimulationRunner(config=cfg)
        self.assertEqual(runner.results_dir, self.test_dir)
        self.assertEqual(runner.temporal_modes, ["heterogeneous", "homogeneous"])
        self.assertEqual(runner._resolve_n_activations("heterogeneous"), 4)
        self.assertEqual(runner._resolve_n_activations("homogeneous"), 2)

    def test_main_cli_plot_flag(self) -> None:
        # Pre-populate dummy metrics in test_dir
        tracker = ExperimentTracker(base_dir=self.test_dir)
        run_logger = tracker.create_run(
            run_id="run_test",
            config_payload={"buffer_type": "average"},
        )
        run_logger.log_generation(1, 2.0, 5.0, 1.0, 1.0, 0.1)
        run_logger.finish(status="completed")

        tracker.record_run_completion(
            run_id="run_test",
            buffer_type="average",
            memory_regime="short_term",
            temporal_mode="heterogeneous",
            n_activations=3,
            seed=42,
            generations=1,
            pop_size=4,
            status="completed",
            final_top_mean_fitness=2.0,
            best_top_mean_fitness=2.0,
            duration_sec=0.1,
            timestamp="2026-09-16T00:00:00Z",
        )

        cfg = copy.deepcopy(self.fast_config)
        cfg.simulation.results_dir = str(self.test_dir)
        cfg_file = self.test_dir / "test_cfg.yaml"
        with open(cfg_file, "w") as f:
            import yaml

            yaml.dump(cfg.model_dump(mode="json"), f)

        with patch("sys.argv", ["main.py", "--config", str(cfg_file), "--plot"]):
            main()

    def test_config_simulation_max_workers(self) -> None:
        cfg = SimulationConfig(max_workers=4)
        self.assertEqual(cfg.max_workers, 4)

        with self.assertRaises(ValueError):
            SimulationConfig(max_workers=0)

    def test_simulation_runner_parallel_execution(self) -> None:
        grid_config = copy.deepcopy(self.fast_config)
        grid_config.evolution.n_generations = 1
        grid_config.evolution.pop_size = 4
        grid_config.agent.max_agent_steps = 15
        grid_config.simulation.num_runs = 1

        runner = SimulationRunner(
            config=grid_config,
            results_dir=self.test_dir,
            temporal_modes=["heterogeneous", "homogeneous"],
            max_workers=2,
        )

        results = runner.run_all(verbose=False, generate_plots=False)
        self.assertEqual(len(results), 12)
        for r in results:
            self.assertEqual(r["status"], "completed")

    def test_main_config_workers(self) -> None:
        cfg = copy.deepcopy(self.fast_config)
        cfg.simulation.results_dir = str(self.test_dir)
        cfg.simulation.max_workers = 6
        cfg_file = self.test_dir / "test_cfg_workers.yaml"
        with open(cfg_file, "w") as f:
            import yaml

            yaml.dump(cfg.model_dump(mode="json"), f)

        # Verify main initializes SimulationRunner with configuration from file
        with (
            patch("src.main.SimulationRunner") as mock_runner,
            patch(
                "sys.argv",
                ["main.py", "--config", str(cfg_file)],
            ),
        ):
            instance = mock_runner.return_value
            instance.run_all.return_value = []
            main()
            mock_runner.assert_called_once()
            _, kwargs = mock_runner.call_args
            passed_cfg = kwargs.get("config")
            self.assertEqual(passed_cfg.simulation.max_workers, 6)


if __name__ == "__main__":
    unittest.main()
