import csv
import json
import time
from pathlib import Path
from typing import Any


class RunLogger:
    def __init__(
        self,
        run_dir: Path,
        run_id: str,
        config_payload: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.run_dir = run_dir
        self.run_id = run_id
        self.config_payload = config_payload
        self.metadata = metadata or {}
        self.start_time = time.time()
        self.history: list[dict[str, Any]] = []

        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.run_dir / "metrics.csv"
        self.jsonl_path = self.run_dir / "metrics.jsonl"
        self.manifest_path = self.run_dir / "manifest.json"

        self.fieldnames = [
            "generation",
            "top_mean_fitness",
            "max_fitness",
            "min_fitness",
            "std_fitness",
            "elapsed_time_sec",
        ]

        self.csv_file = open(self.csv_path, mode="w", newline="", encoding="utf-8")
        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=self.fieldnames)
        self.csv_writer.writeheader()
        self.csv_file.flush()

        self.jsonl_file = open(self.jsonl_path, mode="w", encoding="utf-8")

        self._save_manifest(status="running")

    def _save_manifest(
        self, status: str, final_metrics: dict[str, Any] | None = None
    ) -> None:
        payload = {
            "run_id": self.run_id,
            "status": status,
            "start_time": self.start_time,
            "duration_sec": time.time() - self.start_time,
            "config": self.config_payload,
            "metadata": self.metadata,
            "final_metrics": final_metrics or {},
        }
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def log_generation(
        self,
        generation: int,
        top_mean: float,
        max_fitness: float,
        min_fitness: float,
        std_fitness: float,
        elapsed_sec: float,
    ) -> None:
        row = {
            "generation": generation,
            "top_mean_fitness": round(top_mean, 4),
            "max_fitness": round(max_fitness, 4),
            "min_fitness": round(min_fitness, 4),
            "std_fitness": round(std_fitness, 4),
            "elapsed_time_sec": round(elapsed_sec, 4),
        }
        self.history.append(row)

        self.csv_writer.writerow(row)
        self.csv_file.flush()

        self.jsonl_file.write(json.dumps(row) + "\n")
        self.jsonl_file.flush()

    def finish(self, status: str = "completed") -> None:
        final_metrics: dict[str, Any] = {}
        if self.history:
            last = self.history[-1]
            best_top_mean = max(r["top_mean_fitness"] for r in self.history)
            overall_max = max(r["max_fitness"] for r in self.history)
            final_metrics = {
                "final_top_mean_fitness": last["top_mean_fitness"],
                "best_top_mean_fitness": best_top_mean,
                "overall_max_fitness": overall_max,
                "total_generations": len(self.history),
            }

        self._save_manifest(status=status, final_metrics=final_metrics)

        if not self.csv_file.closed:
            self.csv_file.close()
        if not self.jsonl_file.closed:
            self.jsonl_file.close()


class ExperimentTracker:
    def __init__(self, base_dir: Path | str = "results") -> None:
        self.base_dir = Path(base_dir)
        self.runs_dir = self.base_dir / "runs"
        self.figures_dir = self.base_dir / "figures"
        self.index_csv_path = self.base_dir / "experiments_index.csv"

        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

        self.index_fields = [
            "run_id",
            "buffer_type",
            "memory_regime",
            "temporal_mode",
            "n_activations",
            "seed",
            "generations",
            "pop_size",
            "status",
            "final_top_mean_fitness",
            "best_top_mean_fitness",
            "duration_sec",
            "timestamp",
        ]
        self._ensure_index_file()

    def _ensure_index_file(self) -> None:
        if not self.index_csv_path.exists():
            with open(self.index_csv_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.index_fields)
                writer.writeheader()

    def create_run(
        self,
        run_id: str,
        config_payload: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> RunLogger:
        run_dir = self.runs_dir / run_id
        return RunLogger(
            run_dir=run_dir,
            run_id=run_id,
            config_payload=config_payload,
            metadata=metadata,
        )

    def record_run_completion(
        self,
        run_id: str,
        buffer_type: str,
        memory_regime: str,
        temporal_mode: str,
        n_activations: int,
        seed: int,
        generations: int,
        pop_size: int,
        status: str,
        final_top_mean_fitness: float,
        best_top_mean_fitness: float,
        duration_sec: float,
        timestamp: str,
    ) -> None:
        row = {
            "run_id": run_id,
            "buffer_type": buffer_type,
            "memory_regime": memory_regime,
            "temporal_mode": temporal_mode,
            "n_activations": n_activations,
            "seed": seed,
            "generations": generations,
            "pop_size": pop_size,
            "status": status,
            "final_top_mean_fitness": round(final_top_mean_fitness, 4),
            "best_top_mean_fitness": round(best_top_mean_fitness, 4),
            "duration_sec": round(duration_sec, 2),
            "timestamp": timestamp,
        }
        with open(self.index_csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.index_fields)
            writer.writerow(row)

    def load_index(self) -> list[dict[str, Any]]:
        if not self.index_csv_path.exists():
            return []
        with open(self.index_csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return list(reader)

    def load_run_metrics(self, run_id: str) -> list[dict[str, float]]:
        metrics_file = self.runs_dir / run_id / "metrics.csv"
        if not metrics_file.exists():
            return []
        results: list[dict[str, float]] = []
        with open(metrics_file, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                results.append({k: float(v) for k, v in row.items()})
        return results
