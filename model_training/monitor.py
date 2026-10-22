import argparse
import sys
from pathlib import Path
from typing import Mapping

import mlflow
import numpy as np
import pandas as pd
from mlflow.models import evaluate


DEFAULT_DATA = Path("data/daily_production_inferences.csv")
EXCLUDED_FEATURES = {"label", "prediction", "latency_ms"}


def population_stability_index(
    reference: pd.Series, current: pd.Series, bins: int = 10
) -> float:
    reference_values = reference.dropna().to_numpy(dtype=float)
    current_values = current.dropna().to_numpy(dtype=float)
    if reference_values.size == 0 or current_values.size == 0:
        raise ValueError("PSI requires non-empty numeric feature values.")

    quantiles = np.unique(np.quantile(reference_values, np.linspace(0, 1, bins + 1)))
    if quantiles.size < 2:
        edges = np.array([-np.inf, quantiles[0], np.inf])
    else:
        edges = np.concatenate(([-np.inf], quantiles[1:-1], [np.inf]))

    reference_counts = np.histogram(reference_values, bins=edges)[0]
    current_counts = np.histogram(current_values, bins=edges)[0]
    reference_share = np.clip(reference_counts / reference_counts.sum(), 1e-6, None)
    current_share = np.clip(current_counts / current_counts.sum(), 1e-6, None)
    return float(
        np.sum((current_share - reference_share) * np.log(current_share / reference_share))
    )


def validate_evaluation_results(
    metrics: Mapping[str, float],
    min_f1: float,
    max_latency_ms: float,
    max_psi: float,
) -> None:
    failures = []
    f1_score = metrics.get("f1_score")
    if f1_score is None:
        raise ValueError("MLflow evaluation did not produce an f1_score metric.")
    if f1_score < min_f1:
        failures.append(f"F1 {f1_score:.4f} is below the minimum {min_f1:.4f}")

    latency = metrics.get("p95_latency_ms")
    if latency is not None and latency > max_latency_ms:
        failures.append(
            f"p95 latency {latency:.2f} ms exceeds the maximum {max_latency_ms:.2f} ms"
        )

    drifting_features = {
        key.removeprefix("psi_"): value
        for key, value in metrics.items()
        if key.startswith("psi_") and value > max_psi
    }
    if drifting_features:
        details = ", ".join(
            f"{feature}={value:.4f}" for feature, value in drifting_features.items()
        )
        failures.append(f"PSI exceeds the maximum {max_psi:.4f}: {details}")

    if failures:
        raise ValueError("Model monitoring threshold failure: " + "; ".join(failures))


def monitor(
    data_path: Path = DEFAULT_DATA,
    baseline_path: Path | None = None,
    min_f1: float = 0.80,
    max_latency_ms: float = 500.0,
    max_psi: float = 0.20,
    experiment_name: str = "Production_Model_Monitoring",
) -> dict[str, float]:
    data = pd.read_csv(data_path)
    required_columns = {"label", "prediction"}
    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        raise ValueError(f"Inference data is missing columns: {sorted(missing_columns)}")
    if data.empty:
        raise ValueError("Inference data must contain at least one row.")

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run() as run:
        evaluation = evaluate(
            model=None,
            data=data,
            targets="label",
            predictions="prediction",
            model_type="classifier",
        )
        metrics = {key: float(value) for key, value in evaluation.metrics.items()}

        if "latency_ms" in data.columns:
            latency = float(
                pd.to_numeric(data["latency_ms"], errors="raise").quantile(0.95)
            )
            metrics["p95_latency_ms"] = latency
            mlflow.log_metric("p95_latency_ms", latency)

        if baseline_path is not None:
            baseline = pd.read_csv(baseline_path)
            feature_columns = [
                column
                for column in data.select_dtypes(include="number").columns
                if column not in EXCLUDED_FEATURES and column in baseline.columns
            ]
            for column in feature_columns:
                psi = population_stability_index(baseline[column], data[column])
                metrics[f"psi_{column}"] = psi
                mlflow.log_metric(f"psi_{column}", psi)
            mlflow.log_artifact(str(baseline_path), artifact_path="baseline")

        mlflow.log_artifact(str(data_path), artifact_path="inference_data")
        mlflow.set_tags(
            {
                "monitoring.data_path": str(data_path),
                "monitoring.baseline_path": str(baseline_path or "not_configured"),
            }
        )
        validate_evaluation_results(metrics, min_f1, max_latency_ms, max_psi)
        print(f"Monitoring passed. MLflow run: {run.info.run_id}")
        return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Monitor production fraud predictions.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--min-f1", type=float, default=0.80)
    parser.add_argument("--max-latency-ms", type=float, default=500.0)
    parser.add_argument("--max-psi", type=float, default=0.20)
    parser.add_argument(
        "--experiment-name", default="Production_Model_Monitoring"
    )
    args = parser.parse_args()

    try:
        monitor(
            data_path=args.data,
            baseline_path=args.baseline,
            min_f1=args.min_f1,
            max_latency_ms=args.max_latency_ms,
            max_psi=args.max_psi,
            experiment_name=args.experiment_name,
        )
    except Exception as error:
        print(f"Monitoring failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())