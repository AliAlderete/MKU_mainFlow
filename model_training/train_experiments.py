import json
import os
from collections.abc import Mapping, Sequence

import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
from mlflow.entities import SpanType
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


EXPERIMENTS = (
    {"fit_intercept": True, "positive": False},
    {"fit_intercept": False, "positive": False},
    {"fit_intercept": True, "positive": True},
)
MLFLOW_EXPERIMENT_NAME = os.getenv(
    "MLFLOW_EXPERIMENT_NAME", "tips-linear-regression"
)


@mlflow.trace(name="run_experiments", span_type=SpanType.CHAIN)
def run_experiments(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    experiments: Sequence[Mapping[str, bool]] = EXPERIMENTS,
) -> list[dict[str, object]]:
    results = []
 
    for experiment_number, parameters in enumerate(experiments, start=1):
        with mlflow.start_run(
            run_name=f"linear-regression-{experiment_number}",
            nested=mlflow.active_run() is not None,
        ):
            model = LinearRegression(**parameters)
            model.fit(X_train, y_train)
 
            predictions = model.predict(X_test)
            metrics = {
                "test_mean_squared_error": float(
                    mean_squared_error(y_test, predictions)
                ),
                "test_r_squared": float(r2_score(y_test, predictions)),
            }
            result = {
                "experiment": experiment_number,
                "parameters": dict(parameters),
                "intercept": float(model.intercept_),
                "coefficients": model.coef_.tolist(),
                "model_name": f"Experiment {experiment_number}",
                **metrics,
            }
            results.append(result)

            mlflow.log_params(parameters)
            mlflow.log_metrics(metrics)
            mlflow.set_tags(
                {
                    "experiment_number": str(experiment_number),
                    "model_class": "sklearn.linear_model.LinearRegression",
                }
            )

            print(f"\n--- Experiment {experiment_number} ---")
            print(f"parameters: {json.dumps(result['parameters'], sort_keys=True)}")
            print(f"intercept: {result['intercept']:.6f}")
            print(f"coefficients: {json.dumps(result['coefficients'])}")
            print(f"test_mean_squared_error: {metrics['test_mean_squared_error']:.4f}")
            print(f"test_r_squared: {metrics['test_r_squared']:.4f}")

    return results


@mlflow.trace(name="training_pipeline", span_type=SpanType.CHAIN)
def run_training_pipeline() -> list[dict[str, object]]:
    np.random.seed(42)

    tips = sns.load_dataset("tips")
    data = pd.get_dummies(tips, drop_first=True)
    X = data.drop(columns=["tip"])
    y = data["tip"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    with mlflow.start_run(
        run_name="tips-linear-regression",
        nested=mlflow.active_run() is not None,
    ):
        mlflow.log_params(
            {
                "dataset": "seaborn_tips",
                "random_state": 42,
                "test_size": 0.2,
                "feature_count": X_train.shape[1],
                "training_rows": X_train.shape[0],
                "test_rows": X_test.shape[0],
            }
        )
        return run_experiments(
            X_train_scaled,
            X_test_scaled,
            y_train,
            y_test,
        )


def main() -> None:
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    run_training_pipeline()


if __name__ == "__main__":
    main()