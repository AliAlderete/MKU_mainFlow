import joblib
import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow.entities import SpanType

from model_training.train_experiments import run_experiments, train_best_model


@pytest.fixture(autouse=True)
def mlflow_tracking_store(tmp_path):
    tracking_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    mlflow.create_experiment(
        "Training_Experiments",
        artifact_location=str(tmp_path / "artifacts"),
    )
    mlflow.set_experiment("Training_Experiments")
    yield
    mlflow.flush_trace_async_logging()
    mlflow.set_tracking_uri(tracking_uri)


def _trace_call(function, name):
    traced_function = mlflow.trace(function, name=name, span_type=SpanType.CHAIN)
    result = traced_function()
    trace_id = mlflow.get_last_active_trace_id()
    assert trace_id is not None

    mlflow.flush_trace_async_logging()
    trace = mlflow.get_trace(trace_id)
    assert trace is not None
    assert any(span.name == name for span in trace.data.spans)
    return result


def test_run_experiments_tracks_parameters_and_metrics(capsys):
    X_train = np.array([[0.0], [1.0], [2.0], [3.0]])
    y_train = np.array([1.0, 3.0, 5.0, 7.0])
    X_test = np.array([[4.0], [5.0]])
    y_test = np.array([9.0, 11.0])

    results = _trace_call(
        lambda: run_experiments(X_train, X_test, y_train, y_test),
        "run_experiments_test",
    )
    output = capsys.readouterr().out
    logged_runs = mlflow.search_runs(experiment_names=["Training_Experiments"])

    assert len(results) == 3
    assert len(logged_runs) == 3
    assert logged_runs["metrics.test_mean_squared_error"].notna().all()
    assert results[0]["parameters"] == {
        "fit_intercept": True,
        "positive": False,
    }
    assert results[0]["test_mean_squared_error"] == 0.0
    assert "parameters:" in output
    assert "coefficients:" in output
    assert "test_r_squared:" in output


def test_train_best_model_saves_the_lowest_mse_model(tmp_path):
    X = pd.DataFrame({"feature": [1.0, 2.0, 3.0]})
    y = np.array([3.0, 5.0, 7.0])
    results = [
        {
            "test_mean_squared_error": 2.0,
            "parameters": {"fit_intercept": True, "positive": False},
        },
        {
            "test_mean_squared_error": 1.0,
            "parameters": {"fit_intercept": False, "positive": False},
        },
    ]
    artifact_path = tmp_path / "model" / "best_model.joblib"

    artifact_summary = _trace_call(
        lambda: {
            "parameters": train_best_model(X, y, results, artifact_path)["parameters"],
            "feature_columns": list(X.columns),
        },
        "train_best_model_test",
    )

    artifact = joblib.load(artifact_path)
    logged_runs = mlflow.search_runs(experiment_names=["Training_Experiments"])
    assert artifact_summary["parameters"] == artifact["parameters"]
    assert len(logged_runs) == 1
    assert logged_runs.iloc[0]["tags.mlflow.runName"] == "Best_Model"
    assert logged_runs.iloc[0]["metrics.feature_count"] == 1
    assert artifact_summary["feature_columns"] == artifact["feature_columns"]
    assert artifact["parameters"] == {
        "fit_intercept": False,
        "positive": False,
    }
    assert artifact["feature_columns"] == ["feature"]
    assert artifact["model"].n_features_in_ == 1
