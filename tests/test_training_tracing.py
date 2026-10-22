import mlflow
import numpy as np

from model_training.train_experiments import run_experiments


def test_run_experiments_creates_trace_and_runs(tmp_path):
    tracking_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    experiment_id = mlflow.create_experiment(
        "Training_Experiments",
        artifact_location=str(tmp_path / "artifacts"),
    )
    mlflow.set_experiment("Training_Experiments")

    try:
        X_train = np.array([[0.0], [1.0], [2.0], [3.0]])
        y_train = np.array([1.0, 3.0, 5.0, 7.0])
        X_test = np.array([[4.0], [5.0]])
        y_test = np.array([9.0, 11.0])

        results = run_experiments(X_train, X_test, y_train, y_test)
        trace_id = mlflow.get_last_active_trace_id()
        mlflow.flush_trace_async_logging()
        trace = mlflow.get_trace(trace_id)
        runs = mlflow.search_runs(experiment_ids=[experiment_id])

        assert trace is not None
        assert any(span.name == "run_experiments" for span in trace.data.spans)
        assert len(results) == 3
        assert len(runs) == 3
        assert runs["metrics.test_mean_squared_error"].notna().all()
    finally:
        mlflow.flush_trace_async_logging()
        mlflow.set_tracking_uri(tracking_uri)
