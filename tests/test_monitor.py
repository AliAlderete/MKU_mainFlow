import pandas as pd
import pytest

from model_training.monitor import (
    population_stability_index,
    validate_evaluation_results,
)


def test_validate_evaluation_results_accepts_nominal_metrics():
    validate_evaluation_results(
        {"f1_score": 0.88, "p95_latency_ms": 120.0, "psi_amount": 0.01},
        min_f1=0.80,
        max_latency_ms=500.0,
        max_psi=0.20,
    )


@pytest.mark.parametrize(
    ("metrics", "message"),
    [
        ({"f1_score": 0.65}, "F1"),
        ({"f1_score": 0.88, "p95_latency_ms": 501.0}, "latency"),
        ({"f1_score": 0.88, "psi_amount": 0.21}, "PSI"),
    ],
)
def test_validate_evaluation_results_rejects_threshold_breaches(metrics, message):
    with pytest.raises(ValueError, match=message):
        validate_evaluation_results(metrics, 0.80, 500.0, 0.20)


def test_population_stability_index_is_zero_for_identical_data():
    values = pd.Series([1, 2, 3, 4, 5])

    assert population_stability_index(values, values) == 0.0