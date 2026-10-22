# MKU_main
Target repository

## Model training experiments

Run the parameter experiments with:

```bash
python -m model_training.train_experiments
```

The script uses the same Seaborn `tips` data and preprocessing pipeline as
`starter/baseline_model.py`. It compares linear regression with different
intercept and positivity settings, printing each experiment's settings,
learned coefficients, and test metrics.

On pushes that change the training code or its dependencies, the
`Model Training Artifact` workflow selects the parameters with the lowest test
mean squared error, retrains that model, and uploads
`artifacts/best_model.joblib`. The artifact includes the fitted model, scaler,
feature columns, and selected parameters.

## Production fraud model monitoring

Run the monitor locally with:

```bash
python -m model_training.monitor \
	--data data/daily_production_inferences.csv \
	--baseline data/reference_baseline.csv
```

The inference CSV must contain `label` and `prediction`; optional `latency_ms`
is checked using its 95th percentile. Numeric features present in both the
inference and baseline CSVs are checked with population stability index (PSI).
The defaults fail below F1 `0.80`, above p95 latency `500` ms, or above PSI
`0.20`; each threshold can be overridden by a command-line option. MLflow
records classifier evaluation metrics and input artifacts under
`Production_Model_Monitoring`. Set `MLFLOW_TRACKING_URI` to a shared tracking
server to view runs centrally.

The `Production Model Monitoring` workflow runs daily or by manual dispatch.
For manual runs, set `inference_file` to `data/drifted_inferences.csv` to exercise
the failing F1 scenario. The workflow creates an Issue when evaluation fails.
To enable the retraining extension,
configure the repository secret `RETRAINING_DISPATCH_TOKEN` with permission to
dispatch repository events; failures then emit the `fraud-model-drift` event.
The checked-in CSV files are small examples and should be replaced by the
production inference export and a fixed reference sample for deployment.
