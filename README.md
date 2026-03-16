# EV Battery Health Analyzer (V2 Web App)

V2 includes a one-step web app on top of the ML pipeline:

- Dataset loader for NASA battery `.mat` files
- Feature engineering for degradation trend signals
- Baseline SOH/RUL models and anomaly-based risk levels
- FastAPI backend with JSON endpoints
- Browser dashboard (HTML/JS)
- Auto-training bootstrap on app startup

## Project Structure

```text
.
|-- battery_analysis.py
|-- data/
|-- requirements.txt
|-- webapp.py
|-- web/
|   `-- index.html
`-- src/
    |-- data_loader.py
    |-- features.py
    |-- models.py
    |-- train.py
    |-- infer.py
    `-- dashboard.py
```

## Setup

```bash
python -m pip install -r requirements.txt
```

## One-Step Run (Recommended)

Run this single command:

```bash
python webapp.py
```

What it does:

- Ensures model artifacts exist (trains automatically if missing)
- Loads data and serves predictions
- Starts web app at `http://127.0.0.1:8000`

Optional environment variables:

- `RETRAIN_ON_START=1` force retraining on startup
- `EOL_SOH_THRESHOLD=0.80` set EOL threshold
- `TEST_BATTERY=B0007` holdout battery for training

## Screenshots

![Dashboard overview](web/static/screenshots/Screenshot%202026-03-16%20150616.png)

![Cycle inspector and quick meanings](web/static/screenshots/Screenshot%202026-03-16%20150522.png)

![SOH and RUL trends](web/static/screenshots/Screenshot%202026-03-16%20150544.png)

![Definitions and model explanation](web/static/screenshots/Screenshot%202026-03-16%20150603.png)

## API Endpoints

- `GET /api/health`
- `GET /api/metrics`
- `GET /api/batteries`
- `GET /api/battery/{battery_id}`
- `POST /api/retrain`

## CLI (Optional)

```bash
python -m src.train --data-dir data --artifact-dir artifacts --test-battery B0007 --eol-soh-threshold 0.80
```

Outputs:

- `artifacts/battery_model_bundle.joblib`
- `artifacts/metrics.json`
- `artifacts/test_predictions.csv`
- `artifacts/model_config.json`

## Infer

```bash
python -m src.infer --artifact-path artifacts/battery_model_bundle.joblib --input-data-dir data --output-csv artifacts/inference_predictions.csv --eol-soh-threshold 0.80
```

## Legacy Plot Script

`battery_analysis.py` now uses the shared loader and still plots B0005 capacity fade.

## RUL Definition

RUL is end-of-life threshold based in V1.1:

- EOL is reached when `SOH <= eol_soh_threshold` (default `0.80`)
- `RUL = max(eol_cycle - current_cycle, 0)`
