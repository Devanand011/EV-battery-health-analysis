from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import joblib
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.data_loader import filter_discharge_cycles, load_nasa_battery_directory
from src.features import DEFAULT_EOL_SOH_THRESHOLD, build_training_frame
from src.models import anomaly_to_risk_level
from src.train import run_training


ARTIFACT_DIR = Path("artifacts")
DATA_DIR = Path("data")
MODEL_PATH = ARTIFACT_DIR / "battery_model_bundle.joblib"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"
CONFIG_PATH = ARTIFACT_DIR / "model_config.json"
WEB_DIR = Path("web")

app = FastAPI(title="EV Battery Health Web App", version="1.0.0")
app.mount("/static", StaticFiles(directory=WEB_DIR / "static"), name="static")

STATE: dict[str, Any] = {
    "bundle": None,
    "frame": None,
    "scored_frame": None,
    "metrics": {},
    "eol_soh_threshold": DEFAULT_EOL_SOH_THRESHOLD,
}


class RetrainRequest(BaseModel):
    test_battery: str = "B0007"
    eol_soh_threshold: float = DEFAULT_EOL_SOH_THRESHOLD
    random_state: int = 42


def _load_metrics() -> dict[str, float]:
    if not METRICS_PATH.exists():
        return {}
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


def _load_threshold() -> float:
    if not CONFIG_PATH.exists():
        return DEFAULT_EOL_SOH_THRESHOLD
    data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return float(data.get("eol_soh_threshold", DEFAULT_EOL_SOH_THRESHOLD))


def _ensure_training() -> None:
    retrain_flag = os.getenv("RETRAIN_ON_START", "0").strip() == "1"
    if MODEL_PATH.exists() and not retrain_flag:
        return
    run_training(
        data_dir=DATA_DIR,
        artifact_dir=ARTIFACT_DIR,
        test_battery=os.getenv("TEST_BATTERY", "B0007"),
        random_state=int(os.getenv("RANDOM_STATE", "42")),
        eol_soh_threshold=float(os.getenv("EOL_SOH_THRESHOLD", str(DEFAULT_EOL_SOH_THRESHOLD))),
    )


def _refresh_state() -> None:
    _ensure_training()
    eol_soh_threshold = _load_threshold()
    raw_df = load_nasa_battery_directory(DATA_DIR)
    discharge_df = filter_discharge_cycles(raw_df)
    frame = build_training_frame(discharge_df, eol_soh_threshold=eol_soh_threshold)
    bundle = joblib.load(MODEL_PATH)

    x = frame[bundle.feature_columns].copy()
    soh_pred = bundle.soh_model.predict(x)
    rul_pred = bundle.rul_model.predict(x).clip(min=0.0)
    anomaly_score = bundle.anomaly_model.decision_function(x)

    scored = frame.copy()
    scored["soh_pred"] = soh_pred
    scored["rul_pred_cycles"] = rul_pred
    scored["anomaly_score"] = anomaly_score
    scored["risk_level"] = anomaly_to_risk_level(anomaly_score)

    STATE["bundle"] = bundle
    STATE["frame"] = frame
    STATE["scored_frame"] = scored
    STATE["metrics"] = _load_metrics()
    STATE["eol_soh_threshold"] = eol_soh_threshold


@app.on_event("startup")
def startup_event() -> None:
    _refresh_state()


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    index_path = WEB_DIR / "index.html"
    return index_path.read_text(encoding="utf-8")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/metrics")
def metrics() -> dict[str, Any]:
    return {
        "metrics": STATE["metrics"],
        "eol_soh_threshold": STATE["eol_soh_threshold"],
    }


@app.get("/api/batteries")
def batteries() -> dict[str, list[str]]:
    scored = STATE["scored_frame"]
    if scored is None:
        raise HTTPException(status_code=500, detail="State not initialized.")
    ids = sorted(scored["battery_id"].unique().tolist())
    return {"batteries": ids}


@app.get("/api/battery/{battery_id}")
def battery_details(battery_id: str) -> dict[str, Any]:
    scored = STATE["scored_frame"]
    if scored is None:
        raise HTTPException(status_code=500, detail="State not initialized.")

    subset = scored.loc[scored["battery_id"] == battery_id].copy()
    if subset.empty:
        raise HTTPException(status_code=404, detail=f"Battery not found: {battery_id}")

    latest = subset.iloc[-1]
    payload = {
        "battery_id": battery_id,
        "latest": {
            "cycle_number": int(latest["cycle_number"]),
            "capacity_ah": float(latest["capacity_ah"]),
            "soh_pred": float(latest["soh_pred"]),
            "rul_pred_cycles": float(latest["rul_pred_cycles"]),
            "risk_level": str(latest["risk_level"]),
        },
        "series": {
            "cycle_number": subset["cycle_number"].astype(int).tolist(),
            "capacity_ah": subset["capacity_ah"].astype(float).tolist(),
            "soh_actual": subset["soh"].astype(float).tolist(),
            "soh_pred": subset["soh_pred"].astype(float).tolist(),
            "rul_actual_cycles": subset["rul_cycles"].astype(float).tolist(),
            "rul_pred_cycles": subset["rul_pred_cycles"].astype(float).tolist(),
            "risk_level": subset["risk_level"].astype(str).tolist(),
        },
    }
    return payload


@app.post("/api/retrain")
def retrain(request: RetrainRequest) -> dict[str, Any]:
    metrics, _ = run_training(
        data_dir=DATA_DIR,
        artifact_dir=ARTIFACT_DIR,
        test_battery=request.test_battery,
        random_state=request.random_state,
        eol_soh_threshold=request.eol_soh_threshold,
    )
    _refresh_state()
    return {
        "status": "ok",
        "metrics": metrics,
        "eol_soh_threshold": request.eol_soh_threshold,
    }


if __name__ == "__main__":
    uvicorn.run("webapp:app", host="127.0.0.1", port=8000, reload=False)

