from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    from .features import FEATURE_COLUMNS
except ImportError:
    from features import FEATURE_COLUMNS


@dataclass
class BatteryModelBundle:
    feature_columns: list[str]
    soh_model: RandomForestRegressor
    rul_model: RandomForestRegressor
    anomaly_model: IsolationForest


def train_models(train_df: pd.DataFrame, random_state: int = 42) -> BatteryModelBundle:
    x_train = train_df[FEATURE_COLUMNS].copy()
    y_soh = train_df["soh"].astype(float).values
    y_rul = train_df["rul_cycles"].astype(float).values

    soh_model = RandomForestRegressor(
        n_estimators=300, max_depth=12, random_state=random_state, n_jobs=1
    )
    rul_model = RandomForestRegressor(
        n_estimators=300, max_depth=14, random_state=random_state, n_jobs=1
    )
    anomaly_model = IsolationForest(
        n_estimators=300, contamination=0.08, random_state=random_state
    )

    soh_model.fit(x_train, y_soh)
    rul_model.fit(x_train, y_rul)
    anomaly_model.fit(x_train)

    return BatteryModelBundle(
        feature_columns=list(FEATURE_COLUMNS),
        soh_model=soh_model,
        rul_model=rul_model,
        anomaly_model=anomaly_model,
    )


def evaluate_models(
    bundle: BatteryModelBundle, test_df: pd.DataFrame
) -> tuple[dict[str, float], pd.DataFrame]:
    x_test = test_df[bundle.feature_columns].copy()
    y_soh = test_df["soh"].astype(float).values
    y_rul = test_df["rul_cycles"].astype(float).values

    soh_pred = bundle.soh_model.predict(x_test)
    rul_pred = np.clip(bundle.rul_model.predict(x_test), a_min=0.0, a_max=None)
    anomaly_raw = bundle.anomaly_model.decision_function(x_test)

    metrics = {
        "soh_mae": float(mean_absolute_error(y_soh, soh_pred)),
        "soh_rmse": float(np.sqrt(mean_squared_error(y_soh, soh_pred))),
        "rul_mae_cycles": float(mean_absolute_error(y_rul, rul_pred)),
        "rul_rmse_cycles": float(np.sqrt(mean_squared_error(y_rul, rul_pred))),
    }

    prediction_df = test_df[
        ["battery_id", "cycle_index", "cycle_number", "capacity_ah", "soh", "rul_cycles"]
    ].copy()
    prediction_df["soh_pred"] = soh_pred
    prediction_df["rul_pred_cycles"] = rul_pred
    prediction_df["anomaly_score"] = anomaly_raw
    prediction_df["risk_level"] = anomaly_to_risk_level(anomaly_raw)

    return metrics, prediction_df


def anomaly_to_risk_level(scores: np.ndarray) -> np.ndarray:
    q25 = float(np.quantile(scores, 0.25))
    q75 = float(np.quantile(scores, 0.75))

    out = np.full(scores.shape, "medium", dtype=object)
    out[scores <= q25] = "high"
    out[scores >= q75] = "low"
    return out
