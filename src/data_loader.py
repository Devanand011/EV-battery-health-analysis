from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import scipy.io


def _safe_mean(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.mean(values))


def _safe_max(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.max(values))


def _safe_min(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.min(values))


def _safe_std(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.std(values))


def _field_to_array(data, field_name: str) -> np.ndarray:
    if field_name not in data.dtype.names:
        return np.array([], dtype=float)
    return np.asarray(data[field_name]).ravel().astype(float)


def _extract_cycle_metrics(
    battery_id: str, cycle_idx: int, cycle_type: str, cycle_struct
) -> dict:
    data = cycle_struct["data"][0][0]
    time_values = _field_to_array(data, "Time")
    v_values = _field_to_array(data, "Voltage_measured")
    i_values = _field_to_array(data, "Current_measured")
    t_values = _field_to_array(data, "Temperature_measured")

    duration_sec = (
        float(time_values[-1] - time_values[0]) if time_values.size > 1 else float("nan")
    )
    dt = np.diff(time_values) if time_values.size > 1 else np.array([], dtype=float)
    current_mid = i_values[:-1] if i_values.size > 1 else np.array([], dtype=float)
    throughput_ah = (
        float(np.sum(np.abs(current_mid) * dt) / 3600.0) if dt.size else float("nan")
    )

    row = {
        "battery_id": battery_id,
        "cycle_index": int(cycle_idx),
        "cycle_type": cycle_type,
        "ambient_temperature_c": float(cycle_struct["ambient_temperature"][0][0]),
        "duration_sec": duration_sec,
        "voltage_mean_v": _safe_mean(v_values),
        "voltage_min_v": _safe_min(v_values),
        "voltage_max_v": _safe_max(v_values),
        "current_mean_a": _safe_mean(i_values),
        "current_abs_mean_a": _safe_mean(np.abs(i_values)),
        "temperature_mean_c": _safe_mean(t_values),
        "temperature_max_c": _safe_max(t_values),
        "temperature_std_c": _safe_std(t_values),
        "charge_throughput_ah": throughput_ah,
    }

    if cycle_type == "discharge" and "Capacity" in data.dtype.names:
        row["capacity_ah"] = float(data["Capacity"][0][0])
    else:
        row["capacity_ah"] = float("nan")

    return row


def load_nasa_battery_directory(data_dir: str | Path) -> pd.DataFrame:
    data_dir = Path(data_dir)
    rows: list[dict] = []

    for mat_file in sorted(data_dir.glob("B*.mat")):
        battery_id = mat_file.stem
        mat_data = scipy.io.loadmat(mat_file.as_posix())
        battery = mat_data[battery_id][0][0]
        cycles = battery["cycle"][0]

        for cycle_idx, cycle in enumerate(cycles, start=1):
            cycle_type = str(cycle["type"][0])
            rows.append(_extract_cycle_metrics(battery_id, cycle_idx, cycle_type, cycle))

    if not rows:
        raise FileNotFoundError(f"No NASA battery .mat files found in: {data_dir}")

    df = pd.DataFrame(rows).sort_values(["battery_id", "cycle_index"]).reset_index(drop=True)
    return df


def load_cycle_csv(csv_path: str | Path) -> pd.DataFrame:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    return pd.read_csv(csv_path)


def filter_discharge_cycles(df: pd.DataFrame) -> pd.DataFrame:
    out = df.loc[df["cycle_type"] == "discharge"].copy()
    out = out.sort_values(["battery_id", "cycle_index"]).reset_index(drop=True)
    return out


def list_batteries(df: pd.DataFrame) -> Iterable[str]:
    return sorted(df["battery_id"].dropna().unique().tolist())
