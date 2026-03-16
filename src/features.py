from __future__ import annotations

import numpy as np
import pandas as pd


SOH_REFERENCE_MODE = "first_discharge_capacity"
DEFAULT_EOL_SOH_THRESHOLD = 0.80


FEATURE_COLUMNS = [
    "cycle_number",
    "ambient_temperature_c",
    "duration_sec",
    "voltage_mean_v",
    "voltage_min_v",
    "voltage_max_v",
    "current_mean_a",
    "current_abs_mean_a",
    "temperature_mean_c",
    "temperature_max_c",
    "temperature_std_c",
    "charge_throughput_ah",
    "capacity_roll3_mean_ah",
    "capacity_roll5_mean_ah",
    "capacity_drop_from_start_ah",
    "capacity_fade_rate_roll5",
]


def build_training_frame(
    discharge_df: pd.DataFrame, eol_soh_threshold: float = DEFAULT_EOL_SOH_THRESHOLD
) -> pd.DataFrame:
    df = discharge_df.copy()
    df = df.sort_values(["battery_id", "cycle_index"]).reset_index(drop=True)
    df["cycle_number"] = df.groupby("battery_id").cumcount() + 1

    baseline_capacity = df.groupby("battery_id")["capacity_ah"].transform("first")
    df["soh"] = df["capacity_ah"] / baseline_capacity
    df["capacity_drop_from_start_ah"] = baseline_capacity - df["capacity_ah"]

    df["capacity_roll3_mean_ah"] = (
        df.groupby("battery_id")["capacity_ah"]
        .transform(lambda s: s.rolling(3, min_periods=1).mean())
        .astype(float)
    )
    df["capacity_roll5_mean_ah"] = (
        df.groupby("battery_id")["capacity_ah"]
        .transform(lambda s: s.rolling(5, min_periods=1).mean())
        .astype(float)
    )
    df["capacity_fade_rate_roll5"] = (
        df.groupby("battery_id")["capacity_ah"]
        .transform(lambda s: s.diff().rolling(5, min_periods=2).mean())
        .fillna(0.0)
    )

    def _battery_rul(battery_id: str, group: pd.DataFrame) -> pd.DataFrame:
        g = group.copy()
        g["battery_id"] = battery_id
        eol_candidates = g.loc[g["soh"] <= eol_soh_threshold, "cycle_number"]
        eol_cycle = (
            int(eol_candidates.iloc[0]) if not eol_candidates.empty else int(g["cycle_number"].max())
        )
        g["eol_cycle_number"] = eol_cycle
        g["rul_cycles"] = (eol_cycle - g["cycle_number"]).clip(lower=0).astype(float)
        g["is_eol_or_beyond"] = (g["cycle_number"] >= eol_cycle).astype(int)
        return g

    parts = []
    for battery_id, group in df.groupby("battery_id", sort=False):
        parts.append(_battery_rul(str(battery_id), group))
    df = pd.concat(parts, ignore_index=True)

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=FEATURE_COLUMNS + ["soh", "rul_cycles"]).reset_index(drop=True)
    return df


def split_by_battery(
    df: pd.DataFrame, test_batteries: list[str] | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_batteries = sorted(df["battery_id"].unique().tolist())
    if not all_batteries:
        raise ValueError("No batteries in training frame.")

    if test_batteries is None:
        test_batteries = [all_batteries[-1]]

    train_df = df.loc[~df["battery_id"].isin(test_batteries)].copy()
    test_df = df.loc[df["battery_id"].isin(test_batteries)].copy()

    if train_df.empty or test_df.empty:
        raise ValueError(
            f"Invalid split. train rows={len(train_df)} test rows={len(test_df)} "
            f"test_batteries={test_batteries}"
        )

    return train_df, test_df
