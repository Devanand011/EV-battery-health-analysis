from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from .data_loader import filter_discharge_cycles, load_cycle_csv, load_nasa_battery_directory
from .features import DEFAULT_EOL_SOH_THRESHOLD, build_training_frame
from .models import anomaly_to_risk_level


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SOH/RUL inference.")
    parser.add_argument(
        "--artifact-path",
        type=Path,
        default=Path("artifacts/battery_model_bundle.joblib"),
        help="Path to trained model bundle.",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=None,
        help="CSV of cycle-level data compatible with training frame columns.",
    )
    parser.add_argument(
        "--input-data-dir",
        type=Path,
        default=None,
        help="Directory containing NASA .mat files. Used if --input-csv is not provided.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("artifacts/inference_predictions.csv"),
        help="Where to save predictions.",
    )
    parser.add_argument(
        "--eol-soh-threshold",
        type=float,
        default=DEFAULT_EOL_SOH_THRESHOLD,
        help="SOH threshold used to derive EOL-based RUL when building frame from raw data.",
    )
    return parser.parse_args()


def _load_input(args: argparse.Namespace) -> pd.DataFrame:
    if args.input_csv is not None:
        df = load_cycle_csv(args.input_csv)
        if "soh" not in df.columns or "rul_cycles" not in df.columns:
            raise ValueError(
                "Input CSV must already contain engineered frame columns including soh and rul_cycles. "
                "For raw NASA data, use --input-data-dir."
            )
        return df

    data_dir = args.input_data_dir or Path("data")
    raw_df = load_nasa_battery_directory(data_dir)
    discharge_df = filter_discharge_cycles(raw_df)
    return build_training_frame(discharge_df, eol_soh_threshold=args.eol_soh_threshold)


def main() -> None:
    args = parse_args()
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    bundle = joblib.load(args.artifact_path)
    df = _load_input(args)
    x = df[bundle.feature_columns].copy()

    soh_pred = bundle.soh_model.predict(x)
    rul_pred = bundle.rul_model.predict(x).clip(min=0.0)
    anomaly_score = bundle.anomaly_model.decision_function(x)
    risk = anomaly_to_risk_level(anomaly_score)

    out = df[
        ["battery_id", "cycle_index", "cycle_number", "capacity_ah", "soh", "rul_cycles"]
    ].copy()
    out["soh_pred"] = soh_pred
    out["rul_pred_cycles"] = rul_pred
    out["anomaly_score"] = anomaly_score
    out["risk_level"] = risk

    out.to_csv(args.output_csv, index=False)
    print(f"Wrote predictions to: {args.output_csv.resolve()}")


if __name__ == "__main__":
    main()
