from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from .data_loader import filter_discharge_cycles, load_nasa_battery_directory
from .features import DEFAULT_EOL_SOH_THRESHOLD, build_training_frame, split_by_battery
from .models import evaluate_models, train_models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train EV battery SOH/RUL baseline models.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Path to .mat files")
    parser.add_argument(
        "--artifact-dir", type=Path, default=Path("artifacts"), help="Output dir for models/reports"
    )
    parser.add_argument(
        "--test-battery",
        type=str,
        default="B0007",
        help="Battery ID used as holdout test set (battery-level split).",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--eol-soh-threshold",
        type=float,
        default=DEFAULT_EOL_SOH_THRESHOLD,
        help="SOH threshold used to define end-of-life (EOL) and RUL target.",
    )
    return parser.parse_args()


def run_training(
    data_dir: Path,
    artifact_dir: Path,
    test_battery: str = "B0007",
    random_state: int = 42,
    eol_soh_threshold: float = DEFAULT_EOL_SOH_THRESHOLD,
) -> tuple[dict[str, float], pd.DataFrame]:
    artifact_dir.mkdir(parents=True, exist_ok=True)

    raw_df = load_nasa_battery_directory(data_dir)
    discharge_df = filter_discharge_cycles(raw_df)
    frame = build_training_frame(discharge_df, eol_soh_threshold=eol_soh_threshold)
    train_df, test_df = split_by_battery(frame, test_batteries=[test_battery])

    bundle = train_models(train_df, random_state=random_state)
    metrics, predictions = evaluate_models(bundle, test_df)

    joblib.dump(bundle, artifact_dir / "battery_model_bundle.joblib")
    predictions.to_csv(artifact_dir / "test_predictions.csv", index=False)
    with open(artifact_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(artifact_dir / "model_config.json", "w", encoding="utf-8") as f:
        json.dump({"eol_soh_threshold": eol_soh_threshold}, f, indent=2)

    return metrics, predictions


def main() -> None:
    args = parse_args()
    metrics, _ = run_training(
        data_dir=args.data_dir,
        artifact_dir=args.artifact_dir,
        test_battery=args.test_battery,
        random_state=args.random_state,
        eol_soh_threshold=args.eol_soh_threshold,
    )

    print("Training complete.")
    print(f"Holdout battery: {args.test_battery}")
    print(f"EOL SOH threshold: {args.eol_soh_threshold:.2f}")
    print(f"Artifacts: {args.artifact_dir.resolve()}")
    print("Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.6f}")


if __name__ == "__main__":
    main()
