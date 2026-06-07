"""Generate AAS-compatible prediction records for the revised framework."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline

from run_baseline_experiments import FEATURES, RANDOM_STATE, build_preprocessor, predict_scores


FAILURE_LABELS = ["TWF", "HDF", "PWF", "OSF", "RNF"]


def risk_status(probability: float, threshold: float) -> str:
    if probability >= threshold:
        return "AtRisk"
    if probability >= threshold * 0.5:
        return "Warning"
    return "Safe"


def aas_record(row: pd.Series, probability: float, threshold: float, predicted_failure: bool) -> dict:
    return {
        "assetAdministrationShell": {
            "idShort": f"AI4I_Machine_{int(row['UDI']):05d}",
            "assetInformation": {
                "assetKind": "Instance",
                "globalAssetId": str(row["Product ID"]),
            },
            "submodels": {
                "AssetIdentification": {
                    "productType": row["Type"],
                    "dataset": "AI4I 2020 synthetic predictive maintenance benchmark",
                },
                "OperationalData": {
                    "airTemperatureK": float(row["Air temperature [K]"]),
                    "processTemperatureK": float(row["Process temperature [K]"]),
                    "rotationalSpeedRpm": float(row["Rotational speed [rpm]"]),
                    "torqueNm": float(row["Torque [Nm]"]),
                    "toolWearMin": float(row["Tool wear [min]"]),
                },
                "FailureState": {
                    "machineFailure": int(row["Machine failure"]),
                    "toolWearFailure": int(row["TWF"]),
                    "heatDissipationFailure": int(row["HDF"]),
                    "powerFailure": int(row["PWF"]),
                    "overstrainFailure": int(row["OSF"]),
                    "randomFailure": int(row["RNF"]),
                },
                "PredictionResult": {
                    "target": "Machine failure",
                    "model": "hist_gradient_weighted",
                    "failureProbability": round(float(probability), 6),
                    "decisionThreshold": round(float(threshold), 6),
                    "predictedFailure": bool(predicted_failure),
                    "riskStatus": risk_status(float(probability), float(threshold)),
                },
            },
        }
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/ai4i2020 (4).csv")
    parser.add_argument("--best-models", default="results/best_models_by_target.csv")
    parser.add_argument("--out", default="aas/aas_prediction_samples.json")
    parser.add_argument("--sample-size", type=int, default=25)
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    best = pd.read_csv(args.best_models)
    threshold = float(best.loc[best["target"] == "Machine failure", "threshold"].iloc[0])

    x = df[FEATURES]
    y = df["Machine failure"].astype(int)
    model = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                HistGradientBoostingClassifier(
                    learning_rate=0.05,
                    max_iter=120,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    model.fit(x, y)
    probabilities = predict_scores(model, x)
    predictions = probabilities >= threshold

    ranked = df.assign(_probability=probabilities, _prediction=predictions)
    examples = pd.concat(
        [
            ranked[ranked["Machine failure"] == 1].sort_values("_probability", ascending=False).head(args.sample_size // 2),
            ranked[ranked["Machine failure"] == 0].sort_values("_probability", ascending=False).head(args.sample_size - args.sample_size // 2),
        ],
        ignore_index=True,
    )

    records = [
        aas_record(row, row["_probability"], threshold, bool(row["_prediction"]))
        for _, row in examples.iterrows()
    ]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(records, indent=2), encoding="utf-8")

    mapping = pd.DataFrame(
        [
            ("Type", "AssetIdentification", "productType"),
            ("Air temperature [K]", "OperationalData", "airTemperatureK"),
            ("Process temperature [K]", "OperationalData", "processTemperatureK"),
            ("Rotational speed [rpm]", "OperationalData", "rotationalSpeedRpm"),
            ("Torque [Nm]", "OperationalData", "torqueNm"),
            ("Tool wear [min]", "OperationalData", "toolWearMin"),
            ("Machine failure", "FailureState", "machineFailure"),
            ("TWF", "FailureState", "toolWearFailure"),
            ("HDF", "FailureState", "heatDissipationFailure"),
            ("PWF", "FailureState", "powerFailure"),
            ("OSF", "FailureState", "overstrainFailure"),
            ("RNF", "FailureState", "randomFailure"),
            ("model probability", "PredictionResult", "failureProbability"),
            ("validation threshold", "PredictionResult", "decisionThreshold"),
            ("predicted label", "PredictionResult", "predictedFailure"),
            ("risk class", "PredictionResult", "riskStatus"),
        ],
        columns=["source_field", "aas_submodel", "aas_property"],
    )
    mapping.to_csv(out_path.parent / "aas_mapping_table.csv", index=False)

    print(f"Wrote {len(records)} AAS prediction examples to {out_path.resolve()}")
    print(f"Wrote mapping table to {(out_path.parent / 'aas_mapping_table.csv').resolve()}")


if __name__ == "__main__":
    main()
