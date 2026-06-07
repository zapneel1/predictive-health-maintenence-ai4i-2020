"""Reviewer-oriented baseline and imbalance experiments for AI4I 2020.

This script is intentionally separate from the exploratory notebook. It creates
reproducible tables for the paper: classical baselines, imbalance-aware models,
threshold tuning, and metrics that are meaningful for rare failures.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


RANDOM_STATE = 42
TARGETS = ["Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"]
FEATURES = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
NUMERIC_FEATURES = [feature for feature in FEATURES if feature != "Type"]
CATEGORICAL_FEATURES = ["Type"]


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ]
    )


def balanced_sample_weight(y: pd.Series) -> np.ndarray:
    counts = y.value_counts().to_dict()
    total = len(y)
    classes = len(counts)
    return y.map(lambda value: total / (classes * counts[value])).to_numpy()


def predict_scores(model: Pipeline, x_test: pd.DataFrame) -> np.ndarray:
    classifier = model.named_steps["classifier"]
    if hasattr(classifier, "predict_proba"):
        probabilities = model.predict_proba(x_test)
        if probabilities.shape[1] == 1:
            return np.zeros(len(x_test))
        return probabilities[:, 1]
    if hasattr(classifier, "decision_function"):
        scores = model.decision_function(x_test)
        return np.asarray(scores)
    return model.predict(x_test)


def tune_threshold(y_true: pd.Series, scores: np.ndarray, beta: float = 2.0) -> float:
    thresholds = np.unique(np.quantile(scores, np.linspace(0.01, 0.99, 99)))
    if len(thresholds) == 0:
        return 0.5

    best_threshold = float(thresholds[0])
    best_score = -1.0
    beta_squared = beta**2
    for threshold in thresholds:
        predictions = (scores >= threshold).astype(int)
        precision = precision_score(y_true, predictions, zero_division=0)
        recall = recall_score(y_true, predictions, zero_division=0)
        denominator = beta_squared * precision + recall
        score = 0.0 if denominator == 0 else (1 + beta_squared) * precision * recall / denominator
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def safe_auc(metric: Callable, y_true: pd.Series, scores: np.ndarray) -> float:
    if y_true.nunique() < 2:
        return float("nan")
    return float(metric(y_true, scores))


def evaluate_predictions(
    y_true: pd.Series,
    scores: np.ndarray,
    threshold: float,
    target: str,
    model_name: str,
) -> dict:
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "target": target,
        "model": model_name,
        "positives": int(y_true.sum()),
        "support": int(len(y_true)),
        "threshold": threshold,
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, predictions),
        "pr_auc": safe_auc(average_precision_score, y_true, scores),
        "roc_auc": safe_auc(roc_auc_score, y_true, scores),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def model_specs() -> dict[str, tuple[object, bool, bool]]:
    return {
        "dummy_most_frequent": (DummyClassifier(strategy="most_frequent"), False, False),
        "logistic_balanced": (
            LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                random_state=RANDOM_STATE,
            ),
            False,
            False,
        ),
        "logistic_smote": (
            LogisticRegression(
                max_iter=2000,
                random_state=RANDOM_STATE,
            ),
            False,
            True,
        ),
        "decision_tree_balanced": (
            DecisionTreeClassifier(
                class_weight="balanced",
                min_samples_leaf=10,
                random_state=RANDOM_STATE,
            ),
            False,
            False,
        ),
        "random_forest_balanced": (
            RandomForestClassifier(
                class_weight="balanced_subsample",
                n_estimators=120,
                min_samples_leaf=3,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            False,
            False,
        ),
        "random_forest_smote": (
            RandomForestClassifier(
                n_estimators=120,
                min_samples_leaf=3,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            False,
            True,
        ),
        "hist_gradient_weighted": (
            HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=120,
                random_state=RANDOM_STATE,
            ),
            True,
            False,
        ),
    }


def run_for_target(df: pd.DataFrame, target: str) -> list[dict]:
    x = df[FEATURES]
    y = df[target].astype(int)

    x_train, x_temp, y_train, y_temp = train_test_split(
        x,
        y,
        test_size=0.4,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=0.5,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )

    rows = []
    for model_name, (classifier, uses_sample_weight, uses_smote) in model_specs().items():
        steps = [("preprocessor", build_preprocessor())]
        if uses_smote:
            minority_count = int(y_train.sum())
            k_neighbors = max(1, min(5, minority_count - 1))
            steps.append(("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=k_neighbors)))
        steps.append(("classifier", classifier))
        pipeline_cls = ImbPipeline if uses_smote else Pipeline
        model = pipeline_cls(steps=steps)

        if uses_sample_weight:
            model.fit(x_train, y_train, classifier__sample_weight=balanced_sample_weight(y_train))
        else:
            model.fit(x_train, y_train)

        val_scores = predict_scores(model, x_val)
        tuned_threshold = tune_threshold(y_val, val_scores)
        test_scores = predict_scores(model, x_test)

        rows.append(evaluate_predictions(y_test, test_scores, 0.5, target, f"{model_name}@0.50"))
        rows.append(
            evaluate_predictions(
                y_test,
                test_scores,
                tuned_threshold,
                target,
                f"{model_name}@tuned_f2",
            )
        )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/ai4i2020 (4).csv")
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    data_path = Path(args.data)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    missing_columns = sorted(set(FEATURES + TARGETS) - set(df.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing expected columns: {missing_columns}")

    class_counts = df[TARGETS].sum().rename("positive_count").reset_index()
    class_counts.columns = ["target", "positive_count"]
    class_counts["total"] = len(df)
    class_counts["positive_rate"] = class_counts["positive_count"] / len(df)

    rows = []
    for target in TARGETS:
        rows.extend(run_for_target(df, target))

    metrics = pd.DataFrame(rows)
    metrics = metrics.sort_values(["target", "pr_auc", "recall", "f1"], ascending=[True, False, False, False])

    class_counts.to_csv(out_dir / "class_distribution.csv", index=False)
    metrics.to_csv(out_dir / "baseline_metrics.csv", index=False)

    best_by_target = (
        metrics.sort_values(["target", "f1", "recall", "pr_auc"], ascending=[True, False, False, False])
        .groupby("target", as_index=False)
        .head(1)
    )
    best_by_target.to_csv(out_dir / "best_models_by_target.csv", index=False)

    summary = {
        "data": str(data_path),
        "rows": int(len(df)),
        "features": FEATURES,
        "targets": TARGETS,
        "outputs": {
            "class_distribution": str(out_dir / "class_distribution.csv"),
            "baseline_metrics": str(out_dir / "baseline_metrics.csv"),
            "best_models_by_target": str(out_dir / "best_models_by_target.csv"),
        },
    }
    (out_dir / "experiment_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nClass distribution:")
    print(class_counts.to_string(index=False))
    print("\nBest model per target:")
    columns = ["target", "model", "precision", "recall", "f1", "pr_auc", "roc_auc", "tp", "fp", "fn", "tn"]
    print(best_by_target[columns].to_string(index=False))
    print(f"\nWrote results to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
