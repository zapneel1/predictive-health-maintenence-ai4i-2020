"""Second-dataset validation on the UCI SECOM manufacturing dataset.

SECOM is a real semiconductor manufacturing process dataset with hundreds of
sensor/process variables and an imbalanced pass/fail label. It is useful as an
external validation dataset because it is not synthetic AI4I data and it keeps
the task in the manufacturing fault-detection family.
"""

from __future__ import annotations

import argparse
import json
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
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
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
SECOM_URL = "https://archive.ics.uci.edu/static/public/179/secom.zip"


def download_secom(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / "secom.zip"
    if not zip_path.exists():
        print(f"Downloading SECOM from {SECOM_URL}")
        urllib.request.urlretrieve(SECOM_URL, zip_path)

    expected = [data_dir / "secom.data", data_dir / "secom_labels.data"]
    if all(path.exists() for path in expected):
        return

    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(data_dir)


def load_secom(data_dir: Path) -> tuple[pd.DataFrame, pd.Series]:
    download_secom(data_dir)
    x = pd.read_csv(data_dir / "secom.data", sep=r"\s+", header=None, na_values=["NaN"])
    labels = pd.read_csv(data_dir / "secom_labels.data", sep=r"\s+", header=None, usecols=[0])
    y = (labels.iloc[:, 0] == 1).astype(int)
    x.columns = [f"sensor_{idx:03d}" for idx in range(x.shape[1])]
    return x, y


def balanced_sample_weight(y: pd.Series) -> np.ndarray:
    counts = y.value_counts().to_dict()
    total = len(y)
    classes = len(counts)
    return y.map(lambda value: total / (classes * counts[value])).to_numpy()


def preprocessing_steps() -> list[tuple[str, object]]:
    return [
        ("imputer", SimpleImputer(strategy="median")),
        ("variance", VarianceThreshold()),
        ("scaler", StandardScaler()),
    ]


def model_specs() -> dict[str, tuple[object, bool, bool]]:
    return {
        "dummy_most_frequent": (DummyClassifier(strategy="most_frequent"), False, False),
        "logistic_balanced": (
            LogisticRegression(
                class_weight="balanced",
                max_iter=4000,
                solver="liblinear",
                random_state=RANDOM_STATE,
            ),
            False,
            False,
        ),
        "logistic_smote": (
            LogisticRegression(
                max_iter=4000,
                solver="liblinear",
                random_state=RANDOM_STATE,
            ),
            False,
            True,
        ),
        "random_forest_balanced": (
            RandomForestClassifier(
                class_weight="balanced_subsample",
                n_estimators=200,
                min_samples_leaf=3,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            False,
            False,
        ),
        "random_forest_smote": (
            RandomForestClassifier(
                n_estimators=200,
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
                max_iter=160,
                random_state=RANDOM_STATE,
            ),
            True,
            False,
        ),
    }


def predict_scores(model: Pipeline, x_test: pd.DataFrame) -> np.ndarray:
    classifier = model.named_steps["classifier"]
    if hasattr(classifier, "predict_proba"):
        probabilities = model.predict_proba(x_test)
        if probabilities.shape[1] == 1:
            return np.zeros(len(x_test))
        return probabilities[:, 1]
    if hasattr(classifier, "decision_function"):
        return np.asarray(model.decision_function(x_test))
    return model.predict(x_test)


def tune_threshold(y_true: pd.Series, scores: np.ndarray, beta: float = 2.0) -> float:
    thresholds = np.unique(np.quantile(scores, np.linspace(0.01, 0.99, 99)))
    if len(thresholds) == 0:
        return 0.5

    beta_squared = beta**2
    best_threshold = float(thresholds[0])
    best_score = -1.0
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


def evaluate(y_true: pd.Series, scores: np.ndarray, threshold: float, model_name: str) -> dict:
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "dataset": "SECOM",
        "target": "manufacturing_fail",
        "model": model_name,
        "positives": int(y_true.sum()),
        "support": int(len(y_true)),
        "threshold": threshold,
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, predictions),
        "pr_auc": average_precision_score(y_true, scores),
        "roc_auc": roc_auc_score(y_true, scores),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/secom")
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    x, y = load_secom(data_dir)
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
        steps = preprocessing_steps()
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
        threshold = tune_threshold(y_val, val_scores)
        test_scores = predict_scores(model, x_test)
        rows.append(evaluate(y_test, test_scores, 0.5, f"{model_name}@0.50"))
        rows.append(evaluate(y_test, test_scores, threshold, f"{model_name}@tuned_f2"))

    metrics = pd.DataFrame(rows).sort_values(["f1", "recall", "pr_auc"], ascending=False)
    metrics.to_csv(out_dir / "secom_baseline_metrics.csv", index=False)
    metrics.head(1).to_csv(out_dir / "secom_best_model.csv", index=False)

    summary = {
        "dataset": "UCI SECOM",
        "rows": int(len(x)),
        "features": int(x.shape[1]),
        "positive_failures": int(y.sum()),
        "positive_rate": float(y.mean()),
        "outputs": {
            "metrics": str(out_dir / "secom_baseline_metrics.csv"),
            "best_model": str(out_dir / "secom_best_model.csv"),
        },
    }
    (out_dir / "secom_experiment_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nSECOM distribution:")
    print(f"rows={len(x)} features={x.shape[1]} failures={int(y.sum())} positive_rate={y.mean():.4f}")
    print("\nBest SECOM model:")
    columns = ["model", "precision", "recall", "f1", "pr_auc", "roc_auc", "tp", "fp", "fn", "tn"]
    print(metrics.head(1)[columns].to_string(index=False))
    print(f"\nWrote SECOM results to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
