"""Generate manuscript-ready artifacts from the revised experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from run_baseline_experiments import (
    FEATURES,
    RANDOM_STATE,
    build_preprocessor,
    predict_scores,
)


def plot_class_distribution(class_distribution: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=class_distribution, x="target", y="positive_rate", ax=ax, color="#4C78A8")
    ax.set_title("Positive Failure Rate by Target")
    ax.set_xlabel("Target")
    ax.set_ylabel("Positive rate")
    ax.bar_label(ax.containers[0], fmt="%.3f", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "class_distribution.png", dpi=220)
    plt.close(fig)


def plot_best_model_confusions(best: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    axes = axes.flatten()

    for ax, (_, row) in zip(axes, best.iterrows()):
        matrix = [[row["tn"], row["fp"]], [row["fn"], row["tp"]]]
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".0f",
            cmap="Blues",
            cbar=False,
            xticklabels=["Pred 0", "Pred 1"],
            yticklabels=["True 0", "True 1"],
            ax=ax,
        )
        ax.set_title(f"{row['target']}: {row['model'].split('@')[0]}", fontsize=9)
        ax.set_xlabel("")
        ax.set_ylabel("")

    for ax in axes[len(best) :]:
        ax.axis("off")

    fig.suptitle("Confusion Matrices for Best Model per Target", y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "best_model_confusion_matrices.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_metric_comparison(metrics: pd.DataFrame, out_dir: Path) -> None:
    chosen = metrics[metrics["model"].str.contains("dummy|balanced@0.50|smote@0.50|weighted@0.50", regex=True)]
    chosen = chosen[chosen["target"].isin(["Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"])]

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=chosen, x="target", y="recall", hue="model", ax=ax)
    ax.set_title("Recall Comparison Across Baselines")
    ax.set_xlabel("Target")
    ax.set_ylabel("Recall")
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / "baseline_recall_comparison.png", dpi=220)
    plt.close(fig)


def plot_safe_region_3d(df: pd.DataFrame, best: pd.DataFrame, out_dir: Path) -> None:
    threshold = float(best.loc[best["target"] == "Machine failure", "threshold"].iloc[0])
    x = df[FEATURES]
    y = df["Machine failure"].astype(int)
    _, x_test, _, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

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
    scores = predict_scores(model, x_test)
    risk_status = scores >= threshold

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    colors = risk_status.map({False: "#4C78A8", True: "#E45756"}) if hasattr(risk_status, "map") else risk_status
    ax.scatter(
        x_test["Air temperature [K]"],
        x_test["Torque [Nm]"],
        x_test["Tool wear [min]"],
        c=["#E45756" if status else "#4C78A8" for status in risk_status],
        s=12,
        alpha=0.65,
    )
    ax.set_title("3D Safe and At-Risk Operating Region")
    ax.set_xlabel("Air temperature [K]")
    ax.set_ylabel("Torque [Nm]")
    ax.set_zlabel("Tool wear [min]")
    ax.view_init(elev=24, azim=135)

    safe = plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#4C78A8", label="Predicted safe")
    risky = plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#E45756", label="Predicted at-risk")
    ax.legend(handles=[safe, risky], loc="upper left")

    fig.tight_layout()
    fig.savefig(out_dir / "safe_operating_region_3d.png", dpi=220)
    plt.close(fig)


def main() -> None:
    root = Path(".")
    out_dir = root / "results" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(root / "data" / "ai4i2020 (4).csv")
    class_distribution = pd.read_csv(root / "results" / "class_distribution.csv")
    metrics = pd.read_csv(root / "results" / "baseline_metrics.csv")
    best = pd.read_csv(root / "results" / "best_models_by_target.csv")

    plot_class_distribution(class_distribution, out_dir)
    plot_best_model_confusions(best, out_dir)
    plot_metric_comparison(metrics, out_dir)
    plot_safe_region_3d(df, best, out_dir)

    print(f"Wrote figures to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
