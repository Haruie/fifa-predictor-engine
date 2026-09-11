"""Generate the report figures (outputs/figures/) from the pipeline results.

Run with: python -m src.make_figures
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve

from src.config import load_config, resolve_path
from src.pipeline import build_match_dataset, build_team_profiles, run_seed

SEEDS = [0, 1, 17, 42, 123]
sns.set_theme(style="whitegrid")


def plot_accuracy_comparison(report, out_dir):
    ax = report[["Overall Accuracy", "Accuracy (High-scoring)", "Accuracy (Low-scoring)"]].plot(
        kind="bar", figsize=(7, 5), rot=0, ylim=(0, 1),
    )
    ax.set_title("Ensemble vs. Baseline Accuracy")
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("")
    ax.legend(loc="lower right")
    ax.figure.tight_layout()
    ax.figure.savefig(out_dir / "accuracy_comparison.png", dpi=150)
    plt.close(ax.figure)


def plot_seed_variance(reports_by_seed, out_dir):
    fig, ax = plt.subplots(figsize=(7, 5))
    positions = range(len(SEEDS))
    for label in ("Proposed Method", "Baseline Model"):
        vals = [reports_by_seed[s].loc[label, "Overall Accuracy"] for s in SEEDS]
        ax.plot(positions, vals, marker="o", label=f"{label} (mean={np.mean(vals):.3f}, std={np.std(vals):.3f})")
    ax.set_xlabel("Random seed")
    ax.set_ylabel("Overall accuracy")
    ax.set_title(f"Accuracy Across {len(SEEDS)} Seeds")
    ax.set_xticks(list(positions), [str(s) for s in SEEDS])
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "seed_variance.png", dpi=150)
    plt.close(fig)


def plot_confusion_matrices(report, out_dir):
    y_test, ml_pred, baseline_pred = report.attrs["y_test"], report.attrs["ml_pred"], report.attrs["baseline_pred"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, pred, title in [(axes[0], ml_pred, "Proposed Method"), (axes[1], baseline_pred, "Baseline Model")]:
        cm = confusion_matrix(y_test, pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Away win", "Home win"], yticklabels=["Away win", "Home win"])
        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)


def plot_pca_variance(report, out_dir):
    ratios = report.attrs["pca_explained_variance"]
    if ratios is None:
        return
    cumulative = np.cumsum(ratios)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(range(1, len(cumulative) + 1), cumulative, marker="o")
    ax.axhline(0.95, color="gray", linestyle="--", linewidth=1, label="95% threshold")
    ax.set_xlabel("Number of PCA components")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_title("PCA Variance Retained")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "pca_variance.png", dpi=150)
    plt.close(fig)


def plot_roc_curve(report, out_dir):
    y_test, ml_proba = report.attrs["y_test"], report.attrs["ml_proba"]
    fpr, tpr, _ = roc_curve(y_test, ml_proba)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, label=f"Ensemble (AUC={report.attrs['ensemble_auc']:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC Curve (Ensemble, Soft-Vote Probability)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / "roc_curve.png", dpi=150)
    plt.close(fig)


def plot_feature_importance(report, out_dir):
    importance = report.attrs["feature_importance"]
    if importance is None:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    importance.sort_values().plot(kind="barh", ax=ax, color="#4C72B0")
    ax.set_xlabel("Relative importance (RF + XGBoost, mapped through PCA)")
    ax.set_title("Top Features Driving Predictions")
    fig.tight_layout()
    fig.savefig(out_dir / "feature_importance.png", dpi=150)
    plt.close(fig)


def plot_per_model_accuracy(report, out_dir):
    per_model = report.attrs["per_model_accuracy"].copy()
    per_model["ensemble (majority vote)"] = report.loc["Proposed Method", "Overall Accuracy"]
    per_model["baseline (WWR)"] = report.loc["Baseline Model", "Overall Accuracy"]

    fig, ax = plt.subplots(figsize=(8, 5))
    per_model.sort_values().plot(kind="barh", ax=ax, color="#55A868")
    ax.set_xlabel("Overall accuracy")
    ax.set_xlim(0, 1)
    ax.set_title("Accuracy: Individual Models vs. Ensemble vs. Baseline")
    fig.tight_layout()
    fig.savefig(out_dir / "per_model_accuracy.png", dpi=150)
    plt.close(fig)


def main() -> None:
    cfg = load_config()
    out_dir = resolve_path(cfg["evaluation"]["output_dir"]) / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    profiles = build_team_profiles(cfg)
    feat, wc_all_history, _history_state = build_match_dataset(cfg, profiles)

    reports_by_seed = {s: run_seed(cfg, feat, wc_all_history, s) for s in SEEDS}
    default_report = reports_by_seed[cfg["project"]["random_state"]]

    plot_accuracy_comparison(default_report, out_dir)
    plot_seed_variance(reports_by_seed, out_dir)
    plot_confusion_matrices(default_report, out_dir)
    plot_pca_variance(default_report, out_dir)
    plot_roc_curve(default_report, out_dir)
    plot_feature_importance(default_report, out_dir)
    plot_per_model_accuracy(default_report, out_dir)

    print(f"\nFigures saved to {out_dir}")


if __name__ == "__main__":
    main()
