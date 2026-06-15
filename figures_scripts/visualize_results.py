import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
    accuracy_score,
)

LABEL_ORDER = ["false", "mixture", "true", "unproven"]
LABEL_COLORS = {
    "false":    "#e15759",
    "mixture":  "#f28e2b",
    "true":     "#4e79a7",
    "unproven": "#76b7b2",
}
FIGSIZE_SQUARE = (7, 6)
FIGSIZE_WIDE   = (10, 5)
FIGSIZE_TALL   = (8, 6)
DPI = 150


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _present_labels(df: pd.DataFrame) -> list[str]:
    """Return label order restricted to labels that actually appear."""
    seen = set(df["label"].astype(str)) | set(df["predicted_label"].astype(str))
    return [l for l in LABEL_ORDER if l in seen]


def _conf_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in [f"prob_{l}" for l in LABEL_ORDER] if c in df.columns]


# ---------------------------------------------------------------------------
# 1. Confusion matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(df: pd.DataFrame, out_dir: Path) -> None:
    labels = _present_labels(df)
    y_true = df["label"].astype(str)
    y_pred = df["predicted_label"].astype(str)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, data, title, fmt in zip(
        axes,
        [cm, cm_norm],
        ["Confusion Matrix (counts)", "Confusion Matrix (row-normalised)"],
        [".0f", ".2f"],
    ):
        cmap = LinearSegmentedColormap.from_list("wblue", ["#ffffff", "#4e79a7"])
        im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=0,
                       vmax=data.max())
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted", labelpad=8)
        ax.set_ylabel("True", labelpad=8)
        ax.set_title(title, pad=10, fontsize=12, fontweight="bold")
        thresh = data.max() / 2.0
        for i in range(len(labels)):
            for j in range(len(labels)):
                color = "white" if data[i, j] > thresh else "black"
                ax.text(j, i, format(data[i, j], fmt),
                        ha="center", va="center", fontsize=10, color=color)

    plt.suptitle("SciBERT — 4-class classification", y=1.02, fontsize=13)
    plt.tight_layout()
    path = out_dir / "confusion_matrix.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


# ---------------------------------------------------------------------------
# 2. Per-class precision / recall / F1
# ---------------------------------------------------------------------------

def plot_per_class_metrics(df: pd.DataFrame, out_dir: Path) -> None:
    labels = _present_labels(df)
    report = classification_report(
        df["label"].astype(str),
        df["predicted_label"].astype(str),
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    metrics = ["precision", "recall", "f1-score"]
    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    palette = ["#4e79a7", "#f28e2b", "#59a14f"]
    for i, m in enumerate(metrics):
        vals = [report[l][m] for l in labels]
        bars = ax.bar(x + i * width, vals, width, label=m.capitalize(),
                      color=palette[i], alpha=0.85, edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x + width)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("Per-class Precision / Recall / F1", fontweight="bold")
    ax.legend(framealpha=0.7)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    macro_f1 = report["macro avg"]["f1-score"]
    acc = accuracy_score(df["label"].astype(str), df["predicted_label"].astype(str))
    ax.text(0.98, 0.97,
            f"Accuracy: {acc:.3f}   Macro F1: {macro_f1:.3f}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.8))

    plt.tight_layout()
    path = out_dir / "per_class_metrics.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


# ---------------------------------------------------------------------------
# 3. Gold vs Predicted label distribution
# ---------------------------------------------------------------------------

def plot_label_distribution(df: pd.DataFrame, out_dir: Path) -> None:
    labels = _present_labels(df)
    gold_counts = df["label"].astype(str).value_counts().reindex(labels, fill_value=0)
    pred_counts = df["predicted_label"].astype(str).value_counts().reindex(labels, fill_value=0)

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    b1 = ax.bar(x - width / 2, gold_counts.values, width, label="Gold",
                color="#4e79a7", alpha=0.85, edgecolor="white")
    b2 = ax.bar(x + width / 2, pred_counts.values, width, label="Predicted",
                color="#f28e2b", alpha=0.85, edgecolor="white")

    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
                    str(int(bar.get_height())), ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Count")
    ax.set_title("Label Distribution: Gold vs Predicted", fontweight="bold")
    ax.legend(framealpha=0.7)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    path = out_dir / "label_distribution.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


# ---------------------------------------------------------------------------
# 4. Confidence distribution (correct vs incorrect)
# ---------------------------------------------------------------------------

def plot_confidence_distribution(df: pd.DataFrame, out_dir: Path) -> None:
    if "confidence" not in df.columns:
        print("  skipping confidence plot (no 'confidence' column)")
        return

    correct = df["label"].astype(str) == df["predicted_label"].astype(str)
    conf_correct = df.loc[correct, "confidence"]
    conf_wrong   = df.loc[~correct, "confidence"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # histogram
    ax = axes[0]
    bins = np.linspace(0, 1, 25)
    ax.hist(conf_correct, bins=bins, alpha=0.7, label=f"Correct (n={correct.sum()})",
            color="#59a14f", edgecolor="white", density=True)
    ax.hist(conf_wrong,   bins=bins, alpha=0.7, label=f"Incorrect (n={(~correct).sum()})",
            color="#e15759", edgecolor="white", density=True)
    ax.set_xlabel("Confidence (max softmax prob)")
    ax.set_ylabel("Density")
    ax.set_title("Confidence Distribution", fontweight="bold")
    ax.legend(framealpha=0.7)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # calibration (reliability diagram)
    ax = axes[1]
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers, accs, counts = [], [], []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (df["confidence"] >= lo) & (df["confidence"] < hi)
        if mask.sum() == 0:
            continue
        bin_centers.append((lo + hi) / 2)
        accs.append(correct[mask].mean())
        counts.append(mask.sum())

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    sc = ax.scatter(bin_centers, accs, c=counts, cmap="YlOrRd", s=80,
                    zorder=5, edgecolors="k", linewidths=0.5)
    ax.plot(bin_centers, accs, "-o", color="#4e79a7", markersize=5, lw=1.5, label="Model")
    plt.colorbar(sc, ax=ax, label="# examples in bin")
    ax.set_xlabel("Mean confidence")
    ax.set_ylabel("Fraction correct")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("Reliability Diagram (Calibration)", fontweight="bold")
    ax.legend(framealpha=0.7)
    ax.grid(linestyle="--", alpha=0.4)

    plt.suptitle("SciBERT confidence analysis", fontsize=13)
    plt.tight_layout()
    path = out_dir / "confidence_distribution.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


# ---------------------------------------------------------------------------
# 5. Per-class confidence boxplot
# ---------------------------------------------------------------------------

def plot_confidence_by_class(df: pd.DataFrame, out_dir: Path) -> None:
    if "confidence" not in df.columns:
        return
    labels = _present_labels(df)
    correct = df["label"].astype(str) == df["predicted_label"].astype(str)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, group_col, title in [
        (axes[0], "predicted_label", "Confidence by Predicted Class"),
        (axes[1], "label",           "Confidence by True Class"),
    ]:
        data_by_label = [
            df.loc[df[group_col].astype(str) == l, "confidence"].values
            for l in labels
        ]
        bp = ax.boxplot(data_by_label, patch_artist=True, notch=False,
                        medianprops=dict(color="black", lw=2))
        for patch, lbl in zip(bp["boxes"], labels):
            patch.set_facecolor(LABEL_COLORS.get(lbl, "#aaa"))
            patch.set_alpha(0.8)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Confidence")
        ax.set_title(title, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.suptitle("SciBERT prediction confidence", fontsize=13)
    plt.tight_layout()
    path = out_dir / "confidence_by_class.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


# ---------------------------------------------------------------------------
# 6. Judge verdict analysis (optional, if judge columns present)
# ---------------------------------------------------------------------------

def plot_judge_analysis(df: pd.DataFrame, out_dir: Path) -> None:
    if "judge_verdict" not in df.columns:
        return
    correct = df["label"].astype(str) == df["predicted_label"].astype(str)
    verdicts = ["SUPPORTED", "PARTIALLY_SUPPORTED", "NOT_SUPPORTED"]
    present = [v for v in verdicts if v in df["judge_verdict"].values]
    if not present:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # verdict distribution
    ax = axes[0]
    counts = df["judge_verdict"].value_counts().reindex(present, fill_value=0)
    colors = ["#59a14f", "#f28e2b", "#e15759"][:len(present)]
    bars = ax.bar(range(len(present)), counts.values, color=colors, alpha=0.85, edgecolor="white")
    for bar, v in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                str(v), ha="center", va="bottom", fontsize=10)
    ax.set_xticks(range(len(present)))
    ax.set_xticklabels([v.replace("_", "\n") for v in present], fontsize=9)
    ax.set_ylabel("Count")
    ax.set_title("Judge Verdict Distribution", fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # SciBERT accuracy per verdict group
    ax = axes[1]
    accs = [correct[df["judge_verdict"] == v].mean() for v in present]
    ns   = [(df["judge_verdict"] == v).sum() for v in present]
    bars = ax.bar(range(len(present)), accs, color=colors, alpha=0.85, edgecolor="white")
    for bar, acc, n in zip(bars, accs, ns):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{acc:.2f}\n(n={n})", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(len(present)))
    ax.set_xticklabels([v.replace("_", "\n") for v in present], fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("SciBERT Accuracy")
    ax.set_title("SciBERT Accuracy by Judge Verdict", fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.axhline(correct.mean(), color="grey", lw=1.5, linestyle="--", label=f"Overall acc={correct.mean():.3f}")
    ax.legend(fontsize=9)

    plt.suptitle("Judge LLM vs SciBERT agreement", fontsize=13)
    plt.tight_layout()
    path = out_dir / "judge_analysis.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Plot SciBERT inference results")
    parser.add_argument("csvs", nargs="+", type=Path, help="Input CSV file(s)")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Output directory (default: data/plots/<csv-stem>/)")
    args = parser.parse_args()

    for csv_path in args.csvs:
        if not csv_path.exists():
            print(f"[WARN] {csv_path} not found, skipping")
            continue

        df = pd.read_csv(csv_path)
        print(f"\n=== {csv_path.name} ({len(df)} rows) ===")

        if "label" not in df.columns or "predicted_label" not in df.columns:
            print("  [WARN] Missing 'label' or 'predicted_label' column — skipping")
            continue

        out_dir = args.out_dir or (csv_path.parent / "plots" / csv_path.stem)
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"  output → {out_dir}/")

        plot_confusion_matrix(df, out_dir)
        plot_per_class_metrics(df, out_dir)
        plot_label_distribution(df, out_dir)
        plot_confidence_distribution(df, out_dir)
        plot_confidence_by_class(df, out_dir)
        plot_judge_analysis(df, out_dir)

        print(f"  done.")


if __name__ == "__main__":
    main()
