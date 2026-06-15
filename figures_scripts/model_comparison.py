import json
import math
import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

# =============================================================================
# CONFIG - hardcode changes here; no command-line args are used.
# =============================================================================

PROJECT_ROOT = Path(".")

# Data files.
ORIGINAL_4CLASS_JSONL = PROJECT_ROOT / "data" / "processed" / "FULL_no_reform.jsonl"
THREE_CLASS_JSONL = PROJECT_ROOT / "data" / "processed" / "FULL_no_reform_3class_unknown.jsonl"

# Models.
FOUR_CLASS_RAG_MODEL_DIR = PROJECT_ROOT / "models" / "scibert_20260605_152114_fa6345e0-e3af-4933-bc23-0dbce6a63b12"
THREE_CLASS_RAG_MODEL_DIR = PROJECT_ROOT / "models" / "scibert_3class_unkown_20260614_170223_808dd1b8-8e8b-49be-8ed8-453b24073287"

# Existing prediction files from previous/separate scripts.
EXISTING_FOUR_CLASS_RAG_PREDICTIONS = PROJECT_ROOT / "EDA" / "predictions" / "test_predictions_compact.csv"
NO_RAG_BASELINE_PREDICTIONS = PROJECT_ROOT / "EDA" / "no_rag_baseline" / "predictions" / "test_predictions_no_rag_baseline_compact.csv"

OUTPUT_DIR = PROJECT_ROOT / "EDA" / "model_comparison"
SPLIT_TO_EVALUATE = "test"
MAX_LENGTH = 512
BATCH_SIZE = 16
USE_GPU_IF_AVAILABLE = True
RANDOM_SEED = 42

# If previous 4-class RAG predictions do not exist, the script can rerun the
# original 4-class RAG model. Set False if you only want to use existing outputs.
RUN_4CLASS_RAG_INFERENCE_IF_MISSING = True

# Always run the new 3-class model unless this is set to False.
RUN_3CLASS_RAG_INFERENCE = True

# Fallback label orders are used only if the HuggingFace model config has generic
# LABEL_0/LABEL_1 labels. If your training code used a different label order,
# change it here.
FOUR_CLASS_LABEL_ORDER = ["false", "mixture", "true", "unproven"]
THREE_CLASS_LABEL_ORDER = ["false", "true", "unknown"]

# Input formatting for RAG models. This matches the earlier EDA/inference script.
RAG_INPUT_STYLE = "formatted_claim_query_evidence"

# =============================================================================
# Paths and output registry
# =============================================================================

TABLE_DIR = OUTPUT_DIR / "tables"
FIG_DIR = OUTPUT_DIR / "figures"
PRED_DIR = OUTPUT_DIR / "predictions"
LOG_DIR = OUTPUT_DIR / "logs"
EXAMPLE_DIR = OUTPUT_DIR / "examples"

OUTPUT_RECORDS: list[dict[str, str]] = []
WARNINGS: list[str] = []


def ensure_dirs() -> None:
    for path in [OUTPUT_DIR, TABLE_DIR, FIG_DIR, PRED_DIR, LOG_DIR, EXAMPLE_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def warn(message: str) -> None:
    WARNINGS.append(message)
    print(f"WARNING: {message}")


def record_output(path: Path, category: str, description: str) -> None:
    OUTPUT_RECORDS.append({
        "category": category,
        "path": str(path),
        "description": description,
    })


def save_table(df: pd.DataFrame, name: str, description: str) -> Path:
    path = TABLE_DIR / f"{name}.csv"
    df.to_csv(path, index=False)
    record_output(path, "table", description)
    print(f"Saved table: {path}")
    return path


def save_figure(fig: plt.Figure, path: Path, description: str) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    record_output(path, "figure", description)
    print(f"Saved figure: {path}")


# =============================================================================
# Loading and formatting helpers
# =============================================================================


def set_seeds(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def read_jsonl(path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Could not parse JSONL line {line_number} in {path}: {exc}") from exc
    df = pd.DataFrame(rows)
    print(f"Loaded {path}: {len(df):,} rows, {len(df.columns):,} columns")
    return df


def normalize_label(label: Any) -> str:
    return str(label).strip().lower().replace(" ", "_")


def collapse_to_unknown(label: Any) -> str:
    label_s = normalize_label(label)
    if label_s in {"mixture", "unproven", "unknown", "unkown"}:
        return "unknown"
    return label_s


def add_row_key(df: pd.DataFrame) -> pd.DataFrame:
    """Create robust keys for joining original and 3-class data."""
    out = df.copy()
    if "split" not in out.columns:
        out["split"] = "unknown_split"
    if "example_id" in out.columns:
        out["join_key"] = out["split"].astype(str) + "::" + out["example_id"].astype(str)
    else:
        out["_row_in_split"] = out.groupby("split").cumcount()
        out["join_key"] = out["split"].astype(str) + "::row_" + out["_row_in_split"].astype(str)
    return out


def build_rag_input(row: dict[str, Any]) -> str:
    claim = str(row.get("claim", "")).strip()
    query = str(row.get("rag_query", row.get("reformulated_query", claim))).strip()
    context = str(row.get("rag_context", "")).strip() or "No evidence retrieved."
    if RAG_INPUT_STYLE == "formatted_claim_query_evidence":
        return (
            f"Original claim:\n{claim}\n\n"
            f"Search query:\n{query}\n\n"
            f"Retrieved evidence:\n{context}"
        )
    if RAG_INPUT_STYLE == "claim_plus_evidence":
        return f"{claim}\n\n{context}"
    raise ValueError(f"Unknown RAG_INPUT_STYLE: {RAG_INPUT_STYLE}")


def is_placeholder_label(label: str) -> bool:
    lower = str(label).strip().lower()
    return lower.startswith("label_") or lower in {"0", "1", "2", "3", "none", "null"}


def get_label_maps(model: Any, fallback_order: list[str]) -> tuple[dict[int, str], dict[str, int]]:
    num_labels = int(getattr(model.config, "num_labels", len(fallback_order)))
    id2label_raw = getattr(model.config, "id2label", None) or {}
    id_to_label: dict[int, str] = {}
    try:
        for key, value in id2label_raw.items():
            id_to_label[int(key)] = normalize_label(value)
    except Exception:
        id_to_label = {}

    if len(id_to_label) != num_labels or any(is_placeholder_label(v) for v in id_to_label.values()):
        if num_labels > len(fallback_order):
            raise ValueError(
                f"Model has {num_labels} labels but fallback order only has {len(fallback_order)} labels: {fallback_order}"
            )
        id_to_label = {i: fallback_order[i] for i in range(num_labels)}

    label_to_id = {label: idx for idx, label in id_to_label.items()}
    return id_to_label, label_to_id


# =============================================================================
# Metrics and plots
# =============================================================================


def metric_row(df: pd.DataFrame, gold_col: str, pred_col: str, labels: list[str], setup: str, label_space: str, notes: str = "") -> dict[str, Any]:
    y_true = df[gold_col].map(normalize_label)
    y_pred = df[pred_col].map(normalize_label)
    return {
        "setup": setup,
        "label_space": label_space,
        "rows": int(len(df)),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0),
        "notes": notes,
    }


def classification_report_df(df: pd.DataFrame, gold_col: str, pred_col: str, labels: list[str]) -> pd.DataFrame:
    report = classification_report(
        df[gold_col].map(normalize_label),
        df[pred_col].map(normalize_label),
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    return pd.DataFrame(report).T.reset_index().rename(columns={"index": "label_or_average"})


def plot_label_distribution(df: pd.DataFrame, label_col: str, labels: list[str], path: Path, title: str) -> None:
    if "split" in df.columns:
        splits = [x for x in ["train", "validation", "test"] if x in set(df["split"].astype(str))]
        if not splits:
            splits = sorted(df["split"].astype(str).unique())
        x = np.arange(len(labels))
        width = 0.8 / max(len(splits), 1)
        fig, ax = plt.subplots(figsize=(9, 5))
        for i, split in enumerate(splits):
            counts = df[df["split"].astype(str).eq(split)][label_col].map(normalize_label).value_counts().reindex(labels, fill_value=0)
            ax.bar(x - 0.4 + width / 2 + i * width, counts.values, width=width, label=split)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Count")
        ax.set_title(title)
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
        save_figure(fig, path, title)
    else:
        counts = df[label_col].map(normalize_label).value_counts().reindex(labels, fill_value=0)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(labels, counts.values)
        ax.set_ylabel("Count")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        save_figure(fig, path, title)


def plot_confusion_matrix(cm: np.ndarray, labels: list[str], path: Path, title: str, normalize: bool = False) -> None:
    values = cm.astype(float)
    if normalize:
        row_sums = values.sum(axis=1, keepdims=True)
        values = np.divide(values, row_sums, out=np.zeros_like(values), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(7.5, 6))
    im = ax.imshow(values, aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Gold label")
    ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            text = f"{values[i, j]:.2f}" if normalize else str(int(cm[i, j]))
            ax.text(j, i, text, ha="center", va="center")
    save_figure(fig, path, title)


def plot_gold_vs_pred(df: pd.DataFrame, gold_col: str, pred_col: str, labels: list[str], path: Path, title: str) -> None:
    gold_counts = df[gold_col].map(normalize_label).value_counts().reindex(labels, fill_value=0)
    pred_counts = df[pred_col].map(normalize_label).value_counts().reindex(labels, fill_value=0)
    x = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, gold_counts.values, width=width, label="gold")
    ax.bar(x + width / 2, pred_counts.values, width=width, label="predicted")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, path, title)


def save_eval_assets(df: pd.DataFrame, gold_col: str, pred_col: str, labels: list[str], prefix: str, title: str) -> None:
    report = classification_report_df(df, gold_col, pred_col, labels)
    save_table(report, f"{prefix}_classification_report", f"Classification report for {title}")
    cm = confusion_matrix(df[gold_col].map(normalize_label), df[pred_col].map(normalize_label), labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"true_{x}" for x in labels], columns=[f"pred_{x}" for x in labels])
    save_table(cm_df.reset_index().rename(columns={"index": "true_label"}), f"{prefix}_confusion_matrix", f"Confusion matrix for {title}")
    plot_confusion_matrix(cm, labels, FIG_DIR / f"{prefix}_confusion_matrix.png", f"Confusion matrix - {title}", normalize=False)
    plot_confusion_matrix(cm, labels, FIG_DIR / f"{prefix}_confusion_matrix_normalized.png", f"Normalized confusion matrix - {title}", normalize=True)
    plot_gold_vs_pred(df, gold_col, pred_col, labels, FIG_DIR / f"{prefix}_gold_vs_predicted.png", f"Gold vs predicted - {title}")


def plot_metric_comparison(metrics: pd.DataFrame, path: Path, title: str) -> None:
    if metrics.empty:
        return
    plot_df = metrics.copy()
    plot_df["short_setup"] = plot_df["setup"].str.replace("SciBERT ", "", regex=False).str.replace("RAG ", "RAG\n", regex=False).str.replace("No-RAG ", "No-RAG\n", regex=False)
    x = np.arange(len(plot_df))
    width = 0.27
    fig, ax = plt.subplots(figsize=(max(9, len(plot_df) * 1.9), 5.5))
    for i, metric in enumerate(["accuracy", "macro_f1", "weighted_f1"]):
        ax.bar(x - width + i * width, plot_df[metric].values, width=width, label=metric)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["short_setup"], rotation=25, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, path, title)


# =============================================================================
# Inference
# =============================================================================


def run_rag_model_inference(
    data_df: pd.DataFrame,
    model_dir: Path,
    fallback_label_order: list[str],
    setting_name: str,
    output_prefix: str,
) -> pd.DataFrame:
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except Exception as exc:
        raise RuntimeError("torch and transformers are required for inference") from exc

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    if "split" in data_df.columns:
        test_df = data_df[data_df["split"].astype(str).eq(SPLIT_TO_EVALUATE)].copy().reset_index(drop=True)
    else:
        test_df = data_df.copy().reset_index(drop=True)
    if test_df.empty:
        raise ValueError(f"No rows found for split={SPLIT_TO_EVALUATE} in data for {setting_name}")

    print(f"Loading tokenizer from: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    print(f"Loading model from: {model_dir}")
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    id_to_label, label_to_id = get_label_maps(model, fallback_label_order)
    label_order = [id_to_label[i] for i in sorted(id_to_label)]
    print(f"{setting_name} label order: {label_order}")

    device = torch.device("cuda" if USE_GPU_IF_AVAILABLE and torch.cuda.is_available() else "cpu")
    print(f"Inference device: {device}")
    model.to(device)
    model.eval()

    rows = test_df.to_dict("records")
    all_probs: list[np.ndarray] = []
    all_pred: list[str] = []
    all_conf: list[float] = []

    for start in range(0, len(rows), BATCH_SIZE):
        batch = rows[start:start + BATCH_SIZE]
        texts = [build_rag_input(row) for row in batch]
        enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        pred_ids = probs.argmax(axis=1)
        all_probs.append(probs)
        for prob, pred_id in zip(probs, pred_ids):
            pred_label = id_to_label[int(pred_id)]
            all_pred.append(pred_label)
            all_conf.append(float(prob[int(pred_id)]))
        print(f"Processed {min(start + BATCH_SIZE, len(rows)):,}/{len(rows):,}", end="\r")
    print()

    probs = np.vstack(all_probs)
    out = test_df.copy()
    out["setting"] = setting_name
    out["input_style"] = RAG_INPUT_STYLE
    out["predicted_label_argmax"] = all_pred
    out["confidence_argmax"] = all_conf
    for label, idx in label_to_id.items():
        out[f"prob_{label}"] = probs[:, idx]
    out["correct_argmax"] = out["label"].map(normalize_label).eq(out["predicted_label_argmax"].map(normalize_label))

    compact_cols = [
        c for c in [
            "split", "example_id", "claim", "label", "setting", "input_style",
            "top_rag_score", "predicted_label_argmax", "confidence_argmax", "correct_argmax",
            "prob_false", "prob_mixture", "prob_true", "prob_unproven", "prob_unknown",
        ] if c in out.columns
    ]
    compact_path = PRED_DIR / f"{output_prefix}_predictions_compact.csv"
    out[compact_cols].to_csv(compact_path, index=False)
    record_output(compact_path, "predictions", f"Compact predictions for {setting_name}")
    print(f"Saved predictions: {compact_path}")

    full_path = PRED_DIR / f"{output_prefix}_predictions_full.jsonl"
    out.drop(columns=["rag_results"], errors="ignore").to_json(full_path, orient="records", lines=True, force_ascii=False)
    record_output(full_path, "predictions", f"Full predictions for {setting_name}")
    print(f"Saved predictions: {full_path}")

    return out


# =============================================================================
# EDA sections
# =============================================================================


def make_label_transformation_eda(original_df: pd.DataFrame, three_df: pd.DataFrame) -> pd.DataFrame:
    original = add_row_key(original_df)
    three = add_row_key(three_df)

    # Label distributions.
    for name, df, labels in [
        ("original_4class", original, FOUR_CLASS_LABEL_ORDER),
        ("three_class_unknown", three, THREE_CLASS_LABEL_ORDER),
    ]:
        dist = df.groupby(["split", "label"]).size().reset_index(name="count")
        total = df.groupby("split").size().rename("split_total").reset_index()
        dist = dist.merge(total, on="split", how="left")
        dist["percentage"] = dist["count"] / dist["split_total"] * 100
        save_table(dist, f"label_distribution_{name}_by_split", f"Label distribution for {name}")
        plot_label_distribution(df, "label", labels, FIG_DIR / f"label_distribution_{name}_by_split.png", f"Label distribution - {name}")

    original_small = original[["join_key", "split", "label"]].rename(columns={"label": "original_label"})
    three_small = three[["join_key", "label"]].rename(columns={"label": "new_3class_label"})
    merged = original_small.merge(three_small, on="join_key", how="inner")
    if len(merged) == 0:
        warn("Could not join original and 3-class JSONL by join_key; transformation checks may be incomplete.")
        return merged

    merged["expected_3class_label"] = merged["original_label"].map(collapse_to_unknown)
    merged["transformation_correct"] = merged["new_3class_label"].map(normalize_label).eq(merged["expected_3class_label"])

    map_counts = merged.groupby(["split", "original_label", "new_3class_label"]).size().reset_index(name="count")
    save_table(map_counts, "label_transformation_original_to_unknown_counts", "Original 4-class to 3-class unknown transformation counts")

    audit = merged.groupby("split")["transformation_correct"].agg(rows="count", correct="sum").reset_index()
    audit["incorrect"] = audit["rows"] - audit["correct"]
    audit["pct_correct"] = audit["correct"] / audit["rows"] * 100
    save_table(audit, "label_transformation_audit", "Audit for mixture/unproven to unknown transformation")

    unknown_composition = merged[merged["new_3class_label"].map(normalize_label).eq("unknown")].groupby(
        ["split", "original_label"]
    ).size().reset_index(name="count")
    save_table(unknown_composition, "unknown_label_composition_by_original_label", "Composition of the new unknown label by original labels")

    # Plot unknown composition for the test split.
    test_unknown = unknown_composition[unknown_composition["split"].astype(str).eq(SPLIT_TO_EVALUATE)]
    if not test_unknown.empty:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.bar(test_unknown["original_label"].astype(str), test_unknown["count"].values)
        ax.set_ylabel("Count")
        ax.set_title("Test-set composition of the new unknown label")
        ax.grid(axis="y", alpha=0.25)
        save_figure(fig, FIG_DIR / "unknown_label_test_composition.png", "Unknown label composition on the test split")

    return merged


def load_existing_predictions(path: Path, setting_name: str) -> pd.DataFrame:
    if not path.exists():
        warn(f"Existing prediction file not found for {setting_name}: {path}")
        return pd.DataFrame()
    df = pd.read_csv(path)
    print(f"Loaded existing predictions for {setting_name}: {path} ({len(df):,} rows)")
    return df


def prepare_prediction_df(df: pd.DataFrame, setting_name: str, pred_col: str = "predicted_label_argmax") -> pd.DataFrame:
    out = df.copy()
    if "label" not in out.columns or pred_col not in out.columns:
        warn(f"Prediction dataframe for {setting_name} is missing label or {pred_col}; columns={list(out.columns)}")
        return pd.DataFrame()
    out["setting"] = setting_name
    out["gold_4class_or_native"] = out["label"].map(normalize_label)
    out["pred_4class_or_native"] = out[pred_col].map(normalize_label)
    out["gold_collapsed_3class"] = out["label"].map(collapse_to_unknown)
    out["pred_collapsed_3class"] = out[pred_col].map(collapse_to_unknown)
    return out


def make_unknown_breakdown(three_pred: pd.DataFrame, original_df: pd.DataFrame) -> None:
    if three_pred.empty:
        return
    original_test = add_row_key(original_df)
    if "split" in original_test.columns:
        original_test = original_test[original_test["split"].astype(str).eq(SPLIT_TO_EVALUATE)].copy()
    three_keyed = add_row_key(three_pred)

    original_labels = original_test[["join_key", "label"]].rename(columns={"label": "original_4class_label"})
    merged = three_keyed.merge(original_labels, on="join_key", how="left")
    if "original_4class_label" not in merged.columns or merged["original_4class_label"].isna().all():
        warn("Could not attach original 4-class labels to 3-class predictions for unknown breakdown.")
        return

    cols = [
        c for c in [
            "split", "example_id", "claim", "original_4class_label", "label",
            "predicted_label_argmax", "confidence_argmax", "correct_argmax",
        ] if c in merged.columns
    ]
    examples_path = EXAMPLE_DIR / "three_class_predictions_with_original_4class_labels.csv"
    merged[cols].to_csv(examples_path, index=False)
    record_output(examples_path, "examples", "3-class predictions joined with original 4-class labels")

    # How the model predicts within the collapsed unknown group.
    unknown_gold = merged[merged["label"].map(normalize_label).eq("unknown")].copy()
    if unknown_gold.empty:
        return
    breakdown = unknown_gold.groupby(["original_4class_label", "predicted_label_argmax"]).size().reset_index(name="count")
    save_table(breakdown, "three_class_unknown_gold_prediction_breakdown", "Predictions for examples whose 3-class gold label is unknown")

    pivot = breakdown.pivot_table(index="original_4class_label", columns="predicted_label_argmax", values="count", fill_value=0).reset_index()
    save_table(pivot, "three_class_unknown_gold_prediction_breakdown_pivot", "Pivot table for unknown prediction breakdown")

    # Unknown recall by original label.
    recall_rows = []
    for original_label, group in unknown_gold.groupby("original_4class_label"):
        total = len(group)
        predicted_unknown = int(group["predicted_label_argmax"].map(normalize_label).eq("unknown").sum())
        recall_rows.append({
            "original_4class_label_inside_unknown": original_label,
            "rows": total,
            "predicted_unknown": predicted_unknown,
            "unknown_recall_pct": predicted_unknown / total * 100 if total else 0,
        })
    recall_df = pd.DataFrame(recall_rows)
    save_table(recall_df, "three_class_unknown_recall_by_original_label", "Unknown-class recall split by original mixture vs unproven")

    # Plot breakdown.
    if not pivot.empty:
        pred_cols = [c for c in pivot.columns if c != "original_4class_label"]
        x = np.arange(len(pivot))
        width = 0.8 / max(len(pred_cols), 1)
        fig, ax = plt.subplots(figsize=(8, 5))
        for i, col in enumerate(pred_cols):
            ax.bar(x - 0.4 + width / 2 + i * width, pivot[col].values, width=width, label=f"pred {col}")
        ax.set_xticks(x)
        ax.set_xticklabels(pivot["original_4class_label"].astype(str))
        ax.set_ylabel("Count")
        ax.set_title("3-class model predictions inside the unknown gold class")
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
        save_figure(fig, FIG_DIR / "three_class_unknown_gold_prediction_breakdown.png", "Unknown class prediction breakdown")


def make_common_space_comparisons(
    four_rag: pd.DataFrame,
    no_rag: pd.DataFrame,
    three_rag: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    native_rows: list[dict[str, Any]] = []
    common_rows: list[dict[str, Any]] = []

    # 4-class RAG existing results.
    if not four_rag.empty:
        if "predicted_label_argmax" in four_rag.columns:
            native_rows.append(metric_row(
                four_rag, "label", "predicted_label_argmax", FOUR_CLASS_LABEL_ORDER,
                "RAG SciBERT 4-class argmax", "4-class", "Original four labels."
            ))
            tmp = prepare_prediction_df(four_rag, "RAG SciBERT 4-class argmax", "predicted_label_argmax")
            common_rows.append(metric_row(
                tmp, "gold_collapsed_3class", "pred_collapsed_3class", THREE_CLASS_LABEL_ORDER,
                "RAG SciBERT 4-class argmax collapsed", "common 3-class", "mixture/unproven mapped to unknown after prediction."
            ))
        if "predicted_label_thresholded" in four_rag.columns:
            native_rows.append(metric_row(
                four_rag, "label", "predicted_label_thresholded", FOUR_CLASS_LABEL_ORDER,
                "RAG SciBERT 4-class thresholded", "4-class", "Thresholded post-processing from earlier EDA."
            ))
            tmp = prepare_prediction_df(four_rag, "RAG SciBERT 4-class thresholded", "predicted_label_thresholded")
            common_rows.append(metric_row(
                tmp, "gold_collapsed_3class", "pred_collapsed_3class", THREE_CLASS_LABEL_ORDER,
                "RAG SciBERT 4-class thresholded collapsed", "common 3-class", "Thresholded predictions collapsed to unknown."
            ))

    # No-RAG baseline.
    if not no_rag.empty and "predicted_label_argmax" in no_rag.columns:
        native_rows.append(metric_row(
            no_rag, "label", "predicted_label_argmax", FOUR_CLASS_LABEL_ORDER,
            "No-RAG SciBERT 4-class argmax", "4-class", "Claim-only baseline; RAG context ignored."
        ))
        tmp = prepare_prediction_df(no_rag, "No-RAG SciBERT 4-class argmax", "predicted_label_argmax")
        common_rows.append(metric_row(
            tmp, "gold_collapsed_3class", "pred_collapsed_3class", THREE_CLASS_LABEL_ORDER,
            "No-RAG SciBERT 4-class argmax collapsed", "common 3-class", "Claim-only baseline collapsed to unknown."
        ))

    # Native 3-class RAG model.
    if not three_rag.empty and "predicted_label_argmax" in three_rag.columns:
        native_rows.append(metric_row(
            three_rag, "label", "predicted_label_argmax", THREE_CLASS_LABEL_ORDER,
            "RAG SciBERT 3-class unknown argmax", "3-class", "Model trained with mixture/unproven replaced by unknown."
        ))
        common_rows.append(metric_row(
            three_rag, "label", "predicted_label_argmax", THREE_CLASS_LABEL_ORDER,
            "RAG SciBERT 3-class unknown argmax", "common 3-class", "Native 3-class model."
        ))

    native = pd.DataFrame(native_rows)
    common = pd.DataFrame(common_rows)
    if not native.empty:
        for col in ["accuracy", "macro_f1", "weighted_f1"]:
            native[col] = native[col].astype(float)
        save_table(native, "model_comparison_native_label_spaces", "Model comparison in each setup's native label space")
        plot_metric_comparison(native, FIG_DIR / "model_comparison_native_label_spaces.png", "Model comparison in native label spaces")
    if not common.empty:
        for col in ["accuracy", "macro_f1", "weighted_f1"]:
            common[col] = common[col].astype(float)
        save_table(common, "model_comparison_common_3class_space", "Fairer model comparison after mapping mixture/unproven to unknown")
        plot_metric_comparison(common, FIG_DIR / "model_comparison_common_3class_space.png", "Model comparison in common 3-class space")

    return native, common


# =============================================================================
# Markdown summary
# =============================================================================


def format_metric_value(x: Any) -> str:
    try:
        return f"{float(x):.4f}"
    except Exception:
        return str(x)


def metrics_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "No metrics available."
    out = df.copy()
    for col in ["accuracy", "macro_f1", "weighted_f1"]:
        if col in out.columns:
            out[col] = out[col].map(format_metric_value)
    return out.to_markdown(index=False)


def best_by_metric(df: pd.DataFrame, metric: str) -> str:
    if df.empty or metric not in df.columns:
        return "not available"
    idx = df[metric].astype(float).idxmax()
    row = df.loc[idx]
    return f"{row['setup']} ({metric}={float(row[metric]):.4f})"


def write_report_snippets(native_metrics: pd.DataFrame, common_metrics: pd.DataFrame) -> None:
    path = OUTPUT_DIR / "REPORT_SNIPPETS_MODEL_COMPARISON.md"
    lines = [
        "# Report Snippets: New 3-Class Unknown Model and No-RAG Baseline",
        "",
        "## Native label-space comparison",
        "",
        metrics_to_markdown(native_metrics),
        "",
        "## Common 3-class comparison",
        "",
        "For this comparison, the original `mixture` and `unproven` labels are mapped to `unknown`. This makes the 4-class RAG model, no-RAG baseline, and native 3-class model easier to compare in the same label space.",
        "",
        metrics_to_markdown(common_metrics),
        "",
        "## Report-ready interpretation template",
        "",
        "The project also tested whether combining the two most ambiguous labels, `mixture` and `unproven`, into a single `unknown` class would produce a more stable classification problem. To create this dataset, the original processed RAG JSONL was transformed so that every `mixture` and `unproven` label became `unknown`, while `false` and `true` were preserved. A new SciBERT RAG model was then trained on this three-class version of the task.",
        "",
        "A separate no-RAG baseline was also evaluated. This baseline used the same claim examples but ignored the retrieved RAG context during inference. This provided a direct way to test whether the retrieved evidence was helping beyond the claim text alone.",
        "",
        f"In the common three-class comparison, the best setup by accuracy was **{best_by_metric(common_metrics, 'accuracy')}**.",
        f"The best setup by weighted-F1 was **{best_by_metric(common_metrics, 'weighted_f1')}**.",
        "",
        "These results should be discussed carefully because the native 4-class and native 3-class metrics are not measuring exactly the same task. The fairer comparison is the common three-class view, where all systems are evaluated after mapping `mixture` and `unproven` to `unknown`.",
        "",
        "## Output files",
        "",
        "| Category | Path | Description |",
        "|---|---|---|",
    ]
    for rec in OUTPUT_RECORDS:
        lines.append(f"| {rec['category']} | `{rec['path']}` | {rec['description']} |")
    if WARNINGS:
        lines += ["", "## Warnings", ""]
        for w in WARNINGS:
            lines.append(f"- {w}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    record_output(path, "report_notes", "Report-ready snippets for model comparison section")
    print(f"Saved report snippets: {path}")


def write_index(native_metrics: pd.DataFrame, common_metrics: pd.DataFrame) -> None:
    path = OUTPUT_DIR / "MODEL_COMPARISON_INDEX.md"
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# PubHealth Model Comparison EDA Index",
        "",
        f"Generated: {now}",
        "",
        "## Configuration",
        "",
        f"- Original 4-class JSONL: `{ORIGINAL_4CLASS_JSONL}`",
        f"- New 3-class JSONL: `{THREE_CLASS_JSONL}`",
        f"- 3-class model: `{THREE_CLASS_RAG_MODEL_DIR}`",
        f"- No-RAG predictions expected from: `{NO_RAG_BASELINE_PREDICTIONS}`",
        f"- Split evaluated: `{SPLIT_TO_EVALUATE}`",
        f"- RAG input style: `{RAG_INPUT_STYLE}`",
        "",
        "## Native label-space metrics",
        "",
        metrics_to_markdown(native_metrics),
        "",
        "## Common 3-class metrics",
        "",
        metrics_to_markdown(common_metrics),
        "",
        "## Output files",
        "",
        "| Category | Path | Description |",
        "|---|---|---|",
    ]
    for rec in OUTPUT_RECORDS:
        lines.append(f"| {rec['category']} | `{rec['path']}` | {rec['description']} |")
    if WARNINGS:
        lines += ["", "## Warnings", ""]
        for w in WARNINGS:
            lines.append(f"- {w}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    record_output(path, "index", "Markdown index for model comparison outputs")
    print(f"Saved index: {path}")


def write_manifest(native_metrics: pd.DataFrame, common_metrics: pd.DataFrame) -> None:
    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "script": Path(__file__).name,
        "output_dir": str(OUTPUT_DIR),
        "split": SPLIT_TO_EVALUATE,
        "original_4class_jsonl": str(ORIGINAL_4CLASS_JSONL),
        "three_class_jsonl": str(THREE_CLASS_JSONL),
        "four_class_rag_model_dir": str(FOUR_CLASS_RAG_MODEL_DIR),
        "three_class_rag_model_dir": str(THREE_CLASS_RAG_MODEL_DIR),
        "existing_four_class_rag_predictions": str(EXISTING_FOUR_CLASS_RAG_PREDICTIONS),
        "no_rag_baseline_predictions": str(NO_RAG_BASELINE_PREDICTIONS),
        "native_metrics": native_metrics.to_dict(orient="records") if not native_metrics.empty else [],
        "common_3class_metrics": common_metrics.to_dict(orient="records") if not common_metrics.empty else [],
        "warnings": WARNINGS,
        "outputs": OUTPUT_RECORDS,
    }
    path = OUTPUT_DIR / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved manifest: {path}")


def make_zip_bundle() -> None:
    archive_base = PROJECT_ROOT / "EDA" / "model_comparison_bundle"
    archive_path = shutil.make_archive(str(archive_base), "zip", root_dir=OUTPUT_DIR)
    print(f"Saved zip bundle: {archive_path}")


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    print("=" * 90)
    print("PubHealth model comparison EDA")
    print("=" * 90)
    ensure_dirs()
    set_seeds()

    original_df = read_jsonl(ORIGINAL_4CLASS_JSONL)
    three_df = read_jsonl(THREE_CLASS_JSONL)

    transformation_join = make_label_transformation_eda(original_df, three_df)

    # Load or create 4-class RAG predictions.
    four_rag_pred = load_existing_predictions(EXISTING_FOUR_CLASS_RAG_PREDICTIONS, "existing 4-class RAG")
    if four_rag_pred.empty and RUN_4CLASS_RAG_INFERENCE_IF_MISSING:
        try:
            four_rag_pred = run_rag_model_inference(
                original_df,
                FOUR_CLASS_RAG_MODEL_DIR,
                FOUR_CLASS_LABEL_ORDER,
                "RAG SciBERT 4-class argmax",
                "rag_4class_argmax",
            )
        except Exception as exc:
            warn(f"Could not run 4-class RAG inference: {exc}")
            four_rag_pred = pd.DataFrame()

    # Load no-RAG baseline predictions from separate script.
    no_rag_pred = load_existing_predictions(NO_RAG_BASELINE_PREDICTIONS, "no-RAG baseline")

    # Run new 3-class RAG model.
    three_rag_pred = pd.DataFrame()
    if RUN_3CLASS_RAG_INFERENCE:
        try:
            three_rag_pred = run_rag_model_inference(
                three_df,
                THREE_CLASS_RAG_MODEL_DIR,
                THREE_CLASS_LABEL_ORDER,
                "RAG SciBERT 3-class unknown argmax",
                "rag_3class_unknown_argmax",
            )
            save_eval_assets(
                three_rag_pred,
                "label",
                "predicted_label_argmax",
                THREE_CLASS_LABEL_ORDER,
                "rag_3class_unknown_argmax",
                "RAG SciBERT 3-class unknown argmax",
            )
        except Exception as exc:
            warn(f"Could not run 3-class RAG inference: {exc}")
            three_rag_pred = pd.DataFrame()

    # Extra breakdown for what unknown means in terms of original labels.
    make_unknown_breakdown(three_rag_pred, original_df)

    # Save evaluation assets for loaded predictions too.
    if not four_rag_pred.empty and "predicted_label_argmax" in four_rag_pred.columns:
        save_eval_assets(four_rag_pred, "label", "predicted_label_argmax", FOUR_CLASS_LABEL_ORDER, "rag_4class_argmax_loaded", "RAG SciBERT 4-class argmax")
    if not four_rag_pred.empty and "predicted_label_thresholded" in four_rag_pred.columns:
        save_eval_assets(four_rag_pred, "label", "predicted_label_thresholded", FOUR_CLASS_LABEL_ORDER, "rag_4class_thresholded_loaded", "RAG SciBERT 4-class thresholded")
    if not no_rag_pred.empty and "predicted_label_argmax" in no_rag_pred.columns:
        save_eval_assets(no_rag_pred, "label", "predicted_label_argmax", FOUR_CLASS_LABEL_ORDER, "no_rag_baseline_loaded", "No-RAG SciBERT 4-class argmax")

    native_metrics, common_metrics = make_common_space_comparisons(four_rag_pred, no_rag_pred, three_rag_pred)

    # Save a compact merged comparison of true/predictions by example when possible.
    comparison_frames = []
    for name, df, pred_col in [
        ("rag_4class_argmax", four_rag_pred, "predicted_label_argmax"),
        ("no_rag_4class_argmax", no_rag_pred, "predicted_label_argmax"),
        ("rag_3class_unknown_argmax", three_rag_pred, "predicted_label_argmax"),
    ]:
        if df.empty or pred_col not in df.columns:
            continue
        keyed = add_row_key(df)
        small_cols = [c for c in ["join_key", "split", "example_id", "claim", "label", pred_col, "confidence_argmax"] if c in keyed.columns]
        small = keyed[small_cols].copy()
        small["setup"] = name
        small = small.rename(columns={pred_col: "prediction"})
        comparison_frames.append(small)
    if comparison_frames:
        combined_preds = pd.concat(comparison_frames, ignore_index=True)
        save_table(combined_preds, "combined_prediction_rows_long_format", "Combined predictions in long format for manual comparison")

    write_report_snippets(native_metrics, common_metrics)
    write_index(native_metrics, common_metrics)
    write_manifest(native_metrics, common_metrics)
    make_zip_bundle()
    print("Done.")


if __name__ == "__main__":
    main()
