from __future__ import annotations

# =============================================================================
# CONFIG: hardcode everything here
# =============================================================================

from pathlib import Path

DATA_DIR = Path("data")
SOURCE_DIR = DATA_DIR / "pubhealth_source"
PAIRS_DIR = DATA_DIR / "pubhealth_bigbio_pairs"
PROCESSED_RAG_JSONL = DATA_DIR / "processed" / "FULL_no_reform.jsonl"
THRESHOLD_GRID_CSV = DATA_DIR / "threshold_grid_valid" / "threshold_grid_summary.csv"
MODEL_DIR = Path("models") / "scibert_20260605_152114_fa6345e0-e3af-4933-bc23-0dbce6a63b12"

OUTPUT_DIR = Path("EDA")

SPLITS = ["train", "validation", "test"]
LABEL_ORDER = ["false", "mixture", "true", "unproven"]
DEFAULT_LABEL_MAP = {"false": 0, "mixture": 1, "true": 2, "unproven": 3}

# SciBERT/RAG inference settings
RUN_MODEL_INFERENCE = True
BATCH_SIZE = 16
MAX_LENGTH = 512
USE_GPU_IF_AVAILABLE = True

# Choose best validation threshold from threshold_grid_summary.csv by this metric.
# Good options: "macro_f1", "weighted_f1", "accuracy".
BEST_THRESHOLD_METRIC = "macro_f1"

# Token length analysis can use the model tokenizer if available; otherwise it falls
# back to whitespace token counts. Set to False if tokenizer loading is slow.
USE_MODEL_TOKENIZER_FOR_LENGTHS = True

# If you want token analysis on a sample only, set an integer. None = all rows.
TOKEN_LENGTH_SAMPLE_SIZE = None
RANDOM_STATE = 42

# Static reformulation/chunking diagnostics from the presentation. These are used
# because rerunning reformulation is intentionally avoided at this stage.
SLIDE_REFORMULATION_DIAGNOSTICS = [
    {
        "setting": "No reformulation - early diagnostic",
        "mean_top_score": 1.4662,
        "pct_below_1": 42.51,
        "pct_below_0": 35.33,
        "source": "presentation",
    },
    {
        "setting": "With reformulation - early diagnostic",
        "mean_top_score": 1.2878,
        "pct_below_1": 43.57,
        "pct_below_0": 36.18,
        "source": "presentation",
    },
]

SLIDE_CHUNKING_FIX_DIAGNOSTICS = [
    {
        "setting": "Before chunking fix / no reformulation",
        "pct_below_1": 42.51,
        "pct_below_0": 35.33,
        "source": "presentation",
    },
    {
        "setting": "After sentence-window chunking fix",
        "pct_below_1": 11.28,
        "pct_below_0": 7.81,
        "source": "presentation",
    },
]

# Human-readable descriptions. The script also saves the actual schemas from the
# parquet files, so update these descriptions only if your HuggingFace columns are
# more specific.
COLUMN_DESCRIPTIONS = {
    "pubhealth_source": {
        "claim_id": "Unique claim/source identifier, used to connect source rows with paired examples when available.",
        "claim": "Public-health or medical claim to classify.",
        "explanation": "Human-written explanation/rationale associated with the label.",
        "main_text": "Main article/source text used as evidence for the claim.",
        "label": "Gold verification label: false, mixture, true, or unproven.",
        "subjects": "Optional topic/category tags when present.",
        "date_published": "Optional source publication date when present.",
        "fact_checkers": "Optional fact-checker metadata when present.",
        "sources": "Optional original-source metadata when present.",
    },
    "pubhealth_bigbio_pairs": {
        "claim_id": "Claim identifier when present.",
        "document_id": "Document/source identifier used to connect a pair to a source document.",
        "text_1": "First text in the pair, usually the claim.",
        "text_2": "Second text in the pair, usually evidence or paired source text.",
        "label": "Gold verification/relationship label when present.",
    },
    "processed_rag": {
        "split": "Dataset split: train, validation, or test.",
        "example_id": "Example identifier carried into the processed RAG dataset.",
        "claim": "Original medical/public-health claim.",
        "reformulated_query": "Reformulated query field. For no-reform runs, this may match the original claim.",
        "rag_query": "Query actually used for retrieval.",
        "rag_context": "Formatted retrieved evidence passed to the classifier.",
        "rag_results": "Structured retrieved chunks with scores, text, metadata, retrieval method, BM25 score, dense score, and cross-encoder score.",
        "label": "Gold verification label.",
        "top_rag_score": "Highest cross-encoder retrieval score for the example.",
    },
}

# =============================================================================
# Imports and global setup
# =============================================================================

import contextlib
import json
import math
import os
import re
import sys
import traceback
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Iterable

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
    precision_recall_fscore_support,
)

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x


# =============================================================================
# Logging / output helpers
# =============================================================================

FIG_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
PRED_DIR = OUTPUT_DIR / "predictions"
LOG_DIR = OUTPUT_DIR / "logs"
EXAMPLE_DIR = OUTPUT_DIR / "examples"

MANIFEST: list[dict[str, str]] = []
WARNINGS: list[str] = []


def ensure_output_dirs() -> None:
    for p in [OUTPUT_DIR, FIG_DIR, TABLE_DIR, PRED_DIR, LOG_DIR, EXAMPLE_DIR]:
        p.mkdir(parents=True, exist_ok=True)


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


@contextlib.contextmanager
def tee_to_log(log_path: Path):
    old_stdout, old_stderr = sys.stdout, sys.stderr
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        sys.stdout = Tee(old_stdout, f)
        sys.stderr = Tee(old_stderr, f)
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


def warn(msg: str) -> None:
    print(f"WARNING: {msg}")
    WARNINGS.append(msg)


def record_output(path: Path, kind: str, description: str) -> None:
    MANIFEST.append({
        "kind": kind,
        "path": str(path),
        "description": description,
    })


def df_to_markdown_simple(df: pd.DataFrame, index: bool = False) -> str:
    if index:
        df2 = df.reset_index()
    else:
        df2 = df.copy()
    if df2.empty:
        return "_Empty table._\n"

    # Convert cells to short strings while preserving numbers already rounded.
    df2 = df2.astype(object).where(pd.notna(df2), "")
    headers = [str(c) for c in df2.columns]
    rows = [[str(v) for v in row] for row in df2.to_numpy().tolist()]

    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        safe = [cell.replace("\n", "<br>").replace("|", "\\|") for cell in row]
        out.append("| " + " | ".join(safe) + " |")
    return "\n".join(out) + "\n"


def save_table(df: pd.DataFrame, name: str, description: str, index: bool = False) -> tuple[Path, Path]:
    csv_path = TABLE_DIR / f"{name}.csv"
    md_path = TABLE_DIR / f"{name}.md"
    df.to_csv(csv_path, index=index)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# {description}\n\n")
        f.write(df_to_markdown_simple(df, index=index))
    record_output(csv_path, "table_csv", description)
    record_output(md_path, "table_md", description)
    print(f"Saved table: {csv_path}")
    return csv_path, md_path


def save_text(path: Path, text: str, description: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    record_output(path, "text", description)
    print(f"Saved text: {path}")


def save_fig(fig, path: Path, description: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    record_output(path, "figure", description)
    print(f"Saved figure: {path}")


def clean_filename(value: str) -> str:
    value = str(value).strip().lower()
    value = re.sub(r"[^a-z0-9_\-]+", "_", value)
    return re.sub(r"_+", "_", value).strip("_") or "unnamed"


# =============================================================================
# Loading data
# =============================================================================


def read_parquet_split(base_dir: Path, dataset_name: str) -> dict[str, pd.DataFrame]:
    out = {}
    for split in SPLITS:
        path = base_dir / split / "0000.parquet"
        if not path.exists():
            warn(f"Missing {dataset_name} parquet for split={split}: {path}")
            continue
        df = pd.read_parquet(path)
        df = df.copy()
        df["_split"] = split
        out[split] = df
        print(f"Loaded {dataset_name}/{split}: {len(df):,} rows, {len(df.columns):,} columns")
    return out


def load_processed_rag(path: Path) -> pd.DataFrame:
    if not path.exists():
        warn(f"Processed RAG JSONL not found: {path}")
        return pd.DataFrame()
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                warn(f"Could not parse JSONL line {line_no}: {e}")
    df = pd.DataFrame(rows)
    print(f"Loaded processed RAG: {len(df):,} rows, {len(df.columns):,} columns")
    return df


def concat_splits(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs.values(), ignore_index=True)


# =============================================================================
# Generic computations
# =============================================================================


def safe_str_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series([""] * len(df), index=df.index)
    return df[col].fillna("").astype(str)


def word_count(text: Any) -> int:
    if pd.isna(text):
        return 0
    return len(str(text).split())


def char_count(text: Any) -> int:
    if pd.isna(text):
        return 0
    return len(str(text))


def pct(x: float, denominator: float) -> float:
    if denominator == 0 or pd.isna(denominator):
        return 0.0
    return 100.0 * float(x) / float(denominator)


def flatten_label(value: Any) -> str:
    return str(value).strip().lower()


def label_counts_by_split(df: pd.DataFrame, split_col: str = "_split") -> pd.DataFrame:
    if df.empty or "label" not in df.columns:
        return pd.DataFrame()
    work = df.copy()
    work["label"] = work["label"].map(flatten_label)
    rows = []
    for split, g in work.groupby(split_col):
        total = len(g)
        counts = g["label"].value_counts()
        for label in LABEL_ORDER:
            rows.append({
                "split": split,
                "label": label,
                "count": int(counts.get(label, 0)),
                "percentage": round(pct(counts.get(label, 0), total), 2),
            })
    return pd.DataFrame(rows)


def describe_numeric_by_group(df: pd.DataFrame, value_col: str, group_col: str | None = None) -> pd.DataFrame:
    if df.empty or value_col not in df.columns:
        return pd.DataFrame()
    cols = [value_col] if group_col is None else [group_col, value_col]
    work = df[cols].copy()
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    if group_col is None:
        stats = work[value_col].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95]).to_frame().T
        stats.insert(0, "group", "all")
        return stats.reset_index(drop=True).round(4)
    rows = []
    for group, g in work.groupby(group_col):
        s = g[value_col].dropna()
        if s.empty:
            continue
        rows.append({
            "group": group,
            "count": int(s.count()),
            "mean": round(float(s.mean()), 4),
            "std": round(float(s.std(ddof=1)), 4) if len(s) > 1 else 0.0,
            "min": round(float(s.min()), 4),
            "p25": round(float(s.quantile(0.25)), 4),
            "median": round(float(s.median()), 4),
            "p75": round(float(s.quantile(0.75)), 4),
            "p90": round(float(s.quantile(0.90)), 4),
            "p95": round(float(s.quantile(0.95)), 4),
            "max": round(float(s.max()), 4),
        })
    return pd.DataFrame(rows)


# =============================================================================
# Plotting helpers
# =============================================================================


def plot_grouped_label_distribution(counts_df: pd.DataFrame, out_path: Path, title: str) -> None:
    if counts_df.empty:
        return
    pivot = counts_df.pivot(index="label", columns="split", values="count").reindex(LABEL_ORDER).fillna(0)
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(pivot.index))
    width = 0.8 / max(len(pivot.columns), 1)
    for i, split in enumerate(pivot.columns):
        offsets = x - 0.4 + width / 2 + i * width
        bars = ax.bar(offsets, pivot[split].values, width=width, label=split)
        for bar in bars:
            height = int(bar.get_height())
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(height),
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index)
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    save_fig(fig, out_path, title)


def plot_hist(df: pd.DataFrame, col: str, out_path: Path, title: str, xlabel: str, bins: int = 40, by_label: bool = False) -> None:
    if df.empty or col not in df.columns:
        return
    work = df.copy()
    work[col] = pd.to_numeric(work[col], errors="coerce")
    fig, ax = plt.subplots(figsize=(9, 5))
    if by_label and "label" in work.columns:
        for label in LABEL_ORDER:
            vals = work.loc[work["label"].map(flatten_label) == label, col].dropna()
            if len(vals):
                ax.hist(vals, bins=bins, alpha=0.45, label=label)
        ax.legend()
    else:
        ax.hist(work[col].dropna(), bins=bins, alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Number of examples")
    ax.grid(axis="y", alpha=0.25)
    save_fig(fig, out_path, title)


def plot_box_by_label(df: pd.DataFrame, col: str, out_path: Path, title: str, ylabel: str) -> None:
    if df.empty or col not in df.columns or "label" not in df.columns:
        return
    work = df.copy()
    work["label"] = work["label"].map(flatten_label)
    work[col] = pd.to_numeric(work[col], errors="coerce")
    data = [work.loc[work["label"] == label, col].dropna().values for label in LABEL_ORDER]
    if sum(len(x) for x in data) == 0:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(data, tick_labels=LABEL_ORDER, showfliers=False)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    save_fig(fig, out_path, title)


def plot_simple_bar(df: pd.DataFrame, x_col: str, y_col: str, out_path: Path, title: str, ylabel: str) -> None:
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(df[x_col].astype(str), pd.to_numeric(df[y_col], errors="coerce"))
    for bar in bars:
        v = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.2f}" if abs(v) < 100 else f"{v:.0f}",
                ha="center", va="bottom", fontsize=8)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    save_fig(fig, out_path, title)


def plot_two_metric_comparison(df: pd.DataFrame, category_col: str, metric_cols: list[str], out_path: Path, title: str) -> None:
    if df.empty:
        return
    metric_cols = [c for c in metric_cols if c in df.columns]
    if not metric_cols or category_col not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(df))
    width = 0.8 / len(metric_cols)
    for i, col in enumerate(metric_cols):
        vals = pd.to_numeric(df[col], errors="coerce")
        offsets = x - 0.4 + width / 2 + i * width
        bars = ax.bar(offsets, vals, width=width, label=col)
        for bar in bars:
            v = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(df[category_col].astype(str), rotation=15, ha="right")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    save_fig(fig, out_path, title)


def plot_confusion(cm: np.ndarray, labels: list[str], out_path: Path, title: str, normalize: bool = False) -> None:
    if cm.size == 0:
        return
    data = cm.astype(float)
    if normalize:
        row_sums = data.sum(axis=1, keepdims=True)
        data = np.divide(data, row_sums, out=np.zeros_like(data), where=row_sums != 0)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(data, aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    thresh = data.max() / 2 if data.max() > 0 else 0
    for i in range(len(labels)):
        for j in range(len(labels)):
            txt = f"{data[i, j]:.2f}" if normalize else f"{int(data[i, j])}"
            color = "white" if data[i, j] > thresh else "black"
            ax.text(j, i, txt, ha="center", va="center", color=color)
    save_fig(fig, out_path, title)


def plot_gold_vs_pred(df: pd.DataFrame, pred_col: str, out_path: Path, title: str) -> None:
    if df.empty or "label" not in df.columns or pred_col not in df.columns:
        return
    work = df.copy()
    work["label"] = work["label"].map(flatten_label)
    work[pred_col] = work[pred_col].map(flatten_label)
    gold = work["label"].value_counts().reindex(LABEL_ORDER, fill_value=0)
    pred = work[pred_col].value_counts().reindex(LABEL_ORDER, fill_value=0)
    x = np.arange(len(LABEL_ORDER))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - width / 2, gold.values, width, label="gold")
    b2 = ax.bar(x + width / 2, pred.values, width, label="predicted")
    for bars in [b1, b2]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(int(bar.get_height())),
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(LABEL_ORDER)
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    save_fig(fig, out_path, title)


def plot_heatmap_from_grid(grid: pd.DataFrame, value_col: str, out_path: Path, title: str) -> None:
    required = {"mixture_threshold", "unproven_threshold", value_col}
    if grid.empty or not required.issubset(set(grid.columns)):
        return
    pivot = grid.pivot_table(index="unproven_threshold", columns="mixture_threshold", values=value_col, aggfunc="mean")
    pivot = pivot.sort_index().sort_index(axis=1)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(pivot.values, aspect="auto", origin="lower")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels([str(x) for x in pivot.columns])
    ax.set_yticklabels([str(x) for x in pivot.index])
    ax.set_xlabel("mixture threshold")
    ax.set_ylabel("unproven threshold")
    ax.set_title(title)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8)
    save_fig(fig, out_path, title)


# =============================================================================
# Schema / dataset EDA
# =============================================================================


def make_schema_tables(source_dfs: dict[str, pd.DataFrame], pairs_dfs: dict[str, pd.DataFrame], rag_df: pd.DataFrame) -> None:
    rows = []
    for dataset_name, dfs in [("pubhealth_source", source_dfs), ("pubhealth_bigbio_pairs", pairs_dfs)]:
        for split, df in dfs.items():
            for col in df.columns:
                rows.append({
                    "dataset": dataset_name,
                    "split": split,
                    "column": col,
                    "dtype": str(df[col].dtype),
                    "non_null": int(df[col].notna().sum()),
                    "missing": int(df[col].isna().sum()),
                    "description": COLUMN_DESCRIPTIONS.get(dataset_name, {}).get(col, ""),
                })
    if not rag_df.empty:
        for col in rag_df.columns:
            rows.append({
                "dataset": "processed_rag",
                "split": "all",
                "column": col,
                "dtype": str(rag_df[col].dtype),
                "non_null": int(rag_df[col].notna().sum()),
                "missing": int(rag_df[col].isna().sum()),
                "description": COLUMN_DESCRIPTIONS.get("processed_rag", {}).get(col, ""),
            })
    save_table(pd.DataFrame(rows), "schema_and_column_descriptions", "Schema and column descriptions")


def make_dataset_overview(source_dfs: dict[str, pd.DataFrame], pairs_dfs: dict[str, pd.DataFrame], rag_df: pd.DataFrame) -> None:
    rows = []
    for dataset_name, dfs in [("pubhealth_source", source_dfs), ("pubhealth_bigbio_pairs", pairs_dfs)]:
        for split in SPLITS:
            df = dfs.get(split, pd.DataFrame())
            rows.append({
                "dataset": dataset_name,
                "split": split,
                "rows": int(len(df)),
                "columns": int(len(df.columns)) if not df.empty else 0,
                "path": str((SOURCE_DIR if dataset_name == "pubhealth_source" else PAIRS_DIR) / split / "0000.parquet"),
            })
    if not rag_df.empty:
        if "split" in rag_df.columns:
            for split in SPLITS:
                g = rag_df[rag_df["split"].astype(str) == split]
                rows.append({
                    "dataset": "processed_rag",
                    "split": split,
                    "rows": int(len(g)),
                    "columns": int(len(rag_df.columns)),
                    "path": str(PROCESSED_RAG_JSONL),
                })
        else:
            rows.append({
                "dataset": "processed_rag",
                "split": "all",
                "rows": int(len(rag_df)),
                "columns": int(len(rag_df.columns)),
                "path": str(PROCESSED_RAG_JSONL),
            })
    save_table(pd.DataFrame(rows), "dataset_overview", "Dataset overview")


def make_missing_duplicate_tables(source_df: pd.DataFrame, rag_df: pd.DataFrame) -> None:
    rows = []
    for name, df in [("pubhealth_source_all", source_df), ("processed_rag_all", rag_df)]:
        if df.empty:
            continue
        for col in df.columns:
            miss = int(df[col].isna().sum())
            empty = 0
            if df[col].dtype == object:
                empty = int(df[col].fillna("").astype(str).str.strip().eq("").sum())
            rows.append({
                "dataset": name,
                "column": col,
                "missing_count": miss,
                "missing_percent": round(pct(miss, len(df)), 3),
                "empty_string_count": empty,
                "empty_string_percent": round(pct(empty, len(df)), 3),
            })
    save_table(pd.DataFrame(rows), "missing_values", "Missing values and empty strings")

    checks = []
    for name, df in [("pubhealth_source_all", source_df), ("processed_rag_all", rag_df)]:
        if df.empty:
            continue
        for col in ["claim", "main_text", "explanation", "rag_context"]:
            if col in df.columns:
                checks.append({
                    "dataset": name,
                    "check": f"duplicate_{col}",
                    "count": int(df[col].duplicated().sum()),
                })
                checks.append({
                    "dataset": name,
                    "check": f"unique_{col}",
                    "count": int(df[col].nunique(dropna=True)),
                })
        if "claim" in df.columns:
            checks.append({
                "dataset": name,
                "check": "empty_claims",
                "count": int(df["claim"].fillna("").astype(str).str.strip().eq("").sum()),
            })
    save_table(pd.DataFrame(checks), "duplicate_and_empty_checks", "Duplicate and empty-field checks")


def make_label_eda(source_df: pd.DataFrame, rag_df: pd.DataFrame) -> None:
    if not source_df.empty and "label" in source_df.columns:
        counts = label_counts_by_split(source_df, "_split")
        save_table(counts, "label_distribution_source_by_split", "Label distribution in pubhealth_source by split")
        plot_grouped_label_distribution(counts, FIG_DIR / "label_distribution_source_by_split.png", "Label distribution by split - pubhealth_source")

    if not rag_df.empty and "label" in rag_df.columns:
        split_col = "split" if "split" in rag_df.columns else None
        if split_col:
            counts = label_counts_by_split(rag_df, "split")
            save_table(counts, "label_distribution_processed_rag_by_split", "Label distribution in processed RAG data by split")
            plot_grouped_label_distribution(counts, FIG_DIR / "label_distribution_processed_rag_by_split.png", "Label distribution by split - processed RAG")
        else:
            counts = rag_df["label"].map(flatten_label).value_counts().reindex(LABEL_ORDER, fill_value=0).reset_index()
            counts.columns = ["label", "count"]
            counts["percentage"] = counts["count"].map(lambda x: round(pct(x, len(rag_df)), 2))
            save_table(counts, "label_distribution_processed_rag", "Label distribution in processed RAG data")


# =============================================================================
# Length and text EDA
# =============================================================================


def add_basic_lengths(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[f"{col}_word_count"] = out[col].map(word_count)
            out[f"{col}_char_count"] = out[col].map(char_count)
    return out


def get_tokenizer(model_dir: Path):
    if not USE_MODEL_TOKENIZER_FOR_LENGTHS:
        return None
    try:
        from transformers import AutoTokenizer
        if model_dir.exists():
            print(f"Loading tokenizer for token-length EDA from {model_dir}")
            return AutoTokenizer.from_pretrained(model_dir)
        warn(f"Tokenizer/model path not found for token-length EDA: {model_dir}. Falling back to whitespace counts.")
        return None
    except Exception as e:
        warn(f"Could not load tokenizer for token-length EDA: {e}. Falling back to whitespace counts.")
        return None


def count_tokens(texts: list[str], tokenizer=None) -> list[int]:
    if tokenizer is None:
        return [len(t.split()) for t in texts]
    counts = []
    for t in tqdm(texts, desc="Counting tokens"):
        try:
            counts.append(len(tokenizer(str(t), truncation=False, add_special_tokens=True)["input_ids"]))
        except Exception:
            counts.append(len(str(t).split()))
    return counts


def build_rag_input_for_model(example: dict[str, Any]) -> str:
    claim = str(example.get("claim", "")).strip()
    query = str(example.get("rag_query", example.get("reformulated_query", claim))).strip()
    context = str(example.get("rag_context", "")).strip() or "No evidence retrieved."
    return (
        f"Original claim:\n{claim}\n\n"
        f"Search query:\n{query}\n\n"
        f"Retrieved evidence:\n{context}"
    )


def make_length_eda(source_df: pd.DataFrame, rag_df: pd.DataFrame) -> None:
    if not source_df.empty:
        source_len = add_basic_lengths(source_df, ["claim", "main_text", "explanation"])
        length_cols = [c for c in source_len.columns if c.endswith("_word_count") or c.endswith("_char_count")]
        rows = []
        for col in length_cols:
            overall = describe_numeric_by_group(source_len, col)
            if not overall.empty:
                overall.insert(0, "feature", col)
                rows.append(overall)
            if "label" in source_len.columns:
                by_label = describe_numeric_by_group(source_len, col, "label")
                if not by_label.empty:
                    by_label.insert(0, "feature", col)
                    rows.append(by_label)
        if rows:
            summary = pd.concat(rows, ignore_index=True)
            save_table(summary, "text_length_summary_source", "Text length summary for pubhealth_source")

        for col, label in [
            ("claim_word_count", "Claim length (words)"),
            ("main_text_word_count", "Main text length (words)"),
            ("explanation_word_count", "Explanation length (words)"),
        ]:
            if col in source_len.columns:
                plot_hist(source_len, col, FIG_DIR / f"hist_{col}.png", f"{label} distribution", label, bins=50)
                plot_box_by_label(source_len, col, FIG_DIR / f"box_{col}_by_label.png", f"{label} by label", label)

    if rag_df.empty:
        return

    tokenizer = get_tokenizer(MODEL_DIR)
    work = rag_df.copy()
    if TOKEN_LENGTH_SAMPLE_SIZE is not None and len(work) > TOKEN_LENGTH_SAMPLE_SIZE:
        work = work.sample(TOKEN_LENGTH_SAMPLE_SIZE, random_state=RANDOM_STATE).reset_index(drop=True)

    text_variants = {}
    if "claim" in work.columns:
        text_variants["claim_only"] = safe_str_series(work, "claim").tolist()
    if "claim" in work.columns and "rag_context" in work.columns:
        text_variants["classifier_input_claim_rag"] = [build_rag_input_for_model(r) for r in work.to_dict("records")]
    if "claim" in work.columns and "explanation" in work.columns:
        text_variants["claim_plus_explanation"] = (
            safe_str_series(work, "claim") + "\n\n" + safe_str_series(work, "explanation")
        ).tolist()

    token_summary_rows = []
    for name, texts in text_variants.items():
        counts = count_tokens(texts, tokenizer=tokenizer)
        s = pd.Series(counts)
        work[f"tokens_{name}"] = counts
        token_summary_rows.append({
            "input_format": name,
            "rows_analyzed": int(len(s)),
            "mean_tokens": round(float(s.mean()), 2),
            "median_tokens": round(float(s.median()), 2),
            "p90_tokens": round(float(s.quantile(0.9)), 2),
            "p95_tokens": round(float(s.quantile(0.95)), 2),
            "max_tokens": int(s.max()) if len(s) else 0,
            "pct_over_512": round(pct((s > 512).sum(), len(s)), 2),
            "pct_over_max_length": round(pct((s > MAX_LENGTH).sum(), len(s)), 2),
            "tokenizer": "model_tokenizer" if tokenizer is not None else "whitespace_fallback",
        })
        plot_hist(work, f"tokens_{name}", FIG_DIR / f"hist_tokens_{clean_filename(name)}.png", f"Token length distribution: {name}", "Tokens", bins=50)

    token_summary = pd.DataFrame(token_summary_rows)
    save_table(token_summary, "token_truncation_summary", "Token length and truncation summary")
    plot_simple_bar(token_summary, "input_format", "pct_over_512", FIG_DIR / "truncation_percent_over_512.png", "% of inputs over 512 tokens", "% over 512")


# =============================================================================
# TF-IDF / top word EDA
# =============================================================================


def make_top_terms_eda(source_df: pd.DataFrame) -> None:
    if source_df.empty or "claim" not in source_df.columns or "label" not in source_df.columns:
        return
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except Exception as e:
        warn(f"Could not import TfidfVectorizer, skipping top terms: {e}")
        return

    work = source_df[["claim", "label"]].dropna().copy()
    work["label"] = work["label"].map(flatten_label)
    rows = []
    for label in LABEL_ORDER:
        texts = work.loc[work["label"] == label, "claim"].astype(str).tolist()
        if len(texts) < 2:
            continue
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2, max_features=5000)
        try:
            X = vectorizer.fit_transform(texts)
            mean_scores = np.asarray(X.mean(axis=0)).ravel()
            terms = np.array(vectorizer.get_feature_names_out())
            top_idx = mean_scores.argsort()[::-1][:20]
            for rank, idx in enumerate(top_idx, start=1):
                rows.append({
                    "label": label,
                    "rank": rank,
                    "term": terms[idx],
                    "mean_tfidf": round(float(mean_scores[idx]), 6),
                })
        except Exception as e:
            warn(f"Top terms failed for label={label}: {e}")
    top_terms = pd.DataFrame(rows)
    save_table(top_terms, "top_tfidf_terms_by_label", "Top TF-IDF terms in claims by label")

    if not top_terms.empty:
        # Plot top 10 per label as separate figures for readability.
        for label in LABEL_ORDER:
            sub = top_terms[top_terms["label"] == label].head(10).iloc[::-1]
            if sub.empty:
                continue
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.barh(sub["term"], sub["mean_tfidf"])
            ax.set_title(f"Top claim TF-IDF terms: {label}")
            ax.set_xlabel("Mean TF-IDF")
            ax.grid(axis="x", alpha=0.25)
            save_fig(fig, FIG_DIR / f"top_terms_{label}.png", f"Top claim TF-IDF terms for {label}")


# =============================================================================
# RAG EDA
# =============================================================================


def parse_rag_results(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [x for x in value if isinstance(x, dict)]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [x for x in parsed if isinstance(x, dict)]
        except Exception:
            return []
    return []


def get_score_from_result(r: dict[str, Any]) -> float | None:
    for key in ["score", "cross_encoder_score"]:
        if key in r and r[key] is not None:
            try:
                return float(r[key])
            except Exception:
                pass
    return None


def add_rag_features(rag_df: pd.DataFrame) -> pd.DataFrame:
    if rag_df.empty:
        return rag_df
    out = rag_df.copy()
    results_col = None
    for candidate in ["rag_results", "retrieved_chunks"]:
        if candidate in out.columns:
            results_col = candidate
            break

    parsed_results = []
    top_scores = []
    mean_scores = []
    min_scores = []
    num_chunks = []
    num_below_1 = []
    num_below_0 = []
    retrieved_by_counts = []
    top_sources = []
    any_source_exact = []
    top_source_exact = []

    for _, row in out.iterrows():
        results = parse_rag_results(row.get(results_col, [])) if results_col else []
        parsed_results.append(results)
        scores = [s for s in (get_score_from_result(r) for r in results) if s is not None]
        if "top_rag_score" in out.columns and pd.notna(row.get("top_rag_score")):
            try:
                row_top = float(row.get("top_rag_score"))
            except Exception:
                row_top = None
        else:
            row_top = None

        top = row_top if row_top is not None else (max(scores) if scores else np.nan)
        top_scores.append(top)
        mean_scores.append(float(np.mean(scores)) if scores else np.nan)
        min_scores.append(float(np.min(scores)) if scores else np.nan)
        num_chunks.append(len(results))
        num_below_1.append(int(sum(s < 1 for s in scores)))
        num_below_0.append(int(sum(s < 0 for s in scores)))

        rb_counter = Counter()
        sources = []
        for r in results:
            rb = r.get("retrieved_by", [])
            if isinstance(rb, str):
                rb = [rb]
            if isinstance(rb, list):
                for item in rb:
                    rb_counter[str(item)] += 1
            source = r.get("source", None)
            if source is None:
                meta = r.get("metadata", {})
                if isinstance(meta, dict):
                    source = meta.get("source", None)
            if source is not None:
                sources.append(str(source))
        retrieved_by_counts.append(json.dumps(dict(rb_counter), ensure_ascii=False))
        top_sources.append(sources[0] if sources else "")

        expected_source = None
        if "split" in out.columns and "example_id" in out.columns:
            expected_source = f"{row.get('split')}_{row.get('example_id')}"
        elif "_split" in out.columns and "claim_id" in out.columns:
            expected_source = f"{row.get('_split')}_{row.get('claim_id')}"
        if expected_source:
            any_source_exact.append(any(s == expected_source for s in sources))
            top_source_exact.append(bool(sources and sources[0] == expected_source))
        else:
            any_source_exact.append(np.nan)
            top_source_exact.append(np.nan)

    out["_parsed_rag_results"] = parsed_results
    out["rag_num_chunks"] = num_chunks
    out["rag_top_score_computed"] = top_scores
    out["rag_mean_score"] = mean_scores
    out["rag_min_score"] = min_scores
    out["rag_num_chunks_below_1"] = num_below_1
    out["rag_num_chunks_below_0"] = num_below_0
    out["rag_retrieved_by_counts"] = retrieved_by_counts
    out["rag_top_source"] = top_sources
    out["rag_top_source_exact_match"] = top_source_exact
    out["rag_any_source_exact_match"] = any_source_exact
    return out


def make_rag_eda(rag_df: pd.DataFrame) -> pd.DataFrame:
    if rag_df.empty:
        return rag_df
    rag = add_rag_features(rag_df)
    score_col = "rag_top_score_computed"

    save_table(pd.DataFrame(SLIDE_REFORMULATION_DIAGNOSTICS), "reformulation_diagnostics_from_slides", "Static reformulation diagnostics from presentation")
    plot_two_metric_comparison(
        pd.DataFrame(SLIDE_REFORMULATION_DIAGNOSTICS),
        "setting",
        ["pct_below_1", "pct_below_0"],
        FIG_DIR / "reformulation_below_score_comparison.png",
        "Reformulation diagnostic: low-score retrieval rates",
    )

    static_chunk = pd.DataFrame(SLIDE_CHUNKING_FIX_DIAGNOSTICS)
    save_table(static_chunk, "chunking_fix_diagnostics_from_slides", "Static chunking-fix diagnostics from presentation")
    plot_two_metric_comparison(
        static_chunk,
        "setting",
        ["pct_below_1", "pct_below_0"],
        FIG_DIR / "chunking_fix_below_score_comparison.png",
        "Chunking fix diagnostic: low-score retrieval rates",
    )

    rows = []
    groups = [("all", rag)]
    if "split" in rag.columns:
        groups += [(str(k), g) for k, g in rag.groupby("split")]
    for name, g in groups:
        s = pd.to_numeric(g[score_col], errors="coerce").dropna()
        if s.empty:
            continue
        rows.append({
            "split": name,
            "examples": int(len(g)),
            "examples_with_score": int(len(s)),
            "mean_top_score": round(float(s.mean()), 4),
            "median_top_score": round(float(s.median()), 4),
            "p10_top_score": round(float(s.quantile(0.10)), 4),
            "p90_top_score": round(float(s.quantile(0.90)), 4),
            "pct_top_below_1": round(pct((s < 1).sum(), len(s)), 2),
            "pct_top_below_0": round(pct((s < 0).sum(), len(s)), 2),
            "avg_num_chunks": round(float(pd.to_numeric(g["rag_num_chunks"], errors="coerce").mean()), 2),
        })
    rag_summary = pd.DataFrame(rows)
    save_table(rag_summary, "rag_score_summary_current_processed", "RAG score summary from current processed JSONL")

    plot_hist(rag, score_col, FIG_DIR / "rag_top_score_distribution.png", "RAG top-score distribution - current processed data", "Top cross-encoder score", bins=60, by_label=True)
    plot_box_by_label(rag, score_col, FIG_DIR / "rag_top_score_by_label.png", "RAG top score by gold label", "Top cross-encoder score")

    if "rag_num_chunks" in rag.columns:
        plot_hist(rag, "rag_num_chunks", FIG_DIR / "rag_num_chunks_distribution.png", "Number of retrieved chunks per example", "Retrieved chunks", bins=20)

    # Source-match sanity diagnostic. This can reveal when the indexed corpus contains
    # the exact source document for the same example.
    if "rag_top_source_exact_match" in rag.columns and rag["rag_top_source_exact_match"].notna().any():
        rows = []
        groups = [("all", rag)]
        if "split" in rag.columns:
            groups += [(str(k), g) for k, g in rag.groupby("split")]
        for name, g in groups:
            top_match = g["rag_top_source_exact_match"].dropna().astype(bool)
            any_match = g["rag_any_source_exact_match"].dropna().astype(bool)
            if len(top_match):
                rows.append({
                    "split": name,
                    "examples": int(len(g)),
                    "pct_top_source_exact_match": round(pct(top_match.sum(), len(top_match)), 2),
                    "pct_any_source_exact_match": round(pct(any_match.sum(), len(any_match)), 2),
                })
        source_match = pd.DataFrame(rows)
        save_table(source_match, "rag_source_match_sanity_check", "RAG source-match sanity check")
        plot_simple_bar(source_match, "split", "pct_top_source_exact_match", FIG_DIR / "rag_top_source_exact_match_by_split.png", "Top retrieved source exact-match rate by split", "% top exact match")

    # Save example tables for manual report writing.
    example_cols = [c for c in ["split", "example_id", "claim", "label", score_col, "rag_top_source", "rag_context"] if c in rag.columns]
    low = rag.sort_values(score_col, ascending=True).head(20)[example_cols]
    high = rag.sort_values(score_col, ascending=False).head(20)[example_cols]
    save_table(low, "lowest_rag_score_examples", "Lowest RAG-score examples")
    save_table(high, "highest_rag_score_examples", "Highest RAG-score examples")

    # Write a short Markdown file with qualitative examples and retrieved chunks.
    examples_md = ["# Qualitative RAG examples\n"]
    sample = rag.sort_values(score_col, ascending=True).head(5)
    for _, row in sample.iterrows():
        examples_md.append("\n---\n")
        examples_md.append(f"## Example: split={row.get('split', '')}, example_id={row.get('example_id', '')}, label={row.get('label', '')}\n")
        examples_md.append(f"**Top score:** {row.get(score_col, np.nan)}\n\n")
        examples_md.append(f"**Claim:** {row.get('claim', '')}\n\n")
        results = row.get("_parsed_rag_results", [])
        for i, r in enumerate(results[:3], start=1):
            score = get_score_from_result(r)
            text = str(r.get("text", "")).strip().replace("\n", " ")
            if len(text) > 700:
                text = text[:700] + "..."
            examples_md.append(f"### Retrieved chunk {i}, score={score}\n\n{text}\n\n")
    save_text(EXAMPLE_DIR / "qualitative_low_rag_examples.md", "\n".join(examples_md), "Qualitative low-RAG-score examples")

    return rag


# =============================================================================
# Threshold grid EDA
# =============================================================================


def make_threshold_grid_eda() -> tuple[float | None, float | None, pd.DataFrame]:
    if not THRESHOLD_GRID_CSV.exists():
        warn(f"Threshold grid CSV not found: {THRESHOLD_GRID_CSV}")
        return None, None, pd.DataFrame()
    grid = pd.read_csv(THRESHOLD_GRID_CSV)
    save_table(grid, "threshold_grid_summary_full", "Full validation threshold grid summary")

    metric = BEST_THRESHOLD_METRIC
    if metric not in grid.columns:
        warn(f"BEST_THRESHOLD_METRIC={metric} not found in threshold grid. Using macro_f1 if available.")
        metric = "macro_f1" if "macro_f1" in grid.columns else grid.columns[-1]

    grid_sorted = grid.sort_values(metric, ascending=False).reset_index(drop=True)
    save_table(grid_sorted.head(20), "threshold_grid_top20", f"Top 20 threshold settings by {metric}")

    for col in ["accuracy", "macro_f1", "weighted_f1", "f1_false", "f1_mixture", "f1_true", "f1_unproven"]:
        if col in grid.columns:
            plot_heatmap_from_grid(grid, col, FIG_DIR / f"threshold_grid_heatmap_{col}.png", f"Validation threshold grid heatmap: {col}")

    best = grid_sorted.iloc[0]
    mixture_t = float(best["mixture_threshold"]) if "mixture_threshold" in best else None
    unproven_t = float(best["unproven_threshold"]) if "unproven_threshold" in best else None
    save_text(
        TABLE_DIR / "best_threshold_setting.txt",
        f"Best threshold metric: {metric}\n"
        f"mixture_threshold: {mixture_t}\n"
        f"unproven_threshold: {unproven_t}\n"
        f"Best validation row:\n{best.to_string()}\n",
        "Best threshold setting selected from validation grid",
    )
    return mixture_t, unproven_t, grid_sorted


# =============================================================================
# Model inference and prediction EDA
# =============================================================================


def load_label_map(model_dir: Path) -> dict[str, int]:
    p = model_dir / "label_map.json"
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "label2id" in data:
            return {str(k): int(v) for k, v in data["label2id"].items()}
        return {str(k): int(v) for k, v in data.items()}
    # Try config if no explicit label_map.json.
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model_dir)
        label2id = getattr(cfg, "label2id", None)
        if label2id and set(label2id.keys()) >= set(LABEL_ORDER):
            return {str(k): int(v) for k, v in label2id.items()}
    except Exception:
        pass
    return DEFAULT_LABEL_MAP.copy()


def predict_thresholded(probs: np.ndarray, label_map: dict[str, int], mixture_threshold: float | None, unproven_threshold: float | None) -> list[str]:
    id_to_label = {v: k for k, v in label_map.items()}
    false_id = label_map["false"]
    mixture_id = label_map["mixture"]
    true_id = label_map["true"]
    unproven_id = label_map["unproven"]

    preds = []
    for p in probs:
        if unproven_threshold is not None and p[unproven_id] >= unproven_threshold:
            preds.append("unproven")
        elif mixture_threshold is not None and p[mixture_id] >= mixture_threshold:
            preds.append("mixture")
        else:
            preds.append("true" if p[true_id] >= p[false_id] else "false")
    return preds


def compute_metrics_table(df: pd.DataFrame, pred_col: str, label_names: list[str], setting: str) -> pd.DataFrame:
    y_true = df["label"].astype(str).map(flatten_label)
    y_pred = df[pred_col].astype(str).map(flatten_label)
    per_label = precision_recall_fscore_support(y_true, y_pred, labels=label_names, zero_division=0)
    rows = [{
        "setting": setting,
        "metric_scope": "overall",
        "label": "all",
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "macro_f1": round(float(f1_score(y_true, y_pred, labels=label_names, average="macro", zero_division=0)), 6),
        "weighted_f1": round(float(f1_score(y_true, y_pred, labels=label_names, average="weighted", zero_division=0)), 6),
        "precision": "",
        "recall": "",
        "f1": "",
        "support": int(len(y_true)),
    }]
    for i, label in enumerate(label_names):
        rows.append({
            "setting": setting,
            "metric_scope": "per_label",
            "label": label,
            "accuracy": "",
            "macro_f1": "",
            "weighted_f1": "",
            "precision": round(float(per_label[0][i]), 6),
            "recall": round(float(per_label[1][i]), 6),
            "f1": round(float(per_label[2][i]), 6),
            "support": int(per_label[3][i]),
        })
    return pd.DataFrame(rows)


def save_classification_outputs(df: pd.DataFrame, pred_col: str, label_names: list[str], setting: str) -> None:
    y_true = df["label"].astype(str).map(flatten_label)
    y_pred = df[pred_col].astype(str).map(flatten_label)
    cm = confusion_matrix(y_true, y_pred, labels=label_names)
    cm_df = pd.DataFrame(cm, index=[f"true_{x}" for x in label_names], columns=[f"pred_{x}" for x in label_names])
    save_table(cm_df.reset_index().rename(columns={"index": "true_label"}), f"confusion_matrix_{setting}", f"Confusion matrix - {setting}")
    plot_confusion(cm, label_names, FIG_DIR / f"confusion_matrix_{setting}.png", f"Confusion matrix - {setting}", normalize=False)
    plot_confusion(cm, label_names, FIG_DIR / f"confusion_matrix_{setting}_normalized.png", f"Normalized confusion matrix - {setting}", normalize=True)
    plot_gold_vs_pred(df, pred_col, FIG_DIR / f"gold_vs_predicted_{setting}.png", f"Gold vs predicted label distribution - {setting}")

    report = classification_report(y_true, y_pred, labels=label_names, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).T.reset_index().rename(columns={"index": "label_or_average"})
    save_table(report_df, f"classification_report_{setting}", f"Classification report - {setting}")

    # Per-class metric bar chart.
    sub = report_df[report_df["label_or_average"].isin(label_names)].copy()
    if not sub.empty:
        fig, ax = plt.subplots(figsize=(9, 5))
        x = np.arange(len(sub))
        width = 0.25
        for i, metric in enumerate(["precision", "recall", "f1-score"]):
            vals = pd.to_numeric(sub[metric], errors="coerce").fillna(0)
            ax.bar(x - width + i * width, vals, width=width, label=metric)
        ax.set_xticks(x)
        ax.set_xticklabels(sub["label_or_average"].astype(str))
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score")
        ax.set_title(f"Per-class precision/recall/F1 - {setting}")
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
        save_fig(fig, FIG_DIR / f"per_class_metrics_{setting}.png", f"Per-class metrics - {setting}")


def run_model_inference(rag_df_with_features: pd.DataFrame, mixture_t: float | None, unproven_t: float | None) -> pd.DataFrame:
    if not RUN_MODEL_INFERENCE:
        warn("RUN_MODEL_INFERENCE=False; skipping model inference.")
        return pd.DataFrame()
    if rag_df_with_features.empty:
        warn("Processed RAG data is empty; skipping model inference.")
        return pd.DataFrame()
    if not MODEL_DIR.exists():
        warn(f"Model directory not found: {MODEL_DIR}; skipping model inference.")
        return pd.DataFrame()

    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except Exception as e:
        warn(f"Could not import torch/transformers; skipping model inference: {e}")
        return pd.DataFrame()

    df = rag_df_with_features.copy()
    if "split" in df.columns:
        test_df = df[df["split"].astype(str).eq("test")].copy().reset_index(drop=True)
    else:
        warn("No split column found in processed RAG data; using all rows for inference.")
        test_df = df.copy().reset_index(drop=True)

    if test_df.empty:
        warn("No test rows found in processed RAG data; skipping model inference.")
        return pd.DataFrame()
    if "label" not in test_df.columns:
        warn("No label column found in test data; inference will run but evaluation will be skipped.")

    label_map = load_label_map(MODEL_DIR)
    # Keep only expected labels in ID order for metrics.
    id_to_label = {v: k for k, v in label_map.items()}
    label_names = [id_to_label[i] for i in sorted(id_to_label) if id_to_label[i] in LABEL_ORDER]
    if not label_names:
        label_map = DEFAULT_LABEL_MAP.copy()
        id_to_label = {v: k for k, v in label_map.items()}
        label_names = LABEL_ORDER.copy()

    print(f"Loading model from: {MODEL_DIR}")
    device = torch.device("cuda" if USE_GPU_IF_AVAILABLE and torch.cuda.is_available() else "cpu")
    print(f"Inference device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(device)
    model.eval()

    rows = test_df.to_dict("records")
    all_probs = []
    all_pred_argmax = []
    all_conf_argmax = []

    for start in tqdm(range(0, len(rows), BATCH_SIZE), desc="Running SciBERT inference on test split"):
        batch = rows[start:start + BATCH_SIZE]
        texts = [build_rag_input_for_model(row) for row in batch]
        enc = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        pred_ids = probs.argmax(axis=1)
        all_probs.append(probs)
        for p, pred_id in zip(probs, pred_ids):
            all_pred_argmax.append(id_to_label[int(pred_id)])
            all_conf_argmax.append(float(p[int(pred_id)]))

    probs = np.vstack(all_probs)
    result = test_df.copy()
    result["predicted_label_argmax"] = all_pred_argmax
    result["confidence_argmax"] = all_conf_argmax
    for label in label_names:
        result[f"prob_{label}"] = probs[:, label_map[label]]

    if mixture_t is not None or unproven_t is not None:
        result["predicted_label_thresholded"] = predict_thresholded(probs, label_map, mixture_t, unproven_t)
        result["confidence_thresholded"] = [float(row[label_map[pred]]) for row, pred in zip(probs, result["predicted_label_thresholded"])]
        result["threshold_mixture"] = mixture_t
        result["threshold_unproven"] = unproven_t

    # Save full JSONL and compact CSV.
    full_jsonl = PRED_DIR / "test_predictions_full.jsonl"
    result.drop(columns=["_parsed_rag_results"], errors="ignore").to_json(full_jsonl, orient="records", lines=True, force_ascii=False)
    record_output(full_jsonl, "predictions", "Full test predictions as JSONL")

    compact_cols = [
        c for c in [
            "split", "example_id", "claim", "label", "top_rag_score", "rag_top_score_computed",
            "predicted_label_argmax", "confidence_argmax", "predicted_label_thresholded", "confidence_thresholded",
            "prob_false", "prob_mixture", "prob_true", "prob_unproven",
            "threshold_mixture", "threshold_unproven",
        ] if c in result.columns
    ]
    compact_csv = PRED_DIR / "test_predictions_compact.csv"
    result[compact_cols].to_csv(compact_csv, index=False)
    record_output(compact_csv, "predictions", "Compact test predictions CSV")
    print(f"Saved predictions: {compact_csv}")

    # Evaluation tables/plots.
    if "label" in result.columns:
        metrics_parts = []
        metrics_parts.append(compute_metrics_table(result.rename(columns={"predicted_label_argmax": "pred_tmp"}), "pred_tmp", label_names, "argmax_test"))
        save_classification_outputs(result.rename(columns={"predicted_label_argmax": "pred_tmp"}), "pred_tmp", label_names, "argmax_test")
        if "predicted_label_thresholded" in result.columns:
            metrics_parts.append(compute_metrics_table(result.rename(columns={"predicted_label_thresholded": "pred_tmp"}), "pred_tmp", label_names, "thresholded_test"))
            save_classification_outputs(result.rename(columns={"predicted_label_thresholded": "pred_tmp"}), "pred_tmp", label_names, "thresholded_test")

        metrics = pd.concat(metrics_parts, ignore_index=True)
        save_table(metrics, "test_inference_metrics", "Test inference metrics from loaded model")

        # Summary line table for report.
        overall = metrics[metrics["metric_scope"].eq("overall")].copy()
        save_table(overall, "test_inference_overall_summary", "Overall test inference summary")

        # RAG score vs correctness.
        for setting, pred_col, conf_col in [
            ("argmax_test", "predicted_label_argmax", "confidence_argmax"),
            ("thresholded_test", "predicted_label_thresholded", "confidence_thresholded"),
        ]:
            if pred_col not in result.columns:
                continue
            tmp = result.copy()
            tmp["correct"] = tmp["label"].astype(str).map(flatten_label).eq(tmp[pred_col].astype(str).map(flatten_label))
            score_col = "rag_top_score_computed" if "rag_top_score_computed" in tmp.columns else "top_rag_score"
            if score_col in tmp.columns:
                rows2 = []
                for correct_value, g in tmp.groupby("correct"):
                    s = pd.to_numeric(g[score_col], errors="coerce").dropna()
                    if s.empty:
                        continue
                    rows2.append({
                        "setting": setting,
                        "correct": bool(correct_value),
                        "count": int(len(g)),
                        "mean_rag_top_score": round(float(s.mean()), 4),
                        "median_rag_top_score": round(float(s.median()), 4),
                        "pct_below_1": round(pct((s < 1).sum(), len(s)), 2),
                        "pct_below_0": round(pct((s < 0).sum(), len(s)), 2),
                    })
                save_table(pd.DataFrame(rows2), f"rag_score_by_correctness_{setting}", f"RAG score by correctness - {setting}")

                fig, ax = plt.subplots(figsize=(7, 5))
                data = [
                    pd.to_numeric(tmp.loc[tmp["correct"].eq(True), score_col], errors="coerce").dropna().values,
                    pd.to_numeric(tmp.loc[tmp["correct"].eq(False), score_col], errors="coerce").dropna().values,
                ]
                ax.boxplot(data, tick_labels=["correct", "incorrect"], showfliers=False)
                ax.set_ylabel("Top RAG score")
                ax.set_title(f"RAG score by correctness - {setting}")
                ax.grid(axis="y", alpha=0.25)
                save_fig(fig, FIG_DIR / f"rag_score_by_correctness_{setting}.png", f"RAG score by correctness - {setting}")

            if conf_col in tmp.columns:
                fig, ax = plt.subplots(figsize=(8, 5))
                for correct_value, label in [(True, "correct"), (False, "incorrect")]:
                    vals = pd.to_numeric(tmp.loc[tmp["correct"].eq(correct_value), conf_col], errors="coerce").dropna()
                    if len(vals):
                        ax.hist(vals, bins=25, alpha=0.45, label=label)
                ax.set_xlabel("Prediction confidence")
                ax.set_ylabel("Number of examples")
                ax.set_title(f"Confidence distribution - {setting}")
                ax.legend()
                ax.grid(axis="y", alpha=0.25)
                save_fig(fig, FIG_DIR / f"confidence_correct_vs_incorrect_{setting}.png", f"Confidence distribution by correctness - {setting}")

    try:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    except Exception:
        pass

    return result


# =============================================================================
# Judge columns, if present
# =============================================================================


def make_judge_eda(df: pd.DataFrame) -> None:
    if df.empty or "judge_verdict" not in df.columns:
        return
    counts = df["judge_verdict"].astype(str).value_counts().reset_index()
    counts.columns = ["judge_verdict", "count"]
    counts["percentage"] = counts["count"].map(lambda x: round(pct(x, len(df)), 2))
    save_table(counts, "judge_verdict_distribution", "Judge verdict distribution")

    pred_col = None
    for candidate in ["predicted_label", "predicted_label_argmax", "predicted_label_thresholded"]:
        if candidate in df.columns:
            pred_col = candidate
            break
    if pred_col and "label" in df.columns:
        tmp = df.copy()
        tmp["correct"] = tmp["label"].astype(str).map(flatten_label).eq(tmp[pred_col].astype(str).map(flatten_label))
        rows = []
        for verdict, g in tmp.groupby("judge_verdict"):
            rows.append({
                "judge_verdict": verdict,
                "count": int(len(g)),
                "accuracy": round(float(g["correct"].mean()), 4),
            })
        acc = pd.DataFrame(rows)
        save_table(acc, "accuracy_by_judge_verdict", "Classifier accuracy by judge verdict")
        plot_simple_bar(acc, "judge_verdict", "accuracy", FIG_DIR / "accuracy_by_judge_verdict.png", "Classifier accuracy by judge verdict", "Accuracy")


# =============================================================================
# EDA index / manifest
# =============================================================================


def write_index() -> None:
    manifest_path = OUTPUT_DIR / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(MANIFEST, f, indent=2, ensure_ascii=False)

    lines = []
    lines.append("# PubHealth EDA Output Index\n")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}\n")
    lines.append("\n## Configuration\n")
    lines.append("```text\n")
    lines.append(f"DATA_DIR = {DATA_DIR}\n")
    lines.append(f"SOURCE_DIR = {SOURCE_DIR}\n")
    lines.append(f"PAIRS_DIR = {PAIRS_DIR}\n")
    lines.append(f"PROCESSED_RAG_JSONL = {PROCESSED_RAG_JSONL}\n")
    lines.append(f"THRESHOLD_GRID_CSV = {THRESHOLD_GRID_CSV}\n")
    lines.append(f"MODEL_DIR = {MODEL_DIR}\n")
    lines.append(f"RUN_MODEL_INFERENCE = {RUN_MODEL_INFERENCE}\n")
    lines.append(f"MAX_LENGTH = {MAX_LENGTH}\n")
    lines.append(f"BEST_THRESHOLD_METRIC = {BEST_THRESHOLD_METRIC}\n")
    lines.append("```\n")

    if WARNINGS:
        lines.append("\n## Warnings\n")
        for w in WARNINGS:
            lines.append(f"- {w}\n")

    lines.append("\n## Outputs\n")
    for item in MANIFEST:
        lines.append(f"- **{item['kind']}**: `{item['path']}` — {item['description']}\n")

    save_text(OUTPUT_DIR / "EDA_INDEX.md", "".join(lines), "EDA output index")
    print(f"Saved manifest: {manifest_path}")


# =============================================================================
# Main orchestration
# =============================================================================


def main() -> None:
    ensure_output_dirs()
    print("=" * 90)
    print("PubHealth EDA + inference asset generation")
    print("=" * 90)
    print(f"Output directory: {OUTPUT_DIR.resolve()}")

    source_dfs = read_parquet_split(SOURCE_DIR, "pubhealth_source")
    pairs_dfs = read_parquet_split(PAIRS_DIR, "pubhealth_bigbio_pairs")
    rag_df = load_processed_rag(PROCESSED_RAG_JSONL)

    source_all = concat_splits(source_dfs)
    pairs_all = concat_splits(pairs_dfs)

    make_dataset_overview(source_dfs, pairs_dfs, rag_df)
    make_schema_tables(source_dfs, pairs_dfs, rag_df)
    make_missing_duplicate_tables(source_all, rag_df)
    make_label_eda(source_all, rag_df)
    make_length_eda(source_all, rag_df)
    make_top_terms_eda(source_all)
    rag_with_features = make_rag_eda(rag_df)
    mixture_t, unproven_t, threshold_grid = make_threshold_grid_eda()
    prediction_df = run_model_inference(rag_with_features, mixture_t, unproven_t)

    # Judge EDA if any loaded dataset already contains judge columns.
    make_judge_eda(rag_with_features)
    make_judge_eda(prediction_df)

    # Save a compact project summary for quick report update.
    summary_lines = []
    summary_lines.append("# Quick report notes generated by EDA script\n\n")
    summary_lines.append("Use the CSV tables for exact values. Important output files:\n\n")
    summary_lines.append("- `tables/label_distribution_source_by_split.csv`\n")
    summary_lines.append("- `tables/token_truncation_summary.csv`\n")
    summary_lines.append("- `tables/reformulation_diagnostics_from_slides.csv`\n")
    summary_lines.append("- `tables/chunking_fix_diagnostics_from_slides.csv`\n")
    summary_lines.append("- `tables/rag_score_summary_current_processed.csv`\n")
    summary_lines.append("- `tables/threshold_grid_top20.csv`\n")
    summary_lines.append("- `tables/test_inference_overall_summary.csv`\n")
    summary_lines.append("- `figures/confusion_matrix_argmax_test.png` and `figures/confusion_matrix_thresholded_test.png`\n")
    summary_lines.append("- `predictions/test_predictions_compact.csv`\n")
    save_text(OUTPUT_DIR / "REPORT_UPDATE_NOTES.md", "".join(summary_lines), "Quick report update notes")

    write_index()
    print("\nDONE. Nothing important only lives in the terminal; outputs are saved in EDA/.")


if __name__ == "__main__":
    ensure_output_dirs()
    log_path = LOG_DIR / f"run_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    try:
        with tee_to_log(log_path):
            main()
    except Exception:
        # Save full traceback so failure details are not lost.
        err = traceback.format_exc()
        with open(LOG_DIR / "last_error_traceback.txt", "w", encoding="utf-8") as f:
            f.write(err)
        print(err)
        raise
