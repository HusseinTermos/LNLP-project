from importlib import import_module
from inspect import signature
from itertools import product
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from inference.run_inference import run_inference as run_model


DATASET = r"data\processed\FULL_no_reform.jsonl"
MODEL_DIR = r"models\scibert_20260605_152114_fa6345e0-e3af-4933-bc23-0dbce6a63b12"

SPLITS = ["validation"]

OUTPUT_DIR = Path("data/threshold_grid")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODE = 0

if MODE == 0:
    LABELS = ["false", "unknown", "true"]

    THRESHOLD_GRID = {
        "unknown": [
            None,
            0.05,
            0.08,
            0.10,
            0.12,
            0.15,
            0.18,
            0.20,
            0.25,
            0.30,
            0.35,
            0.40,
            0.45,
            0.50,
        ]
    }

    THRESHOLD_PRIORITY = ["unknown"]
else:
    LABELS = ["false", "mixture", "true", "unproven"]

    THRESHOLD_GRID = {
        "mixture": [None, 0.45, 0.50, 0.55, 0.60, 0.65],
        "unproven": [None, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25],
    }

    THRESHOLD_PRIORITY = ["mixture", "unproven"]

def argmax_predict(df: pd.DataFrame, labels: list[str]) -> pd.Series:
    prob_cols = [f"prob_{label}" for label in labels]
    probs = df[prob_cols].to_numpy()

    pred_idx = np.argmax(probs, axis=1)

    return pd.Series([labels[i] for i in pred_idx], index=df.index)


def predict_with_thresholds(
    df: pd.DataFrame,
    labels: list[str],
    thresholds: dict[str, Optional[float]],
    threshold_priority: list[str],
) -> pd.Series:
    active_thresholds = {
        label: threshold
        for label, threshold in thresholds.items()
        if threshold is not None
    }

    if not active_thresholds:
        return argmax_predict(df, labels)

    preds = []

    thresholded_labels = set(active_thresholds.keys())
    fallback_labels = [label for label in labels if label not in thresholded_labels]

    if not fallback_labels:
        fallback_labels = labels

    for _, row in df.iterrows():
        predicted = None

        for label in threshold_priority:
            threshold = active_thresholds.get(label)

            if threshold is None:
                continue

            prob = float(row[f"prob_{label}"])

            if prob >= threshold:
                predicted = label
                break

        if predicted is None:
            fallback_probs = {
                label: float(row[f"prob_{label}"])
                for label in fallback_labels
            }

            predicted = max(fallback_probs, key=fallback_probs.get)

        preds.append(predicted)

    return pd.Series(preds, index=df.index)


def make_threshold_settings(
    threshold_grid: dict[str, list[Optional[float]]]
) -> list[dict[str, Optional[float]]]:
    labels = list(threshold_grid.keys())
    values = [threshold_grid[label] for label in labels]

    settings = []

    for combo in product(*values):
        settings.append(dict(zip(labels, combo)))

    return settings


def make_setting_name(thresholds: dict[str, Optional[float]]) -> str:
    active = {
        label: threshold
        for label, threshold in thresholds.items()
        if threshold is not None
    }

    if not active:
        return "argmax"

    parts = []

    for label, threshold in active.items():
        parts.append(f"{label}_{threshold}")

    return "__".join(parts)


def evaluate_predictions(
    y_true: pd.Series,
    y_pred: pd.Series,
    labels: list[str],
    setting_name: str,
    thresholds: dict[str, Optional[float]],
    output_path: Path,
) -> dict:
    accuracy = accuracy_score(y_true, y_pred)

    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        labels=labels,
        average="weighted",
        zero_division=0,
    )

    per_label_f1 = f1_score(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )

    row = {
        "setting": setting_name,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "output_path": str(output_path),
    }

    for label, threshold in thresholds.items():
        row[f"threshold_{label}"] = threshold

    for label, value in zip(labels, per_label_f1):
        row[f"f1_{label}"] = value

    for label in labels:
        row[f"gold_{label}"] = int((y_true == label).sum())
        row[f"pred_{label}"] = int((y_pred == label).sum())

    return row


def run_inference() -> pd.DataFrame:

    base_output_path = OUTPUT_DIR / "base_probs.csv"

    kwargs = {
        "jsonl_path": DATASET,
        "model_dir": MODEL_DIR,
        "output_path": str(base_output_path),
        "splits": SPLITS,
        "batch_size": 4,
        "max_length": 512
    }

    # Some of your inference functions accept unknown_threshold, some may not.
    # This avoids crashing when using a 2-class inference file.
    sig = signature(run_model)

    if "unknown_threshold" in sig.parameters:
        kwargs["unknown_threshold"] = None

    df = run_model(**kwargs)

    print(f"Saved base probabilities to: {base_output_path}")

    return df


def main() -> None:
    print("=" * 80)
    print("Running model inference once to get probability columns")
    print("=" * 80)

    base_df = run_inference()

    if "label" not in base_df.columns:
        raise ValueError("Dataset has no gold label column.")

    y_true = base_df["label"].astype(str)

    bad_labels = sorted(set(y_true) - set(LABELS))

    if bad_labels:
        raise ValueError(
            f"Found labels outside current setup: {bad_labels}. "
            f"Expected only: {LABELS}"
        )

    summary_rows = []
    threshold_settings = make_threshold_settings(THRESHOLD_GRID)

    for thresholds in threshold_settings:
        setting_name = make_setting_name(thresholds)

        print("=" * 80)
        print(f"Evaluating setting: {setting_name}")
        print("=" * 80)

        df = base_df.copy()

        df["predicted_label"] = predict_with_thresholds(
            df=df,
            labels=LABELS,
            thresholds=thresholds,
            threshold_priority=THRESHOLD_PRIORITY,
        )

        y_pred = df["predicted_label"].astype(str)

        output_path = OUTPUT_DIR / f"preds_{setting_name}.csv"
        df.to_csv(output_path, index=False)

        summary_rows.append(
            evaluate_predictions(
                y_true=y_true,
                y_pred=y_pred,
                labels=LABELS,
                setting_name=setting_name,
                thresholds=thresholds,
                output_path=output_path,
            )
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["macro_f1", "weighted_f1", "accuracy"],
        ascending=False,
    )

    summary_path = OUTPUT_DIR / "threshold_grid_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\nBest threshold settings:")
    print(summary_df.head(20).to_string(index=False))

    print(f"\nSaved summary to: {summary_path}")


if __name__ == "__main__":
    main()