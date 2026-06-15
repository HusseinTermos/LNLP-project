import json
import os
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from utils import (
    build_dataset,
    get_label_id,
    limit_df,
    load_config,
    load_processed_dataset,
    oversample_minority_classes,
)


# =============================================================================
# Small helpers
# =============================================================================


def _id_to_label(label_map: dict[str, int]) -> dict[int, str]:
    return {idx: label for label, idx in label_map.items()}


def _get_dataset_path(cfg: dict[str, Any]) -> str:
    data_path = cfg.get("data", {}).get("processed_dataset_path")
    train_path = cfg.get("training", {}).get("processed_dataset_path")

    if train_path:
        return train_path
    
    if data_path:
        return data_path


def _infer_architecture(model_name: str, model_cfg: dict[str, Any]) -> str:
    architecture = str(model_cfg.get("architecture", "auto")).strip().lower()

    if architecture and architecture != "auto":
        return architecture

    if "longformer" in model_name.lower():
        return "longformer"

    return "bert"



def add_label_ids(records: pd.DataFrame, label_map: dict[str, int]) -> pd.DataFrame:
    fixed_records = []

    for record in records.iloc:
        if record.get("label") is None:
            continue

        record = dict(record)
        record["label_id"] = get_label_id(record["label"], label_map)
        fixed_records.append(record)

    return pd.DataFrame(fixed_records)


# =============================================================================
# Metrics
# =============================================================================

def compute_metrics(eval_pred) -> dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    accuracy = float((preds == labels).mean())
    macro_f1 = float(f1_score(labels, preds, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(labels, preds, average="weighted", zero_division=0))

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }

# =============================================================================
# TrainingArguments compatibility
# =============================================================================


def make_training_args(train_cfg: dict[str, Any], has_validation: bool) -> TrainingArguments:
    eval_strategy_value = "epoch" if has_validation else "no"
    save_strategy_value = "epoch" if has_validation else train_cfg.get("save_strategy", "steps")

    kwargs = {
        "output_dir": train_cfg["output_dir"],
        "num_train_epochs": train_cfg["epochs"],
        "per_device_train_batch_size": train_cfg["batch_size"],
        "per_device_eval_batch_size": train_cfg["batch_size"],
        "gradient_accumulation_steps": train_cfg["gradient_accumulation_steps"],
        "learning_rate": train_cfg["learning_rate"],
        "weight_decay": train_cfg["weight_decay"],
        "logging_steps": train_cfg["logging_steps"],
        "save_steps": train_cfg.get("save_steps", 500),
        "save_strategy": save_strategy_value,
        "report_to": "none",
        "load_best_model_at_end": bool(has_validation),
        "label_smoothing_factor": train_cfg.get("label_smoothing_factor", 0.05),
        "eval_strategy": eval_strategy_value
    }

    if has_validation:
        kwargs.update(
            {
                "metric_for_best_model": "weighted_f1",
                "greater_is_better": True,
            }
        )

    return TrainingArguments(**kwargs)


# =============================================================================
# Main training
# =============================================================================


def train_classifier_from_config(cfg: dict[str, Any]):
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    model_cfg = cfg["model"]

    dataset_path = _get_dataset_path(cfg)
    splits = load_processed_dataset(dataset_path)

    label_map = model_cfg["label_map"]
    id_to_label = _id_to_label(label_map)

    input_mode = model_cfg.get("input_mode", "claim_query_rag")
    max_length = int(model_cfg["max_length"])
    model_name = model_cfg["model_name"]
    architecture = _infer_architecture(model_name, model_cfg)

    train_records_raw = limit_df(splits["train"], max_rows=data_cfg["max_examples_per_split"])
    val_records_raw = limit_df(splits["validation"], max_rows=data_cfg["max_examples_per_split"])

    train_records = add_label_ids(train_records_raw, label_map)
    val_records = add_label_ids(val_records_raw, label_map)
    print("Train label distribution before oversampling:")
    print(Counter(str(r["label"]).strip().lower() for r in train_records.iloc))

    if train_cfg.get("oversample_minority_classes", False):
        train_records = oversample_minority_classes(
            train_records,
            label_col="label",
            random_state=train_cfg.get("seed", 42),
        )

    print("Loaded processed records:")
    print(f"  train:      {len(train_records)}")
    print(f"  validation: {len(val_records)}")

    print("Train label distribution:")
    print(Counter(str(r["label"]).strip().lower() for r in train_records.iloc))

    if len(train_records) == 0:
        raise ValueError("No training records found after adding label IDs.")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model_kwargs = {
        "num_labels": len(label_map),
        "id2label": id_to_label,
        "label2id": label_map,
    }
    if architecture == "longformer":
        model_kwargs["use_safetensors"] = True
    elif "use_safetensors" in model_cfg:
        model_kwargs["use_safetensors"] = False

    model = AutoModelForSequenceClassification.from_pretrained(model_name, **model_kwargs)

    train_dataset = build_dataset(train_records, tokenizer, max_length, input_mode, train_cfg)
    val_dataset = build_dataset(val_records, tokenizer, max_length, input_mode, train_cfg) if len(val_records) > 0 else None

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = make_training_args(
        train_cfg=train_cfg,
        has_validation=val_dataset is not None,
    )

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": val_dataset if val_dataset is not None else None,
        "processing_class": tokenizer,
        "data_collator": data_collator,
        "compute_metrics": compute_metrics if val_dataset is not None else None,
        "callbacks": [
            EarlyStoppingCallback(
                early_stopping_patience=train_cfg.get("early_stopping_patience", 1)
            )
        ] if val_dataset is not None else None,
    }

    trainer = Trainer(**trainer_kwargs)

    trainer.train()

    max_input_length = getattr(train_dataset, "max_input_length", None)
    print(f"MAX LENGTH: {max_input_length}", flush=True)

    os.makedirs(train_cfg["output_dir"], exist_ok=True)
    trainer.save_model(train_cfg["output_dir"])
    tokenizer.save_pretrained(train_cfg["output_dir"])

    label_map_path = os.path.join(train_cfg["output_dir"], "label_map.json")
    with open(label_map_path, "w", encoding="utf-8") as f:
        # Same normal behavior as train_longformer.py: save the plain label_map.
        json.dump(label_map, f, indent=2, ensure_ascii=False)

    print(f"Saved model to: {train_cfg['output_dir']}")
    print(f"Saved label map to: {label_map_path}")

    return {
        "trainer": trainer,
        "model": model,
        "tokenizer": tokenizer,
        "label_map": label_map,
        "train_records": train_records,
        "val_records": val_records,
    }


def train_classifier_from_config_path(config_path: str):
    cfg = load_config(config_path)
    return train_classifier_from_config(cfg)
