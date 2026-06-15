"""
run_inference.py

Modes:
  --mode rag      : claim + RAG context (default)
  --mode no_rag   : claim only

Class space:
  --num-classes 4 : false / mixture / true / unproven (default)
  --num-classes 3 : false / true / uncertain  (mixture+unproven remapped for eval)

Ensemble:
  --ensemble      : average softmax probs across multiple checkpoints
  --checkpoints   : checkpoint subdirs within --model-dir

Usage:
  python run_inference.py --dataset data/processed/FULL_no_reform.jsonl
  python run_inference.py --dataset ... --mode no_rag --model-dir models/scibert_no_rag_santi
  python run_inference.py --dataset ... --num-classes 3 --model-dir models/longformer_3class_santi
  python run_inference.py --dataset ... --ensemble --checkpoints checkpoint-1269 checkpoint-2539
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from utils import load_processed_dataset

DEFAULT_LABEL_MAP_4 = {"false": 0, "mixture": 1, "true": 2, "unproven": 3}
DEFAULT_LABEL_MAP_3 = {"false": 0, "true": 1, "uncertain": 2}
MERGE_TO_UNCERTAIN  = {"mixture", "unproven"}

DEFAULT_MODEL_DIR  = "models/scibert_santi"
DEFAULT_MAX_LENGTH = 512
DEFAULT_BATCH_SIZE = 16


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _load_label_map(model_dir: str, num_classes: int) -> dict:
    p = Path(model_dir) / "label_map.json"
    if p.exists():
        with open(p) as f:
            data = json.load(f)
        if "label2id" in data:
            return data["label2id"]
        if all(isinstance(v, int) for v in data.values()):
            return data
    return DEFAULT_LABEL_MAP_4 if num_classes == 4 else DEFAULT_LABEL_MAP_3


def _build_input(example: dict, mode: str) -> str:
    claim = str(example.get("claim", "")).strip()
    if mode == "no_rag":
        return f"Original claim:\n{claim}"
    query   = str(example.get("rag_query", example.get("reformulated_query", claim))).strip()
    context = str(example.get("rag_context", "")).strip() or "No evidence retrieved."
    return (
        f"Original claim:\n{claim}\n\n"
        f"Search query:\n{query}\n\n"
        f"Retrieved evidence:\n{context}"
    )


def _run_model(
    model_dir: str,
    texts: list[str],
    batch_size: int,
    max_length: int,
    device: torch.device,
    desc: str = "Inference",
) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model     = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval().to(device)

    is_longformer = "longformer" in model.config.model_type.lower()
    all_probs     = []

    for start in tqdm(range(0, len(texts), batch_size), desc=desc, leave=False):
        batch = texts[start: start + batch_size]
        enc   = tokenizer(batch, truncation=True, max_length=max_length,
                          padding=True, return_tensors="pt")
        if is_longformer:
            global_attn       = torch.zeros_like(enc["input_ids"])
            global_attn[:, 0] = 1
            enc["global_attention_mask"] = global_attn
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            logits = model(**enc).logits

        all_probs.append(torch.softmax(logits, dim=-1).cpu().numpy())

    del model
    torch.cuda.empty_cache()
    return np.vstack(all_probs)


def _classify(
    rows: list[dict],
    model_dir: str,
    label_map: dict,
    batch_size: int,
    max_length: int,
    mode: str,
    device: torch.device,
    checkpoints: list[str] | None = None,
) -> tuple[list[str], list[float], dict[str, list[float]]]:
    id2label    = {v: k for k, v in label_map.items()}
    label_names = [id2label[i] for i in sorted(id2label)]
    texts       = [_build_input(ex, mode) for ex in rows]

    if checkpoints:
        accumulated = np.zeros((len(texts), len(label_map)), dtype=np.float64)
        for ckpt in checkpoints:
            ckpt_dir = str(Path(model_dir) / ckpt) if ckpt else model_dir
            print(f"  Checkpoint: {ckpt_dir}")
            accumulated += _run_model(ckpt_dir, texts, batch_size, max_length, device,
                                      desc=f"  {Path(ckpt_dir).name or 'top-level'}")
        avg_probs = accumulated / len(checkpoints)
    else:
        avg_probs = _run_model(model_dir, texts, batch_size, max_length, device)

    pred_ids    = avg_probs.argmax(axis=1)
    pred_labels = [id2label[i] for i in pred_ids]
    confidences = [float(avg_probs[i, pred_ids[i]]) for i in range(len(pred_ids))]
    probs       = {name: avg_probs[:, label_map[name]].tolist() for name in label_names}

    return pred_labels, confidences, probs


def _print_stats(df: pd.DataFrame, label_names: list[str], num_classes: int) -> None:
    if "label" not in df.columns:
        print("No gold labels — skipping evaluation.")
        return

    y_pred = df["predicted_label"].astype(str)
    if num_classes == 3:
        y_true = df["label"].astype(str).apply(
            lambda l: "uncertain" if l.lower() in MERGE_TO_UNCERTAIN else l.lower()
        )
        print("(Gold labels remapped: mixture/unproven → uncertain)")
    else:
        y_true = df["label"].astype(str)

    print(f"\nTotal     : {len(df)}")
    print(f"Accuracy  : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Macro F1  : {f1_score(y_true, y_pred, labels=label_names, average='macro', zero_division=0):.4f}")
    print(f"Weighted F1: {f1_score(y_true, y_pred, labels=label_names, average='weighted', zero_division=0):.4f}")
    print(classification_report(y_true, y_pred, labels=label_names, zero_division=0, digits=4))
    cm    = confusion_matrix(y_true, y_pred, labels=label_names)
    cm_df = pd.DataFrame(cm,
                         index   =[f"true_{x}" for x in label_names],
                         columns =[f"pred_{x}" for x in label_names])
    print("Confusion matrix (rows=true, cols=predicted):")
    print(cm_df.to_string())


def run_inference(
    jsonl_path: str,
    model_dir: str = DEFAULT_MODEL_DIR,
    mode: str = "rag",
    num_classes: int = 4,
    ensemble: bool = False,
    checkpoints: list[str] | None = None,
    output_path: str | None = None,
    splits: list[str] | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> pd.DataFrame:
    ts = _timestamp()

    if output_path is None:
        stem        = Path(jsonl_path).stem
        tag         = f"{'ensemble_' if ensemble else ''}{mode}_{num_classes}class"
        output_path = Path(jsonl_path).parent / f"{stem}_{tag}_{ts}.csv"
    else:
        p           = Path(output_path)
        output_path = p.with_name(f"{p.stem}_{ts}{p.suffix}")

    split_dfs = load_processed_dataset(jsonl_path)
    if not split_dfs:
        raise ValueError(f"No data found in {jsonl_path}")

    if splits:
        missing = set(splits) - set(split_dfs)
        if missing:
            raise ValueError(f"Splits not found: {missing}. Available: {list(split_dfs)}")
        split_dfs = {k: v for k, v in split_dfs.items() if k in splits}

    df   = pd.concat(split_dfs.values(), ignore_index=True)
    rows = df.to_dict("records")
    print(f"Total examples: {len(df)} | splits: {list(split_dfs)}")

    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_map   = _load_label_map(model_dir, num_classes)
    id2label    = {v: k for k, v in label_map.items()}
    label_names = [id2label[i] for i in sorted(id2label)]

    print(f"Device: {device} | mode: {mode} | classes: {num_classes} | ensemble: {ensemble}\n")

    ckpts = (checkpoints or [""]) if ensemble else None
    pred_labels, confidences, probs = _classify(
        rows, model_dir, label_map, batch_size, max_length, mode, device, ckpts
    )

    df["predicted_label"] = pred_labels
    df["confidence"]      = confidences
    for name in label_names:
        df[f"prob_{name}"] = probs[name]

    _print_stats(df, label_names, num_classes)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")

    return df


def main():
    parser = argparse.ArgumentParser(description="PubHealth fact-checking inference pipeline.")
    parser.add_argument("--dataset",     required=True)
    parser.add_argument("--model-dir",   default=DEFAULT_MODEL_DIR)
    parser.add_argument("--mode",        default="rag", choices=["rag", "no_rag"])
    parser.add_argument("--num-classes", default=4, type=int, choices=[3, 4])
    parser.add_argument("--ensemble",    action="store_true")
    parser.add_argument("--checkpoints", nargs="+", default=None)
    parser.add_argument("--output",      default=None)
    parser.add_argument("--splits",      nargs="+", default=None)
    parser.add_argument("--batch-size",  type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-length",  type=int, default=DEFAULT_MAX_LENGTH)
    args = parser.parse_args()

    run_inference(
        jsonl_path  = args.dataset,
        model_dir   = args.model_dir,
        mode        = args.mode,
        num_classes = args.num_classes,
        ensemble    = args.ensemble,
        checkpoints = args.checkpoints,
        output_path = args.output,
        splits      = args.splits,
        batch_size  = args.batch_size,
        max_length  = args.max_length,
    )


if __name__ == "__main__":
    main()
