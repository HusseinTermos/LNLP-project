import json
from typing import Any
import pandas as pd
from torch.utils.data import Dataset

from .input_formatting import build_model_input

class RagVerificationDataset(Dataset):
    def __init__(
        self,
        examples,
        tokenizer,
        max_length=4096,
        include_labels=True,
        min_chunk_score=None,
        max_chunks=None,
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.include_labels = include_labels
        self.min_chunk_score = min_chunk_score
        self.max_chunks = max_chunks

        self.max_input_length = 0

    def __len__(self):
        return len(self.examples)

    def _token_count(self, text: str) -> int:
        return len(
            self.tokenizer(
                text,
                truncation=False,
                add_special_tokens=True,
            )["input_ids"]
        )

    def _load_chunks(self, example):
        chunks = example.get("retrieved_chunks", None)

        if chunks is None:
            chunks = example.get("rag_results", [])

        if isinstance(chunks, str):
            try:
                chunks = json.loads(chunks)
            except Exception:
                chunks = []

        if not isinstance(chunks, list):
            return []

        clean_chunks = []

        for chunk in chunks:
            if not isinstance(chunk, dict):
                continue

            text = str(chunk.get("text", "")).strip()
            if not text:
                continue

            score = chunk.get("score", chunk.get("cross_encoder_score", None))

            try:
                score = float(score) if score is not None else None
            except Exception:
                score = None

            if self.min_chunk_score is not None and score is not None:
                if score < self.min_chunk_score:
                    continue

            clean_chunks.append(
                {
                    "rank": chunk.get("rank", None),
                    "text": text,
                    "score": score,
                    "source": chunk.get(
                        "source",
                        chunk.get("metadata", {}).get("source", None),
                    ),
                }
            )

        clean_chunks.sort(
            key=lambda c: c["score"] if c["score"] is not None else float("-inf"),
            reverse=True,
        )

        if self.max_chunks is not None:
            clean_chunks = clean_chunks[: self.max_chunks]

        return clean_chunks

    def _build_dynamic_input(self, example):
        return build_model_input(example, self.max_chunks, self.min_chunk_score)
        
    def __getitem__(self, idx):
        example = self.examples.iloc[idx]

        text = self._build_dynamic_input(example)

        full_len = self._token_count(text)
        self.max_input_length = max(self.max_input_length, full_len)

        item = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
        )

        if self.include_labels:
            item["labels"] = int(example["label_id"])

        return item
    

class ClaimOnlyDataset(Dataset):
    """For true no-RAG baselines. Uses only the claim text."""

    def __init__(self, records: pd.DataFrame, tokenizer, max_length: int, include_labels: bool = True):
        self.examples = records
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.include_labels = include_labels
        self.max_input_length = 0

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        example = self.examples.iloc[idx]
        text = str(example.get("claim", "")).strip()

        item = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )

        self.max_input_length = max(self.max_input_length, len(item["input_ids"]))

        if self.include_labels:
            item["labels"] = int(example["label_id"])

        return item


def build_dataset(
    records: pd.DataFrame,
    tokenizer,
    max_length: int,
    input_mode: str,
    train_cfg: dict[str, Any],
):
    """
    Normal path uses LongformerVerificationDataset exactly like train_longformer.py.
    Despite the class name, that dataset also works for SciBERT/BERT with max_length=512.
    """
    input_mode = str(input_mode or "claim_query_rag").strip().lower()

    if input_mode == "claim_only":
        return ClaimOnlyDataset(
            records=records,
            tokenizer=tokenizer,
            max_length=max_length,
            include_labels=True,
        )

    if input_mode != "claim_query_rag":
        raise ValueError(
            f"Unsupported input_mode={input_mode!r}. Keep 'claim_query_rag' for normal RAG training "
            "or use 'claim_only' for a no-RAG baseline."
        )

    kwargs = {
        "examples": records,
        "tokenizer": tokenizer,
        "max_length": max_length,
        "include_labels": True,
        "min_chunk_score": train_cfg.get("min_chunk_score", None),
    }

    if "max_chunks" in train_cfg:
        kwargs["max_chunks"] = train_cfg.get("max_chunks")

    return RagVerificationDataset(**kwargs)

