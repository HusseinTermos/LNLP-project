import json
import logging
from pathlib import Path
from time import sleep
from typing import Any

import numpy as np

from utils import (
    build_rag_from_config,
    build_reformulater_from_config,
    limit_df,
    load_config,
    load_pubhealth_from_config
)


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = Path("configs") / "config1.json"

LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "rag_build.log"

SPLITS = ["train", "validation", "test"]


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
    ],
)

logger = logging.getLogger(__name__)



def normalize_rag_result(result: dict[str, Any], doc_number: int) -> dict[str, Any]:
    """
    Converts one raw RAG result into a stable JSON-friendly format.

    This keeps the same useful fields as rag_results, but adds doc_number
    and doc_header so the downstream training code can format the context
    later without needing a prebuilt giant string.
    """
    score = result["score"]
    text = result['text'].strip()

    if score is not None:
        doc_header = f"[DOC {doc_number} | score={score:.4f}]"
    else:
        doc_header = f"[DOC {doc_number}]"

    return {
        "doc_number": doc_number,
        "doc_header": doc_header,
        "id": result.get("id"),
        "score": score,
        "text": text,
        "metadata": result["metadata"],
        "method": result["method"],
        "retrieved_by": result["retrieved_by"],
        "bm25_score": result.get("bm25_score"),
        "dense_score": result.get("dense_score"),
        "cross_encoder_score": result.get("cross_encoder_score")
    }


def build_structured_rag_context(rag_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Builds structured context instead of one large rag_context string.

    Output format:
    [
        {
            "doc_number": 1,
            "doc_header": "[DOC 1 | score=3.0613]",
            "score": 3.0613,
            "text": "...",
            "metadata": {...},
            ...
        },
        ...
    ]

    The actual text input for the classifier should be created later from
    these complete chunks, adding as many as fit in the context window.
    """
    context_entries = []

    for doc_number, result in enumerate(rag_results, start=1):
        normalized = normalize_rag_result(result, doc_number)

        if not normalized["text"]:
            continue

        context_entries.append(normalized)

    return context_entries


def format_rag_context_for_display(rag_context: list[dict[str, Any]]) -> str:
    """
    Optional helper for debugging only.

    Do not save this as the main rag_context field if you want the JSONL
    file to stay structured. Use it only when printing examples or when a
    downstream model explicitly needs a string.
    """
    chunks = []

    for entry in rag_context:
        header = entry.get("doc_header", f"[DOC {entry.get('doc_number', '?')}]")
        text = str(entry.get("text", "")).strip()

        if text:
            chunks.append(f"{header}\n{text}")

    return "\n\n".join(chunks)


def get_top_rag_score(rag_context: list[dict[str, Any]]) -> float | None:
    scores = [
        entry["score"]
        for entry in rag_context
        if entry.get("score") is not None
    ]

    if not scores:
        return None

    return max(scores)


def get_example_id(record: dict[str, Any], split_name: str, index: int) -> str:
    return str(record.get("id", f"{split_name}_{index}"))


def log_score_summary(split_name: str, split_top_scores: list[float]) -> None:
    if not split_top_scores:
        logger.info("No RAG scores found for %s.", split_name)
        return

    scores = np.array(split_top_scores, dtype=float)

    logger.info("RAG score summary for %s:", split_name)
    logger.info("  mean: %.4f", scores.mean())
    logger.info("  median: %.4f", np.median(scores))
    logger.info("  min: %.4f", scores.min())
    logger.info("  max: %.4f", scores.max())
    logger.info("  %% below 1: %.2f%%", (scores < 1).mean() * 100)
    logger.info("  %% below 0: %.2f%%", (scores < 0).mean() * 100)


def log_weak_examples(weak_examples: list[dict[str, Any]], max_examples: int = 5) -> None:
    logger.info("  weak examples found: %d", len(weak_examples))

    for example in weak_examples[:max_examples]:
        logger.info("Weak RAG example:")
        logger.info("  score: %.4f", example["top_score"])
        logger.info("  label: %s", example["label"])
        logger.info("  claim: %s", example["claim"])
        logger.info("  top chunk: %s", example["top_chunk"][:500])


def make_output_record(
    split_name: str,
    index: int,
    record: dict[str, Any],
    original_claim: str,
    reformulated_query: str,
    rag_query: str,
    rag_context: list[dict[str, Any]],
    top_score: float | None,
) -> dict[str, Any]:
    """
    Main JSONL row format.

    rag_context is now structured, not one large string.
    rag_results is kept in the exact same structured format for compatibility.
    """
    return {
        "split": split_name,
        "example_id": get_example_id(record, split_name, index),
        "claim": original_claim,
        "reformulated_query": reformulated_query,
        "rag_query": rag_query,

        # New preferred format: structured retrieved context.
        "rag_context": rag_context,

        # Kept for compatibility. Same structured entries as rag_context.
        "rag_results": rag_context,

        "label": record.get("label", None),
        "top_rag_score": top_score,
    }


def process_split(
    split_name: str,
    df,
    rag,
    reformulater,
    cfg: dict[str, Any],
    output_file,
) -> None:
    data_cfg = cfg["data"]
    rag_cfg = cfg["rag"]
    reform_cfg = cfg["reformulation"]

    max_examples_per_split = data_cfg["max_examples_per_split"]
    top_k = rag_cfg["top_k"]
    rag_method = rag_cfg["method"]
    query_source = rag_cfg["query_source"]

    if max_examples_per_split is not None:
        df = limit_df(df, max_examples_per_split)

    records = df.to_dict("records")
    claims = [record["claim"] for record in records]

    if reform_cfg.get("enabled", True):
        logger.info("Reformulating %d claims from %s...", len(claims), split_name)
        reformulated_claims = reformulater.reformulate_batch(claims)
    else:
        reformulated_claims = claims

    logger.info("Running RAG for %s...", split_name)

    split_top_scores: list[float] = []
    weak_examples: list[dict[str, Any]] = []

    for index, record in enumerate(records):
        original_claim = str(record["claim"])
        reformulated_query = str(reformulated_claims[index])

        rag_query = original_claim if query_source == "original" else reformulated_query

        raw_rag_results = rag.query(
            query=rag_query,
            top_k=top_k,
            method=rag_method,
        )

        rag_context = build_structured_rag_context(raw_rag_results)
        top_score = max([x["score"] for x in rag_context])

        if top_score is not None:
            split_top_scores.append(top_score)

            if top_score < 1:
                top_chunk = rag_context[0]["text"]

                weak_examples.append(
                    {
                        "claim": original_claim,
                        "label": record.get("label", None),
                        "top_score": top_score,
                        "top_chunk": top_chunk,
                    }
                )

        output_record = make_output_record(
            split_name=split_name,
            index=index,
            record=record,
            original_claim=original_claim,
            reformulated_query=reformulated_query,
            rag_query=rag_query,
            rag_context=rag_context,
            top_score=top_score,
        )

        output_file.write(
            json.dumps(
                output_record,
                ensure_ascii=False,
                default=str,
            )
            + "\n"
        )

        if (index + 1) % 100 == 0:
            logger.info("%s: processed %d/%d", split_name, index + 1, len(records))

    log_score_summary(split_name, split_top_scores)
    log_weak_examples(weak_examples)


def build_dataset_from_config(cfg: dict[str, Any]) -> str:
    dataframes = load_pubhealth_from_config(cfg)

    rag = build_rag_from_config(dataframes, cfg)
    if cfg["reformulation"]["enabled"]:
        reformulater = build_reformulater_from_config(cfg)
    else:
        reformulater = None
    data_cfg = cfg["data"]
    out_path = Path(data_cfg["processed_dataset_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as output_file:
        for split_name in SPLITS:
            process_split(
                split_name=split_name,
                df=dataframes[split_name],
                rag=rag,
                reformulater=reformulater,
                cfg=cfg,
                output_file=output_file,
            )

    logger.info("Saved RAG dataset to: %s", out_path)

    return str(out_path)


def build_dataset_from_config_path(config_path: Path) -> str:
    cfg = load_config(config_path)
    return build_dataset_from_config(cfg)


if __name__ == "__main__":
    build_dataset_from_config_path(CONFIG_PATH)