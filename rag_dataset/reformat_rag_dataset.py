import argparse
import json
from pathlib import Path
from typing import Any


def safe_float(value):
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def load_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def normalize_rag_results(row: dict[str, Any], max_chunks: int | None = None):
    raw_results = row.get("rag_results", [])

    if isinstance(raw_results, str):
        try:
            raw_results = json.loads(raw_results)
        except Exception:
            raw_results = []

    if not isinstance(raw_results, list):
        raw_results = []

    chunks = []

    for result in raw_results:
        if not isinstance(result, dict):
            continue

        text = str(result.get("text", "")).strip()

        if not text:
            continue

        metadata = result.get("metadata", {}) or {}

        score = safe_float(
            result.get(
                "cross_encoder_score",
                result.get("score", None),
            )
        )

        chunk = {
            "rank": None,
            "text": text,
            "score": score,
            "cross_encoder_score": safe_float(result.get("cross_encoder_score", score)),
            "bm25_score": safe_float(result.get("bm25_score", None)),
            "dense_score": safe_float(result.get("dense_score", None)),
            "method": result.get("method", None),
            "retrieved_by": result.get("retrieved_by", None),
            "source": metadata.get("source", None),
            "chunking_method": metadata.get("chunking_method", None),
            "local_chunk_index": metadata.get("local_chunk_index", None),
            "word_count": len(text.split()),
        }

        chunks.append(chunk)

    chunks.sort(
        key=lambda c: c["score"] if c["score"] is not None else float("-inf"),
        reverse=True,
    )

    for i, chunk in enumerate(chunks, start=1):
        chunk["rank"] = i

    if max_chunks is not None:
        chunks = chunks[:max_chunks]

    return chunks


def build_static_context(chunks):
    parts = []

    for chunk in chunks:
        score = chunk.get("score", None)

        if score is None:
            header = f"[CHUNK {chunk['rank']}]"
        else:
            header = f"[CHUNK {chunk['rank']} | score={score:.4f}]"

        parts.append(f"{header}\n{chunk['text']}")

    return "\n\n".join(parts)


def reformat_file(input_path: Path, output_path: Path, max_chunks: int | None = None):
    rows_out = []

    for row in load_jsonl(input_path):
        chunks = normalize_rag_results(row, max_chunks=max_chunks)

        top_score = chunks[0]["score"] if chunks else None

        new_row = {
            "split": row.get("split"),
            "example_id": row.get("example_id"),
            "claim": row.get("claim"),
            "evidence": row.get("evidence", None),
            "label": row.get("label"),
            "reformulated_query": row.get("reformulated_query", row.get("claim")),
            "rag_query": row.get("rag_query", row.get("reformulated_query", row.get("claim"))),
            "top_rag_score": top_score,
            "num_retrieved_chunks": len(chunks),
            "retrieved_chunks": chunks,

            # Keep this for backward compatibility, but the new dataset class
            # should use retrieved_chunks dynamically instead.
            "rag_context": build_static_context(chunks),
        }

        rows_out.append(new_row)

    write_jsonl(rows_out, output_path)

    print(f"Saved reformatted dataset to: {output_path}")
    print(f"Rows written: {len(rows_out)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Old RAG JSONL path")
    parser.add_argument("--output", required=True, help="New reformatted JSONL path")
    parser.add_argument("--max-chunks", type=int, default=None)

    args = parser.parse_args()

    reformat_file(
        input_path=Path(args.input),
        output_path=Path(args.output),
        max_chunks=args.max_chunks,
    )


if __name__ == "__main__":
    main()