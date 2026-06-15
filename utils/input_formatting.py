
import json


def build_model_input(
    example,
    max_chunks=None,
    min_chunk_score=None,
):
    """
    Builds classifier input from the reformatted RAG dataset.

    max_chunks:
        Maximum number of retrieved chunks to include.
        None = include all available chunks.

    min_chunk_score:
        Minimum score required to include a chunk.
        None = no score filtering.
        Example: 0 removes negative-score chunks.
    """

    claim = str(example.get("claim", "")).strip()

    rag_query = str(
        example.get(
            "rag_query",
            example.get("reformulated_query", claim),
        )
    ).strip()

    chunks = example.get("retrieved_chunks", None)

    # Backward compatibility: if retrieved_chunks does not exist,
    # fall back to rag_results.
    if chunks is None:
        chunks = example.get("rag_results", [])

    # If loaded from CSV or pandas, retrieved_chunks may be a JSON string.
    if isinstance(chunks, str):
        try:
            chunks = json.loads(chunks)
        except Exception:
            chunks = []

    if not isinstance(chunks, list):
        chunks = []

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

        if min_chunk_score is not None and score is not None:
            if score < min_chunk_score:
                continue

        clean_chunks.append({
            "text": text,
            "score": score,
            "rank": chunk.get("rank", None),
            "source": chunk.get("source", chunk.get("metadata", {}).get("source", None)),
        })

    clean_chunks.sort(
        key=lambda c: c["score"] if c["score"] is not None else float("-inf"),
        reverse=True,
    )

    if max_chunks is not None:
        clean_chunks = clean_chunks[:max_chunks]

    context_parts = []

    for i, chunk in enumerate(clean_chunks, start=1):
        score = chunk.get("score", None)
        source = chunk.get("source", None)

        if score is None:
            header = f"[CHUNK {i}]"
        else:
            header = f"[CHUNK {i} | score={score:.4f}]"

        if source:
            header += f" | source={source}"

        context_parts.append(
            f"{header}\n{chunk['text']}"
        )

    if context_parts:
        rag_context = "\n\n".join(context_parts)
    else:
        rag_context = "No reliable retrieved evidence was available."

    return (
        "Original claim:\n"
        f"{claim}\n\n"
        "Search query:\n"
        f"{rag_query}\n\n"
        "Retrieved evidence:\n"
        f"{rag_context}"
    )
