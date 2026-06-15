import logging
from typing import Any
from rag_dataset.rag import RAG
from .external_corpus import load_pubmed_texts, load_wikipedia_health_texts

logger = logging.getLogger(__name__)
def build_external_health_rag_from_config(cfg: dict[str, Any]) -> RAG:
    rag_cfg = cfg["rag"]
    corpus_cfg = cfg.get("external_corpus", {})

    num_pubmed = corpus_cfg.get("num_pubmed_abstracts", 30000)
    num_wiki = corpus_cfg.get("num_wikipedia_articles", 20000)

    rag = RAG(
        document="",
        chunk_size=rag_cfg["chunk_size"],
        chunk_overlap=rag_cfg["chunk_overlap"],
        embedding_model_name=rag_cfg["embedding_model_name"],
        cross_encoder_model_name=rag_cfg["cross_encoder_model_name"],
        reset_collection=True,
    )

    pubmed_texts = load_pubmed_texts(num_pubmed)

    if pubmed_texts:
        logger.info("Indexing %d PubMed documents into RAG...", len(pubmed_texts))
        rag.add_texts(pubmed_texts)
        logger.info("PubMed indexing complete.")

    wiki_texts = load_wikipedia_health_texts(num_wiki)

    if wiki_texts:
        logger.info("Indexing %d Wikipedia documents into RAG...", len(wiki_texts))
        rag.add_texts(wiki_texts)
        logger.info("Wikipedia indexing complete.")

    logger.info("Total chunks in RAG index: %d", len(rag.chunks))

    return rag

def build_pubhealth_evidence_rag_from_config(dataframes, cfg):
    rag_cfg = cfg["rag"]

    evidence_column = "main_text"

    rag = RAG(
        document="",
        chunk_size=rag_cfg["chunk_size"],
        chunk_overlap=rag_cfg["chunk_overlap"],
        embedding_model_name=rag_cfg["embedding_model_name"],
        cross_encoder_model_name=rag_cfg["cross_encoder_model_name"],
        reset_collection=rag_cfg["reset_collection"],
    )

    texts_with_sources = []
    seen = set()

    corpus_splits = ["train", "test", "validation"]

    for split_name in corpus_splits:
        df = dataframes[split_name][:5]

        if evidence_column not in df.columns:
            raise ValueError(
                f"Evidence column '{evidence_column}' not found in {split_name}. "
                f"Available columns: {list(df.columns)}"
            )

        for i, row in df.iterrows():
            evidence = str(row[evidence_column]).strip()

            if not evidence or evidence.lower() == "nan":
                continue

            # Avoid indexing exact duplicate evidence repeatedly
            if evidence in seen:
                continue

            seen.add(evidence)

            source_id = row.get("id", f"{split_name}_{i}")
            texts_with_sources.append((evidence, f"{split_name}_{source_id}"))

    if not texts_with_sources:
        raise ValueError("RAG corpus is empty. Check the evidence column name.")

    rag.add_texts(texts_with_sources)

    return rag

def build_rag_from_config(dataframes: dict, cfg: dict[str, Any]) -> RAG:
    rag_cfg = cfg["rag"]
    knowledge_base = rag_cfg.get("knowledge_base", "pubhealth_evidence")

    if knowledge_base == "external_health":
        return build_external_health_rag_from_config(cfg)

    if knowledge_base == "pubhealth_evidence":
        return build_pubhealth_evidence_rag_from_config(dataframes, cfg)

    raise ValueError(f"Unknown RAG knowledge_base: {knowledge_base}")


def format_rag_context(rag_results):
    chunks = []

    for i, result in enumerate(rag_results, 1):
        text = result.get("text", "")
        score = result.get("score", None)

        if score is not None:
            chunks.append(f"[{i}] score={score:.4f}\n{text}")
        else:
            chunks.append(f"[{i}]\n{text}")

    return "\n\n".join(chunks)

