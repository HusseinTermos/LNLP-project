import re
import uuid
from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient, models
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder


@dataclass
class Chunk:
    id: str
    text: str
    metadata: dict[str, Any]


class RAG:
    def __init__(
        self,
        document: str,
        qdrant_path: str | None = None,
        chunk_size: int = 250,
        chunk_overlap: int = 60,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        cross_encoder_model_name: str = "cross-encoder/ms-marco-TinyBERT-L2-v2",
        reset_collection: bool = True,
    ):
        """
        document:
            Initial text to index.

        qdrant_path:
            None      -> in-memory Qdrant
            "rag_db"  -> local persistent Qdrant folder

        chunk_size:
            Number of words per chunk.

        chunk_overlap:
            Number of overlapping words between consecutive chunks.

        embedding_model_name:
            Small sentence-transformer model for dense semantic retrieval.

        cross_encoder_model_name:
            Small cross-encoder model for reranking BM25 + bi-encoder candidates.

        reset_collection:
            True  -> delete and rebuild the Qdrant collection on initialization.
            False -> reuse existing collection if it exists.

            For this simple version, True is safer because we are not loading
            existing Qdrant points back into self.chunks/BM25.
        """
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")

        self.collection_name = "rag_vectors"
        self.vector_name = "dense"
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.model = SentenceTransformer(embedding_model_name)
        self.cross_encoder = CrossEncoder(cross_encoder_model_name)

        self.embedding_dim = self.model.get_embedding_dimension()
        # if self.embedding_dim is None:
        #     test_vector = self.model.encode(["test"], normalize_embeddings=True)[0]
        #     self.embedding_dim = len(test_vector)

        self.client = QdrantClient(":memory:" if qdrant_path is None else qdrant_path)

        self.chunks: list[Chunk] = []
        self.bm25: BM25Okapi | None = None
        self.tokenized_corpus: list[list[str]] = []

        self._create_or_reset_dense_collection(reset_collection=reset_collection)

        if document.strip():
            self.add_text(
                document,
                source="initial_document",
                temporary=False,
            )

    def retrieve_bm25(
        self,
        query: str,
        top_k: int = 5,
        temporary_text: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Retrieve top_k chunks using BM25 keyword retrieval.

        temporary_text:
            Optional text inserted only for this retrieval call.
            It is removed immediately afterward.
        """
        if temporary_text:
            temp_ids = self.add_text(
                temporary_text,
                source="temporary_query_context",
                temporary=True,
            )

            try:
                return self.retrieve_bm25(query=query, top_k=top_k)
            finally:
                self.remove(temp_ids)

        if self.bm25 is None or not self.chunks:
            return []

        query_tokens = self._tokenize(query)

        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)

        ranked = sorted(
            enumerate(scores),
            key=lambda pair: float(pair[1]),
            reverse=True,
        )

        results = []
        for index, score in ranked:
            score = float(score)

            # BM25 score 0 usually means no useful lexical match.
            if score <= 0:
                continue

            chunk = self.chunks[index]
            results.append({
                "id": chunk.id,
                "score": score,
                "text": chunk.text,
                "metadata": chunk.metadata,
                "method": "bm25",
            })

            if len(results) >= top_k:
                break

        return results

    def retrieve_biencoder(
        self,
        query: str,
        top_k: int = 5,
        temporary_text: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Retrieve top_k chunks using dense semantic retrieval.

        This uses a sentence-transformer bi-encoder:
            query -> vector
            chunk -> vector
            Qdrant cosine search over chunk vectors
        """
        if temporary_text:
            temp_ids = self.add_text(
                temporary_text,
                source="temporary_query_context",
                temporary=True,
            )

            try:
                return self.retrieve_biencoder(query=query, top_k=top_k)
            finally:
                self.remove(temp_ids)

        if not self.chunks:
            return []

        query_vector = self.model.encode(
            [query],
            normalize_embeddings=True,
        )[0].tolist()

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            using=self.vector_name,
            limit=top_k,
            with_payload=True,
        )

        results = []
        for point in response.points:
            payload = point.payload or {}

            results.append(
                {
                    "id": str(point.id),
                    "score": float(point.score),
                    "text": payload.get("text", ""),
                    "metadata": {
                        key: value
                        for key, value in payload.items()
                        if key != "text"
                    },
                    "method": "biencoder",
                }
            )

        return results

    def retrieve_hybrid_candidates(
        self,
        query: str,
        bm25_k: int = 30,
        dense_k: int = 30,
        temporary_text: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Retrieve candidates from both BM25 and bi-encoder,
        then merge and deduplicate them by chunk ID.

        This does NOT rerank yet. It only creates the candidate pool
        that the cross-encoder will score.
        """
        if temporary_text:
            temp_ids = self.add_text(
                temporary_text,
                source="temporary_query_context",
                temporary=True,
            )

            try:
                return self.retrieve_hybrid_candidates(
                    query=query,
                    bm25_k=bm25_k,
                    dense_k=dense_k,
                )
            finally:
                self.remove(temp_ids)

        bm25_results = self.retrieve_bm25(query=query, top_k=bm25_k)
        dense_results = self.retrieve_biencoder(query=query, top_k=dense_k)

        merged: dict[str, dict[str, Any]] = {}

        for result in bm25_results:
            item = result.copy()
            item["retrieved_by"] = {"bm25"}
            item["bm25_score"] = item["score"]
            item["dense_score"] = None
            item["method"] = "hybrid_candidate"
            merged[item["id"]] = item

        for result in dense_results:
            if result["id"] in merged:
                merged[result["id"]]["retrieved_by"].add("biencoder")
                merged[result["id"]]["dense_score"] = result["score"]
            else:
                item = result.copy()
                item["retrieved_by"] = {"biencoder"}
                item["bm25_score"] = None
                item["dense_score"] = item["score"]
                item["method"] = "hybrid_candidate"
                merged[item["id"]] = item

        candidates = list(merged.values())

        for candidate in candidates:
            candidate["retrieved_by"] = sorted(candidate["retrieved_by"])

        return candidates

    def retrieve_cross_encoder(
        self,
        query: str,
        top_k: int = 5,
        bm25_k: int = 30,
        dense_k: int = 30,
        temporary_text: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Retrieve candidates using BM25 + bi-encoder,
        then rerank those candidates using a cross-encoder.

        Final result:
            top_k chunks ranked by cross-encoder relevance score.
        """
        candidates = self.retrieve_hybrid_candidates(
            query=query,
            bm25_k=bm25_k,
            dense_k=dense_k,
            temporary_text=temporary_text,
        )

        if not candidates:
            return []

        pairs = [
            [query, candidate["text"]]
            for candidate in candidates
        ]

        scores = self.cross_encoder.predict(pairs)

        reranked = []
        for candidate, score in zip(candidates, scores):
            item = candidate.copy()
            item["score"] = float(score)
            item["cross_encoder_score"] = float(score)
            item["method"] = "cross_encoder"
            reranked.append(item)

        reranked.sort(
            key=lambda item: item["cross_encoder_score"],
            reverse=True,
        )

        return reranked[:top_k]

    def query(
        self,
        query: str,
        top_k: int = 5,
        temporary_text: str | None = None,
        method: str = "cross_encoder",
    ) -> list[dict[str, Any]]:
        """
        Generic retrieval method.
        """
        if method == "bm25":
            return self.retrieve_bm25(
                query=query,
                top_k=top_k,
                temporary_text=temporary_text,
            )

        elif method == "biencoder":
            return self.retrieve_biencoder(
                query=query,
                top_k=top_k,
                temporary_text=temporary_text,
            )

        else:
            return self.retrieve_cross_encoder(
                query=query,
                top_k=top_k,
                temporary_text=temporary_text,
            )

    def add_text(
        self,
        text: str,
        source: str = "added_text",
        temporary: bool = False,
    ) -> list[str]:
        """
        Add one separate text/document.
        """
        return self.add_texts(
            texts_with_sources=[(text, source)],
            temporary=temporary,
        )

    def add_texts(
        self,
        texts_with_sources: list[tuple[str, str]],
        temporary: bool = False,
    ) -> list[str]:
        """
        Add multiple separate texts without allowing chunks to cross
        evidence/document boundaries.
        """
        all_new_chunks: list[Chunk] = []

        for text, source in texts_with_sources:
            text = str(text).strip()

            if not text:
                continue

            new_chunks = self._chunk_text(
                text=text,
                source=source,
                temporary=temporary,
            )

            all_new_chunks.extend(new_chunks)

        if not all_new_chunks:
            return []

        self.chunks.extend(all_new_chunks)

        self._rebuild_bm25()
        self._upsert_dense_chunks(all_new_chunks)
        print("NUM CHUNKS:", len(all_new_chunks))
        return [chunk.id for chunk in all_new_chunks]

    def remove(self, ids: list[str]) -> None:
        """
        Remove chunks from both:
            1. BM25 index
            2. Dense Qdrant index
        """
        if not ids:
            return

        id_set = set(ids)

        self.chunks = [
            chunk
            for chunk in self.chunks
            if chunk.id not in id_set
        ]

        self._rebuild_bm25()

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.PointIdsList(points=ids),
            wait=True,
        )

    def answer(
        self,
        query: str,
        top_k: int = 5,
        method: str = "cross_encoder",
        temporary_text: str | None = None,
    ) -> str:
        
        """
        This does NOT generate a medical true/false answer.
        It only formats retrieved evidence chunks.
        """
        
        def _format_optional_score(score: float | None) -> str:
            if score is None:
                return "None"
            return f"{float(score):.4f}"
        
        results = self.query(
            query=query,
            top_k=top_k,
            method=method,
            temporary_text=temporary_text,
        )

        if not results:
            return "No relevant chunks found."

        lines = []
        for i, result in enumerate(results, start=1):
            metadata = result.get("metadata", {})
            source = metadata.get("source", "unknown")
            temporary = metadata.get("temporary", False)

            extra = ""

            if "bm25_score" in result:
                extra += f" bm25_score={_format_optional_score(result.get('bm25_score'))}"

            if "dense_score" in result:
                extra += f" dense_score={_format_optional_score(result.get('dense_score'))}"

            if "cross_encoder_score" in result:
                extra += f" cross_encoder_score={result['cross_encoder_score']:.4f}"

            if "retrieved_by" in result:
                extra += f" retrieved_by={result['retrieved_by']}"

            lines.append(
                f"[{i}] method={result['method']} "
                f"score={result['score']:.4f} "
                f"source={source} "
                f"temporary={temporary}"
                f"{extra}\n"
                f"{result['text']}"
            )

        return "\n\n".join(lines)

    def _create_or_reset_dense_collection(self, reset_collection: bool) -> None:
        exists = self.client.collection_exists(self.collection_name)

        if exists and reset_collection:
            self.client.delete_collection(self.collection_name)
            exists = False

        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    self.vector_name: models.VectorParams(
                        size=self.embedding_dim,
                        distance=models.Distance.COSINE,
                    )
                },
            )

    def _split_into_sentences(self, text: str) -> list[str]:
        """
        Simple sentence splitter.

        Keeps this dependency-free. Good enough for PubHealth-style text.
        """
        text = re.sub(r"\s+", " ", str(text)).strip()

        if not text:
            return []

        sentences = re.split(r"(?<=[.!?])\s+", text)

        return [
            sentence.strip()
            for sentence in sentences
            if sentence.strip()
        ]
    
    def _word_count(self, text: str) -> int:
        return len(re.findall(r"\S+", text))


    def _make_chunk(
        self,
        text: str,
        source: str,
        temporary: bool,
        local_index: int,
        start_sentence: int,
        end_sentence: int,
        word_count: int,
        chunking_method: str = "sentence_window",
        extra_metadata: dict | None = None,
    ) -> Chunk:
        metadata = {
            "source": source,
            "temporary": temporary,
            "chunking_method": chunking_method,
            "local_chunk_index": local_index,
            "start_sentence": start_sentence,
            "end_sentence": end_sentence,
            "word_count": word_count,
        }

        if extra_metadata:
            metadata.update(extra_metadata)

        return Chunk(
            id=str(uuid.uuid4()),
            text=text.strip(),
            metadata=metadata,
        )


    def _flush_sentence_chunk(
        self,
        current_sentences: list[tuple[int, str, int]],
        source: str,
        temporary: bool,
        local_index: int,
    ) -> Chunk | None:
        if not current_sentences:
            return None

        chunk_text = " ".join(sentence for _, sentence, _ in current_sentences)
        word_count = sum(count for _, _, count in current_sentences)

        return self._make_chunk(
            text=chunk_text,
            source=source,
            temporary=temporary,
            local_index=local_index,
            start_sentence=current_sentences[0][0],
            end_sentence=current_sentences[-1][0],
            word_count=word_count,
        )


    def _split_long_sentence(
        self,
        sentence: str,
        sentence_index: int,
        source: str,
        temporary: bool,
        local_index: int,
    ) -> tuple[list[Chunk], int]:
        words = re.findall(r"\S+", sentence)
        step = max(1, self.chunk_size - self.chunk_overlap)

        chunks: list[Chunk] = []

        for start_word in range(0, len(words), step):
            end_word = min(start_word + self.chunk_size, len(words))
            chunk_text = " ".join(words[start_word:end_word])

            chunks.append(
                self._make_chunk(
                    text=chunk_text,
                    source=source,
                    temporary=temporary,
                    local_index=local_index,
                    start_sentence=sentence_index,
                    end_sentence=sentence_index,
                    word_count=end_word - start_word,
                    chunking_method="sentence_window_long_sentence_split",
                    extra_metadata={
                        "start_word": start_word,
                        "end_word": end_word,
                    },
                )
            )

            local_index += 1

            if end_word >= len(words):
                break

        return chunks, local_index


    def _get_overlap_sentences(
        self,
        current_sentences: list[tuple[int, str, int]],
    ) -> list[tuple[int, str, int]]:
        overlap_sentences: list[tuple[int, str, int]] = []
        overlap_word_count = 0

        for item in reversed(current_sentences):
            _, _, word_count = item

            if overlap_word_count + word_count > self.chunk_overlap:
                break

            overlap_sentences.insert(0, item)
            overlap_word_count += word_count

        return overlap_sentences


    def _chunk_text(
        self,
        text: str,
        source: str,
        temporary: bool,
    ) -> list[Chunk]:
        """
        Sentence-window chunking.

        Uses chunk_size as the maximum number of words per chunk.
        Uses chunk_overlap approximately as overlapping words, but overlap
        is applied at the sentence level.

        Normal chunks preserve sentence boundaries. A sentence is only split
        if the sentence itself is longer than chunk_size.
        """
        sentences = self._split_into_sentences(text)

        if not sentences:
            return []

        chunks: list[Chunk] = []
        current_sentences: list[tuple[int, str, int]] = []
        current_word_count = 0
        local_index = 0

        i = 0

        while i < len(sentences):
            sentence = sentences[i]
            sentence_word_count = self._word_count(sentence)

            # Case 1: single sentence is longer than chunk_size.
            # Save the current chunk first, then split the long sentence.
            if sentence_word_count > self.chunk_size:
                flushed_chunk = self._flush_sentence_chunk(
                    current_sentences=current_sentences,
                    source=source,
                    temporary=temporary,
                    local_index=local_index,
                )

                if flushed_chunk is not None:
                    chunks.append(flushed_chunk)
                    local_index += 1

                current_sentences = []
                current_word_count = 0

                long_sentence_chunks, local_index = self._split_long_sentence(
                    sentence=sentence,
                    sentence_index=i,
                    source=source,
                    temporary=temporary,
                    local_index=local_index,
                )

                chunks.extend(long_sentence_chunks)
                i += 1
                continue

            # Case 2: adding this sentence would exceed chunk_size.
            # Save current chunk, keep sentence-level overlap, then retry
            # the same sentence.
            if current_sentences and current_word_count + sentence_word_count > self.chunk_size:
                flushed_chunk = self._flush_sentence_chunk(
                    current_sentences=current_sentences,
                    source=source,
                    temporary=temporary,
                    local_index=local_index,
                )

                if flushed_chunk is not None:
                    chunks.append(flushed_chunk)
                    local_index += 1

                overlap_sentences = self._get_overlap_sentences(current_sentences)
                overlap_word_count = sum(count for _, _, count in overlap_sentences)

                # Avoid infinite loop if overlap + next sentence is still too long.
                if overlap_sentences and overlap_word_count + sentence_word_count > self.chunk_size:
                    overlap_sentences = []
                    overlap_word_count = 0

                current_sentences = overlap_sentences
                current_word_count = overlap_word_count

                continue

            # Case 3: normal sentence fits in the current chunk.
            current_sentences.append((i, sentence, sentence_word_count))
            current_word_count += sentence_word_count
            i += 1

        # Save final chunk.
        flushed_chunk = self._flush_sentence_chunk(
            current_sentences=current_sentences,
            source=source,
            temporary=temporary,
            local_index=local_index,
        )

        if flushed_chunk is not None:
            chunks.append(flushed_chunk)

        return chunks
    def _rebuild_bm25(self) -> None:
        self.tokenized_corpus = [
            self._tokenize(chunk.text)
            for chunk in self.chunks
        ]

        if not self.tokenized_corpus:
            self.bm25 = None
            return

        self.bm25 = BM25Okapi(self.tokenized_corpus)


    def _upsert_dense_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return

        embedding_batch_size = 64
        upsert_batch_size = 256

        for start in range(0, len(chunks), upsert_batch_size):
            batch_chunks = chunks[start:start + upsert_batch_size]
            texts = [chunk.text for chunk in batch_chunks]

            vectors = self.model.encode(
                texts,
                batch_size=embedding_batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )

            points = []

            for chunk, vector in zip(batch_chunks, vectors):
                payload = {
                    "text": chunk.text,
                    **chunk.metadata,
                }

                points.append(
                    models.PointStruct(
                        id=chunk.id,
                        vector={
                            self.vector_name: vector.tolist(),
                        },
                        payload=payload,
                    )
                )

            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True,
            )


    def _tokenize(self, text: str) -> list[str]:
        """
        Simple tokenizer for BM25.

        Keeps words, numbers, and basic hyphenated/apostrophe terms.
        Good enough for now.
        """
        return re.findall(
            r"\b\w+(?:[-']\w+)*\b",
            text.lower(),
        )
