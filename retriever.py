"""
retriever.py — LangChain RAG Retriever
========================================
Migrated from raw sentence-transformers + numpy → LangChain components:

  - ChunkStore         : LangChain Document + HuggingFaceEmbeddings + FAISS VectorStore
  - MetadataFilter     : Pure Python (deterministic — no embedding needed)
  - SemanticRanker     : FAISS similarity_search_with_score on filtered candidates
  - CrossEncoderReranker: LangChain CrossEncoderReranker (sentence-transformers under hood)

Architecture (unchanged from original):
  1. DETERMINISTIC METADATA FILTER  (hard rules, no guessing)
  2. BI-ENCODER SEMANTIC RANKING within filtered pool
  3. CROSS-ENCODER RERANK (Use Case 2 only)
  4. Returns top-K RetrievedChunk with full provenance

Install:
    pip install langchain langchain-community langchain-huggingface
    pip install faiss-cpu sentence-transformers
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

# ── LangChain imports ──────────────────────────────────────────────────────────
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker as LCCrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
CHUNKS_PATH         = Path("all_chunks.json")
EMBED_MODEL_NAME    = "all-MiniLM-L6-v2"
CROSS_ENCODER_NAME  = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TOP_K               = 2
BI_ENCODER_RECALL_K = 5
SCORE_THRESHOLD     = 0.20


# ══════════════════════════════════════════════════════════════════════════════
# Data models  (unchanged from original)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ParsedQuery:
    intent:           str            = "unknown"
    chunk_type:       str            = "unknown"
    chapter_number:   Optional[int]  = None
    chapter_name:     Optional[str]  = None
    activity_number:  Optional[int]  = None
    exercise_number:  Optional[float]= None
    topic:            str            = ""
    subject:          Optional[str]  = None

    @classmethod
    def from_dict(cls, d: dict) -> "ParsedQuery":
        return cls(
            intent          = d.get("intent", "unknown"),
            chunk_type      = d.get("chunk_type", "unknown"),
            chapter_number  = _safe_int(d.get("chapter_number")),
            chapter_name    = d.get("chapter_name"),
            activity_number = _safe_int(d.get("activity_number")),
            exercise_number = _safe_float(d.get("exercise_number")),
            topic           = d.get("topic", ""),
            subject         = d.get("subject"),
        )


@dataclass
class RetrievedChunk:
    chunk_id:       str
    subject:        str
    chapter_number: str
    chapter_title:  str
    section_title:  str
    chunk_type:     str
    activity_number:str
    text:           str
    score:          float
    filter_path:    str


# ══════════════════════════════════════════════════════════════════════════════
# Helpers  (unchanged from original)
# ══════════════════════════════════════════════════════════════════════════════

def _safe_int(v: Any) -> Optional[int]:
    try:
        return int(v) if v is not None else None
    except (ValueError, TypeError):
        return None

def _safe_float(v: Any) -> Optional[float]:
    try:
        return float(v) if v is not None else None
    except (ValueError, TypeError):
        return None

def _normalise_chunk_type(ct: str) -> str:
    ct = ct.lower().strip()
    if ct in {"exercise", "exercises"}:
        return "exercise"
    if ct in {"activity", "activities"}:
        return "activity"
    if ct in {"theory", "content", "text"}:
        return "theory"
    return ct

def _normalise_activity_str(v: str) -> str:
    v = v.strip()
    try:
        f = float(v)
        if f == int(f):
            return str(int(f))
        else:
            return str(f)
    except (ValueError, TypeError):
        return v


# ══════════════════════════════════════════════════════════════════════════════
# AmbiguityError
# ══════════════════════════════════════════════════════════════════════════════

class AmbiguityError(Exception):
    """Raised when the query cannot be resolved deterministically."""


# ══════════════════════════════════════════════════════════════════════════════
# ChunkStore — LangChain-backed
# Loads all chunks → wraps in LangChain Documents → builds FAISS index
# ══════════════════════════════════════════════════════════════════════════════

class ChunkStore:
    def __init__(self, chunks_path: Path, embed_model_name: str):
        log.info("Loading chunks from %s …", chunks_path)
        with open(chunks_path, encoding="utf-8") as f:
            raw: list[dict] = json.load(f)
        self.chunks: list[dict] = raw
        log.info("Loaded %d chunks.", len(self.chunks))

        # ── HuggingFace bi-encoder via LangChain ──────────────────────────────
        log.info("Loading embedding model '%s' …", embed_model_name)
        self._embeddings = HuggingFaceEmbeddings(
            model_name=embed_model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        # ── Wrap chunks as LangChain Documents ───────────────────────────────
        # All original metadata fields are stored in Document.metadata
        # so MetadataFilter can access them by index
        log.info("Building LangChain Documents …")
        self._documents: list[Document] = [
            Document(
                page_content=c.get("text", ""),
                metadata={
                    "idx": i,
                    "chunk_id":       c.get("chunk_id", ""),
                    "subject":        c.get("subject", ""),
                    "chapter_number": str(c.get("chapter_number", "")),
                    "chapter_title":  c.get("chapter_title", ""),
                    "section_title":  c.get("section_title", ""),
                    "chunk_type":     c.get("chunk_type", ""),
                    "activity_number":str(c.get("activity_number", "")),
                },
            )
            for i, c in enumerate(raw)
        ]

        # ── FAISS VectorStore ─────────────────────────────────────────────────
        log.info("Building FAISS index (this runs once) …")
        self._vectorstore = FAISS.from_documents(
            self._documents,
            self._embeddings,
        )
        log.info("FAISS index ready — %d vectors", len(self._documents))

        # ── Cross-encoder (LangChain wrapper) ────────────────────────────────
        log.info("Loading cross-encoder '%s' …", CROSS_ENCODER_NAME)
        self._cross_encoder_model = HuggingFaceCrossEncoder(
            model_name=CROSS_ENCODER_NAME
        )

        # ── Pre-build metadata lookup indexes for O(1) access ─────────────────
        self._by_chapter: dict[tuple[str, str], list[int]] = {}
        self._by_subject: dict[str, list[int]] = {}

        for idx, c in enumerate(raw):
            subj = c.get("subject", "").strip().lower()
            chap = str(c.get("chapter_number", "")).strip()
            self._by_subject.setdefault(subj, []).append(idx)
            self._by_chapter.setdefault((subj, chap), []).append(idx)

    def subjects(self) -> list[str]:
        return list(self._by_subject.keys())

    def chapters_for_subject(self, subject: str) -> list[str]:
        subj = subject.strip().lower()
        return [k[1] for k in self._by_chapter if k[0] == subj]

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a query string and return a numpy array (for cosine scoring)."""
        vec = self._embeddings.embed_query(text)
        return np.array(vec, dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# MetadataFilter  (deterministic — unchanged logic from original)
# ══════════════════════════════════════════════════════════════════════════════

class MetadataFilter:
    def __init__(self, store: ChunkStore):
        self.store = store

    def filter(
        self,
        pq: ParsedQuery,
        query_text: str,
    ) -> tuple[list[int], str]:
        steps: list[str] = []
        candidates: list[int] = list(range(len(self.store.chunks)))

        is_reference_query = (
            pq.activity_number is not None or pq.exercise_number is not None
        )

        # ── Step 1: Subject ───────────────────────────────────────────────────
        if pq.subject:
            subj_key = pq.subject.strip().lower()
            if subj_key not in self.store._by_subject:
                raise AmbiguityError(
                    f"Subject '{pq.subject}' not found. "
                    f"Available: {self.store.subjects()}"
                )
            candidates = self.store._by_subject[subj_key]
            steps.append(f"subject={pq.subject}")
        else:
            steps.append("subject=ALL (not specified)")

        # ── Step 2: Chapter ───────────────────────────────────────────────────
        if pq.chapter_number is not None:
            chap_str = str(pq.chapter_number)
            if pq.subject:
                subj_key = pq.subject.strip().lower()
                key = (subj_key, chap_str)
                if key not in self.store._by_chapter:
                    raise AmbiguityError(
                        f"Chapter {pq.chapter_number} not found in subject '{pq.subject}'. "
                        f"Available chapters: {self.store.chapters_for_subject(pq.subject)}"
                    )
                chapter_idxs = set(self.store._by_chapter[key])
            else:
                chapter_idxs = set()
                for (_, chap), idxs in self.store._by_chapter.items():
                    if chap == chap_str:
                        chapter_idxs.update(idxs)

            candidates = [i for i in candidates if i in chapter_idxs]
            steps.append(f"chapter={pq.chapter_number}")

            if not candidates:
                raise AmbiguityError(
                    f"No chunks found for chapter={pq.chapter_number}. "
                    "Check chapter number or subject."
                )

        # ── USE CASE 1: reference query ───────────────────────────────────────
        if is_reference_query:
            ct = _normalise_chunk_type(pq.chunk_type)
            if ct not in ("unknown", ""):
                candidates = [
                    i for i in candidates
                    if _normalise_chunk_type(
                        self.store.chunks[i].get("chunk_type", "")
                    ) == ct
                ]
                steps.append(f"chunk_type={ct}")

            if pq.activity_number is not None:
                act_str = _normalise_activity_str(str(pq.activity_number))
                exact = [
                    i for i in candidates
                    if _normalise_activity_str(
                        str(self.store.chunks[i].get("activity_number", ""))
                    ) == act_str
                ]
                if exact:
                    candidates = exact
                    steps.append(f"activity_number={pq.activity_number} [EXACT]")
                else:
                    raise AmbiguityError(
                        f"Activity {pq.activity_number} not found "
                        f"in chapter={pq.chapter_number}, subject={pq.subject}. "
                        "Verify the activity number or chapter."
                    )

            if pq.exercise_number is not None:
                ex_str = _normalise_activity_str(str(pq.exercise_number))
                exact = [
                    i for i in candidates
                    if _normalise_activity_str(
                        str(self.store.chunks[i].get("activity_number", ""))
                    ) == ex_str
                ]
                if exact:
                    candidates = exact
                    steps.append(f"exercise_number={pq.exercise_number} [EXACT]")
                else:
                    raise AmbiguityError(
                        f"Exercise {pq.exercise_number} not found "
                        f"in chapter={pq.chapter_number}, subject={pq.subject}. "
                        "Verify the exercise number or chapter."
                    )

        # ── USE CASE 2: topic/concept query ───────────────────────────────────
        else:
            ALLOWED_TYPES = {"theory", "activity"}
            candidates = [
                i for i in candidates
                if _normalise_chunk_type(
                    self.store.chunks[i].get("chunk_type", "")
                ) in ALLOWED_TYPES
            ]
            steps.append("chunk_type=theory|activity [SEMANTIC SEARCH]")

            if not candidates:
                raise AmbiguityError(
                    "No theory or activity chunks found after applying subject/chapter filters. "
                    "Check subject or chapter constraints."
                )

        filter_path = " → ".join(steps) if steps else "no-filter (open query)"
        return candidates, filter_path


# ══════════════════════════════════════════════════════════════════════════════
# SemanticRanker  — cosine similarity within filtered pool (FAISS embeddings)
# ══════════════════════════════════════════════════════════════════════════════

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


class SemanticRanker:
    def __init__(self, store: ChunkStore):
        self.store = store
        # Extract FAISS index vectors as numpy for direct cosine scoring
        # (same approach as original, just sourced from LangChain FAISS store)
        self._embeddings_matrix: Optional[np.ndarray] = None

    def _get_embeddings_matrix(self) -> np.ndarray:
        """Lazily extract embedding matrix from FAISS store."""
        if self._embeddings_matrix is None:
            index = self.store._vectorstore.index
            n = index.ntotal
            d = index.d
            matrix = np.zeros((n, d), dtype=np.float32)
            index.reconstruct_n(0, n, matrix)
            self._embeddings_matrix = matrix
            log.info("Extracted FAISS embedding matrix — shape %s", matrix.shape)
        return self._embeddings_matrix

    def rank(
        self,
        query_vec: np.ndarray,
        candidate_indices: list[int],
        top_k: int = TOP_K,
    ) -> list[tuple[int, float]]:
        if not candidate_indices:
            return []
        matrix = self._get_embeddings_matrix()
        scores = [
            (idx, _cosine(query_vec, matrix[idx]))
            for idx in candidate_indices
        ]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# ══════════════════════════════════════════════════════════════════════════════
# CrossEncoderReranker  — LangChain HuggingFaceCrossEncoder, Use Case 2 only
# ══════════════════════════════════════════════════════════════════════════════

class CrossEncoderReranker:
    """
    Second-pass reranker using LangChain's HuggingFaceCrossEncoder.
    ONLY used for Use Case 2 (topic/concept queries).
    """

    def __init__(self, store: ChunkStore):
        self.store = store
        self._model = store._cross_encoder_model

    def rerank(
        self,
        query_text: str,
        bi_encoder_results: list[tuple[int, float]],
        top_k: int = TOP_K,
    ) -> list[tuple[int, float]]:
        if not bi_encoder_results:
            return []

        # Build (query, chunk_text) pairs
        pairs = [
            (query_text, self.store.chunks[idx].get("text", ""))
            for idx, _ in bi_encoder_results
        ]

        # LangChain HuggingFaceCrossEncoder.score() — returns list[float]
        ce_scores: list[float] = self._model.score(pairs)

        reranked = [
            (bi_encoder_results[i][0], round(ce_scores[i], 4))
            for i in range(len(bi_encoder_results))
        ]
        reranked.sort(key=lambda x: x[1], reverse=True)

        log.info(
            "Cross-encoder reranked %d candidates → top %d",
            len(reranked), top_k,
        )
        return reranked[:top_k]


# ══════════════════════════════════════════════════════════════════════════════
# Public API: Retriever
# ══════════════════════════════════════════════════════════════════════════════

class Retriever:
    """
    Single entry point for the RAG retrieval pipeline.
    Drop-in replacement for the original Retriever — same public API.
    """

    def __init__(
        self,
        chunks_path: Path = CHUNKS_PATH,
        embed_model: str  = EMBED_MODEL_NAME,
        top_k: int        = TOP_K,
    ):
        self.top_k    = top_k
        self.store    = ChunkStore(chunks_path, embed_model)
        self.filter   = MetadataFilter(self.store)
        self.ranker   = SemanticRanker(self.store)
        self.reranker = CrossEncoderReranker(self.store)

    def retrieve(
        self,
        parsed_query: dict | ParsedQuery,
        raw_query_text: str,
    ) -> list[RetrievedChunk]:
        if isinstance(parsed_query, dict):
            pq = ParsedQuery.from_dict(parsed_query)
        else:
            pq = parsed_query

        log.info("Parsed query: %s", pq)

        candidates, filter_path = self.filter.filter(pq, raw_query_text)

        log.info("Filter path: [%s] → %d candidates", filter_path, len(candidates))

        if not candidates:
            raise AmbiguityError(
                "Metadata filters produced 0 candidates. "
                "Relax constraints or check the query."
            )

        is_reference_query = (
            pq.activity_number is not None or pq.exercise_number is not None
        )

        # ── USE CASE 1: exact reference lookup (bi-encoder only) ──────────────
        if is_reference_query:
            query_vec = self.store.embed_query(raw_query_text)
            ranked    = self.ranker.rank(query_vec, candidates, top_k=self.top_k)
            log.info("Use Case 1 — exact reference: %d results", len(ranked))

        # ── USE CASE 2: topic/concept (bi-encoder → cross-encoder) ────────────
        else:
            query_vec  = self.store.embed_query(raw_query_text)
            recall_k   = max(self.top_k, min(BI_ENCODER_RECALL_K, len(candidates)))
            bi_shortlist = self.ranker.rank(query_vec, candidates, top_k=recall_k)
            log.info(
                "Use Case 2 — bi-encoder shortlist=%d, reranking to top %d …",
                len(bi_shortlist), self.top_k,
            )
            ranked = self.reranker.rerank(raw_query_text, bi_shortlist, top_k=self.top_k)

        # ── Build output ───────────────────────────────────────────────────────
        results: list[RetrievedChunk] = []
        for idx, score in ranked:
            if is_reference_query and score < SCORE_THRESHOLD:
                log.warning(
                    "Low similarity score %.3f for chunk '%s'.",
                    score, self.store.chunks[idx]["chunk_id"],
                )
            c = self.store.chunks[idx]
            results.append(
                RetrievedChunk(
                    chunk_id        = c.get("chunk_id", ""),
                    subject         = c.get("subject", ""),
                    chapter_number  = str(c.get("chapter_number", "")),
                    chapter_title   = c.get("chapter_title", ""),
                    section_title   = c.get("section_title", ""),
                    chunk_type      = c.get("chunk_type", ""),
                    activity_number = str(c.get("activity_number", "")),
                    text            = c.get("text", ""),
                    score           = round(score, 4),
                    filter_path     = filter_path,
                )
            )
        return results

    def retrieve_safe(
        self,
        parsed_query: dict | ParsedQuery,
        raw_query_text: str,
    ) -> tuple[list[RetrievedChunk], Optional[str]]:
        try:
            return self.retrieve(parsed_query, raw_query_text), None
        except AmbiguityError as e:
            return [], str(e)


# ── CLI smoke-test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import json as _json
    from query_parser_v2 import parse_query_with_slm

    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "explain activity 2 from chapter 1"
    print(f"\nQuery: {query!r}\n")

    raw_parse = parse_query_with_slm(query)
    parsed    = _json.loads(raw_parse)
    print("Parsed metadata:", _json.dumps(parsed, indent=2))

    retriever = Retriever()
    results, err = retriever.retrieve_safe(parsed, query)

    if err:
        print(f"\n⚠  Ambiguity / Error: {err}")
        sys.exit(1)

    print(f"\nTop {len(results)} chunks:\n")
    for i, r in enumerate(results, 1):
        print(f"{'─'*60}")
        print(f"[{i}] {r.chunk_id}  score={r.score}")
        print(f"    Subject  : {r.subject}")
        print(f"    Chapter  : {r.chapter_number} — {r.chapter_title}")
        print(f"    Section  : {r.section_title}")
        print(f"    Type     : {r.chunk_type}  activity={r.activity_number}")
        print(f"    Filter   : {r.filter_path}")
        print(f"    Text     : {r.text[:200].strip()} …")
    print(f"{'─'*60}")
