"""
memory_graph.py — LangGraph Memory Layer  (Option B: Semantic Follow-up Detection)
=====================================================================================
Wraps the full RAG pipeline inside a LangGraph StateGraph so every turn is
automatically checkpointed to a local SQLite database (memory.db).

Follow-up detection strategy — Option B (Semantic Similarity):
  Instead of brittle keyword/pronoun matching, we embed the current query and
  compare it to the last spoken answer using cosine similarity.

  Decision logic (in order of priority):
    1. HARD ANCHOR — query explicitly says "chapter N" / "activity N" / "exercise N"
       -> always use that chapter, skip similarity entirely  (fast path)
    2. NO PREVIOUS CONTEXT — first turn, or state has no last_spoken_answer
       -> fresh search, no chapter injection
    3. SEMANTIC SIMILARITY — embed(query) vs embed(last_spoken_answer) >= THRESHOLD
       -> inject last_chapter_number so retrieval stays anchored to the same chapter
    4. LOW SIMILARITY — genuinely different topic
       -> fresh search across all chapters

  This catches every follow-up phrasing without keyword rules:
    "what is it"               -> high similarity to last answer -> inject chapter
    "explain in detail"        -> high similarity                -> inject chapter
    "what are the materials"   -> semantically related           -> inject chapter
    "what is the conclusion"   -> semantically related           -> inject chapter
    "explain Newton's laws"    -> low similarity to litmus answer-> fresh search
    "activity 3 chapter 2"     -> hard anchor                   -> use chapter 2

State stored per turn (all checkpointed to SQLite):
  student_id, subject, messages,
  last_query, resolved_query, last_parsed,
  last_chapter_number, last_chapter_title,   <- your request
  last_chunk_ids, last_chunk_scores,          <- your request
  last_filter_path, last_answer_type,
  last_spoken_answer, last_display_md,
  last_confidence, last_warning,
  last_similarity, memory_used,               <- diagnostic fields
  turn_count, error, retrieved_chunks

Install:
  pip install langgraph langgraph-checkpoint-sqlite
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from pathlib import Path
from typing import Optional

import numpy as np

# LangGraph
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from typing_extensions import TypedDict

# Pipeline components
from query_parser_v2 import parse_query_with_slm
from parse_sanitizer import sanitize
from retriever import Retriever, RetrievedChunk
from generator import Generator

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# Constants
MEMORY_DB_PATH    = Path("memory.db")
MAX_HISTORY_TURNS = 10     # conversation pairs kept in messages list

# Per-anchor similarity thresholds — each anchor has a different precision level
# so each gets its own threshold instead of one global value.
#
#  TITLE_THRESHOLD  = 0.25  — chapter title is short and precise.
#                             A score of 0.25+ against a clean title is a
#                             strong signal the query is about the same topic.
#
#  QUERY_THRESHOLD  = 0.45  — previous query anchor is noisy because queries
#                             share intent words ("explain", "what", "how")
#                             regardless of topic. Needs a higher bar to avoid
#                             false positives like "explain Newton" matching
#                             "explain acid/base" just due to shared "explain".
#
#  ANSWER_THRESHOLD = 0.30  — spoken answer is medium precision after LaTeX
#                             stripping. 0.30 is a reasonable middle ground.
#
# To make memory MORE sticky (follows context longer) → lower all thresholds
# To make memory LESS sticky (switches topic sooner) → raise all thresholds
TITLE_THRESHOLD  = 0.25
QUERY_THRESHOLD  = 0.45
ANSWER_THRESHOLD = 0.30

# Phase 4 — Chapter lock timeout
# If the student asks N consecutive fresh-search turns (no anchor hit),
# we assume they have genuinely moved on and reset the chapter lock.
# This prevents memory drift in long sessions.
# Set to 2 as requested — raise to 3 if you want more tolerance.
DRIFT_RESET_AFTER = 2

# Intent-only queries — pure continuation signals with zero semantic content.
# Similarity scoring is useless for these because they have no domain words.
# If there is previous context, these are ALWAYS treated as follow-ups.
# Examples: "why?", "how?", "example?", "draw diagram", "can you simplify?"
_INTENT_ONLY = re.compile(
    r"^\s*("
    r"why\??"
    r"|how\??"
    r"|what\??"
    r"|example\??"
    r"|examples\??"
    r"|diagram\??"
    r"|draw\s+diagram"
    r"|simplif(y|ied|ication)\??"
    r"|can\s+you\s+simplif(y|ied)(\s+this|\s+that|\s+it)?\??"
    r"|explain\s+(more|again|further|that|it|this)\??"
    r"|more\s+(detail|details|info|information)?\??"
    r"|tell\s+me\s+more\??"
    r"|go\s+on"
    r"|continue"
    r"|and\s+then\s*\??"
    r"|so\s+what\s+(happens|next)\??"
    r"|again\??"
    r"|repeat\??"
    r")\s*$",
    re.IGNORECASE,
)


# ══════════════════════════════════════════════════════════════════════════════
# Graph State Schema
# ══════════════════════════════════════════════════════════════════════════════

class RAGState(TypedDict):
    """Complete state of one student session — every field checkpointed to SQLite."""

    # Identity
    student_id:           str
    subject:              str

    # Conversation history
    messages:             list[BaseMessage]

    # Current turn - query
    last_query:           str
    resolved_query:       str            # logs what chapter injection happened
    last_parsed:          dict

    # Current turn - retrieval
    last_chapter_number:  Optional[str]
    last_chapter_title:   Optional[str]
    last_chunk_ids:       list[str]
    last_chunk_scores:    list[float]
    last_filter_path:     Optional[str]

    # Current turn - generation
    last_answer_type:     Optional[str]
    last_spoken_answer:   Optional[str]  # also used as similarity reference next turn
    last_display_md:      Optional[str]
    last_confidence:      Optional[float]
    last_warning:         Optional[str]

    # Memory diagnostics
    last_similarity:      Optional[float]  # cosine similarity score this turn
    memory_used:          Optional[str]    # "hard_anchor"|"intent_only"|"semantic"|"fresh"|"no_context"

    # Session bookkeeping
    turn_count:              int
    consecutive_fresh_turns: int           # Phase 4: counts back-to-back fresh turns
                                           # when this hits DRIFT_RESET_AFTER → chapter lock released
    error:                   Optional[str]
    retrieved_chunks:        list[dict]    # transient — cleared each turn


# ══════════════════════════════════════════════════════════════════════════════
# Numpy type safety
# LangGraph msgpack serialiser cannot handle numpy.float32/int32
# ══════════════════════════════════════════════════════════════════════════════

def _to_python(val):
    """Recursively convert numpy scalars to native Python types."""
    if isinstance(val, np.floating):  return float(val)
    if isinstance(val, np.integer):   return int(val)
    if isinstance(val, np.ndarray):   return val.tolist()
    if isinstance(val, list):         return [_to_python(v) for v in val]
    if isinstance(val, dict):         return {k: _to_python(v) for k, v in val.items()}
    return val


# ══════════════════════════════════════════════════════════════════════════════
# Semantic Follow-up Detector  (Option B — Three-Anchor Scoring)
#
# Problem with single-anchor (last_spoken_answer only):
#   - Chunks contain LaTeX: "$\mathrm{H}^{+}$ ions..." — embedding is noisy
#   - Vague queries like "what is it explain in detail" have no domain words
#     so similarity against ANY domain text is always low
#
# Solution — three anchors, take the MAX:
#   Anchor 1: last_chapter_title   (clean, short, always high signal)
#   Anchor 2: last_query           (previous question, clean natural language)
#   Anchor 3: clean(last_spoken_answer[:200])  (LaTeX stripped, capped short)
#
#   "what is it explain in detail" vs "explain about acid and base difference?"
#   -> sim = 0.71 (anchor 2 wins) -> correctly identified as follow-up
#
#   "explain Newton laws" vs "explain about acid and base difference?"
#   -> sim = 0.31 across all anchors -> correctly identified as fresh
#
# LaTeX is only stripped for the similarity comparison text.
# Chunks, FAISS index, and retrieval are completely untouched.
# ══════════════════════════════════════════════════════════════════════════════

_HARD_ANCHOR = re.compile(
    r"\b(chapter|activity|exercise)\s*\d+",
    re.IGNORECASE,
)

# LaTeX patterns to strip before embedding anchor text
_LATEX_INLINE   = re.compile(r"\$.*?\$",           re.DOTALL)   # $...$
_LATEX_DISPLAY  = re.compile(r"\$\$.*?\$\$",     re.DOTALL)   # $$...$$
_LATEX_CMD      = re.compile(r"\\[a-zA-Z]+\{.*?\}", re.DOTALL) # \cmd{...}
_LATEX_CMD_BARE = re.compile(r"\\[a-zA-Z]+")                     # \cmd
_LATEX_BRACES   = re.compile(r"[\{\}\^_]")                      # stray { } ^ _


def _strip_latex(text: str) -> str:
    """
    Strip LaTeX math notation from text before using it as a similarity anchor.

    IMPORTANT: This function is ONLY called on the anchor strings used for
    the similarity comparison (chapter title, previous query, spoken answer).
    It is NEVER called on chunk text — chunks are retrieved and displayed
    with full LaTeX intact.

    Examples:
      "$\\mathrm{H}^{+}$"  ->  ""
      "Chapter 2 - Acids, Bases and Salts"  ->  unchanged (no LaTeX)
      "The pH scale from 0 to 14."  ->  unchanged
    """
    text = _LATEX_DISPLAY.sub(" ", text)    # $$...$$ first (before inline)
    text = _LATEX_INLINE.sub(" ", text)     # $...$
    text = _LATEX_CMD.sub(" ", text)        # \cmd{...}
    text = _LATEX_CMD_BARE.sub(" ", text)   # bare \cmd
    text = _LATEX_BRACES.sub(" ", text)     # stray { } ^ _
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def _check_followup(
    query:    str,
    state:    RAGState,
    embed_fn,              # retriever.store.embed_query — already loaded, ~5ms per call
) -> tuple[bool, float, str, dict]:
    """
    Decide whether this query is a follow-up using three-anchor per-threshold scoring.

    Returns (is_followup, best_similarity, reason, anchor_scores)
      reason       : "hard_anchor" | "no_context" | "semantic" | "fresh"
      anchor_scores: {"title": float, "query": float, "answer": float,
                      "title_hit": bool, "query_hit": bool, "answer_hit": bool}

    Decision order:
      1. Hard anchor  -> not a follow-up (explicit chapter/activity/exercise number)
      2. No context   -> not a follow-up (first turn, no previous chapter)
      3. Per-anchor check — ANY anchor hitting its threshold → follow-up:
           sim_title  >= TITLE_THRESHOLD  (0.25)  title match is precise
           sim_query  >= QUERY_THRESHOLD  (0.45)  query needs higher bar (intent noise)
           sim_answer >= ANSWER_THRESHOLD (0.30)  answer, LaTeX stripped
      4. No anchor hit threshold → fresh search

    Per-anchor thresholds prevent false positives from shared intent words.
    Example: "explain Newton's laws" vs "explain about acid/base difference"
      sim_query = 0.31  <  QUERY_THRESHOLD 0.45  → no hit → fresh search  ✓
    Example: "what is it can you explain in detail" vs same context
      sim_query = 0.52  >= QUERY_THRESHOLD 0.45  → hit → inject chapter   ✓
    """
    empty_scores = {
        "title": 0.0, "query": 0.0, "answer": 0.0,
        "title_hit": False, "query_hit": False, "answer_hit": False,
    }

    # Priority 1: hard anchor — explicit chapter/activity/exercise number
    if _HARD_ANCHOR.search(query):
        log.info("Memory | hard_anchor in %r", query)
        return False, 0.0, "hard_anchor", empty_scores

    # Priority 2: intent-only queries — pure continuation signals
    # These have no domain words so similarity scoring is useless.
    # "why?", "how?", "example?", "draw diagram", "can you simplify?" etc.
    # If there is previous context → always a follow-up.
    # If no previous context → no_context (nothing to continue from).
    if _INTENT_ONLY.match(query):
        has_ctx = bool(state.get("last_chapter_number"))
        if has_ctx:
            log.info("Memory | intent_only query %r — forcing follow-up", query)
            return True, 1.0, "intent_only", {
                "title": 1.0, "query": 1.0, "answer": 1.0,
                "title_hit": True, "query_hit": True, "answer_hit": True,
            }
        else:
            log.info("Memory | intent_only query %r — no context yet", query)
            return False, 0.0, "no_context", empty_scores

    # Priority 3: no previous context to compare against
    last_answer = state.get("last_spoken_answer") or ""
    last_title  = state.get("last_chapter_title")  or ""
    last_q      = state.get("last_query")           or ""
    if not state.get("last_chapter_number") or (not last_answer and not last_title):
        log.info("Memory | no_context — first turn or missing previous chapter")
        return False, 0.0, "no_context", empty_scores

    # Priority 3: per-anchor similarity check
    try:
        q_vec = embed_fn(query)

        # ── Anchor 1: chapter title ───────────────────────────────────────────
        # Short, clean, precise. Low threshold (0.25) is safe here.
        # Example: "what are the materials" vs "Acids, Bases and Salts" → ~0.32
        sim_title = 0.0
        if last_title:
            sim_title = _cosine(q_vec, embed_fn(_strip_latex(last_title)))

        # ── Anchor 2: previous query ──────────────────────────────────────────
        # Best for vague follow-ups ("what is it", "explain that").
        # Both query and previous query share intent structure.
        # Needs HIGH threshold (0.45) because queries starting with "explain"
        # share intent words regardless of topic — without this, "explain Newton"
        # would falsely match "explain acid/base" (~0.31 similarity).
        sim_query = 0.0
        if last_q and last_q.strip().lower() != query.strip().lower():
            sim_query = _cosine(q_vec, embed_fn(_strip_latex(last_q)))

        # ── Anchor 3: spoken answer (LaTeX stripped, capped at 300 chars) ─────
        # Medium precision. LaTeX stripped so math symbols don't pollute embedding.
        # Capped at 300 chars — longer text adds noise, not signal.
        sim_answer = 0.0
        if last_answer:
            sim_answer = _cosine(q_vec, embed_fn(_strip_latex(last_answer)[:300]))

        # ── Per-anchor threshold check ────────────────────────────────────────
        title_hit  = sim_title  >= TITLE_THRESHOLD
        query_hit  = sim_query  >= QUERY_THRESHOLD
        answer_hit = sim_answer >= ANSWER_THRESHOLD
        is_followup = title_hit or query_hit or answer_hit

        # Best raw similarity for logging/display
        best_sim = max(sim_title, sim_query, sim_answer)

        anchor_scores = {
            "title":      round(sim_title,  3),
            "query":      round(sim_query,  3),
            "answer":     round(sim_answer, 3),
            "title_hit":  title_hit,
            "query_hit":  query_hit,
            "answer_hit": answer_hit,
        }

        # Which anchor(s) triggered the follow-up decision
        hits = [
            k for k, v in [
                ("title",  title_hit),
                ("query",  query_hit),
                ("answer", answer_hit),
            ] if v
        ]
        hit_str = "+".join(hits) if hits else "none"

        log.info(
            "Memory | followup=%s  hits=%s  "
            "title=%.3f(th=%.2f)  query=%.3f(th=%.2f)  answer=%.3f(th=%.2f)  "
            "query=%r",
            is_followup, hit_str,
            sim_title,  TITLE_THRESHOLD,
            sim_query,  QUERY_THRESHOLD,
            sim_answer, ANSWER_THRESHOLD,
            query,
        )

        reason = "semantic" if is_followup else "fresh"
        return is_followup, best_sim, reason, anchor_scores

    except Exception as e:
        log.warning("Memory | embedding error (%s) — treating as fresh", e)
        return False, 0.0, "fresh", empty_scores


# ══════════════════════════════════════════════════════════════════════════════
# Graph Nodes
# ══════════════════════════════════════════════════════════════════════════════

def parse_node(state: RAGState, retriever: Retriever) -> dict:
    """
    Node 1: Semantic follow-up check + parse + chapter memory injection.

    This is the heart of Option B.  In a single node:
      1. Embed the query, compare to last spoken answer
      2. Parse the query with the SLM
      3. If semantic follow-up AND SLM found no chapter -> inject last chapter
         This keeps retrieval anchored to where the student was studying.
    """
    query    = state["last_query"]
    embed_fn = retriever.store.embed_query   # reuse already-loaded model

    # Step 1: three-anchor semantic follow-up check
    is_followup, similarity, reason, anchor_scores = _check_followup(
        query, state, embed_fn
    )

    # Step 2: parse with SLM
    try:
        raw    = parse_query_with_slm(query)
        parsed = json.loads(raw)
        parsed = sanitize(parsed, query)
        parsed["subject"] = state["subject"]   # hard override

        # ── Step 3a: Phase 4 — Chapter lock timeout (drift prevention) ─────────
        # Track how many consecutive turns scored below all thresholds.
        # Once DRIFT_RESET_AFTER fresh turns accumulate, the chapter lock is
        # released — memory stops injecting the stale chapter.
        #
        # Reset counter to 0 on any follow-up or hard_anchor turn.
        # Increment counter on fresh/no_context turns.
        # When counter reaches DRIFT_RESET_AFTER: wipe last_chapter_number.
        #
        # Example (DRIFT_RESET_AFTER=2):
        #   Turn 5: explain acid/base         → semantic  → counter=0
        #   Turn 6: what is it                → semantic  → counter=0
        #   Turn 7: explain Newton (fresh)    → fresh     → counter=1
        #   Turn 8: what is force (fresh)     → fresh     → counter=2 → RESET
        #   Turn 9: why                       → no_context now → fresh search ✓

        prev_fresh_count = state.get("consecutive_fresh_turns", 0)

        if reason in ("semantic", "hard_anchor"):
            # Follow-up or explicit anchor → reset the drift counter
            new_fresh_count = 0
        else:
            # fresh or no_context → increment
            new_fresh_count = prev_fresh_count + 1

        # Apply drift reset — wipe chapter lock if threshold crossed
        chapter_locked = state.get("last_chapter_number")
        if new_fresh_count >= DRIFT_RESET_AFTER and chapter_locked:
            log.info(
                "Memory | DRIFT RESET after %d consecutive fresh turns — "
                "releasing chapter lock (was chapter=%s)",
                new_fresh_count, chapter_locked,
            )
            # Wipe from parsed too so this turn retrieves fresh
            parsed["chapter_number"] = None

        # ── Step 3b: Chapter + Topic injection (intent continuation) ──────────
        # Inject BOTH chapter_number AND topic from last turn so that:
        #   (a) retrieval stays in the right chapter
        #   (b) semantic search within the chapter targets the right section
        #
        # Without topic injection, "why?" in chapter 2 might retrieve any
        # section of chapter 2 — with it, retrieval is anchored to the exact
        # topic the student was studying (e.g. "pH scale and indicators").
        #
        # Topic is only injected when:
        #   - follow-up is detected (is_followup=True)
        #   - drift reset did NOT just fire (chapter_number still valid)
        #   - SLM produced a vague/empty topic for the follow-up query
        injected_chapter = None
        injected_topic   = None

        if is_followup and parsed.get("chapter_number") is None:
            # Only inject chapter if drift reset didn't just clear it
            if new_fresh_count < DRIFT_RESET_AFTER:
                prev_chapter = state.get("last_chapter_number")
                if prev_chapter:
                    try:
                        parsed["chapter_number"] = int(prev_chapter)
                        injected_chapter = parsed["chapter_number"]
                    except (ValueError, TypeError):
                        pass

        if is_followup and injected_chapter is not None:
            # Topic injection — anchor retrieval to the specific section
            # Use last_parsed topic only if current query topic is vague
            # (short queries like "why?", "how?", "example?" have generic topics)
            current_topic = parsed.get("topic", "").strip()
            prev_topic    = (state.get("last_parsed") or {}).get("topic", "").strip()

            topic_is_vague = (
                len(current_topic.split()) <= 4        # very short topic
                or current_topic.lower() in {          # SLM returned placeholder
                    "unknown", "none", "", "topic",
                    "what is it", "explain", "why", "how",
                    "example", "simplify", "diagram",
                }
            )

            if topic_is_vague and prev_topic:
                parsed["topic"] = prev_topic
                injected_topic  = prev_topic
                log.info("Memory | injected topic=%r (current was vague: %r)",
                         injected_topic, current_topic)
            elif prev_topic and not current_topic:
                parsed["topic"] = prev_topic
                injected_topic  = prev_topic
                log.info("Memory | injected topic=%r (current was empty)",
                         injected_topic)

        # Log final injection summary
        hits = "+".join(
            k for k, v in [
                ("title",  anchor_scores.get("title_hit",  False)),
                ("query",  anchor_scores.get("query_hit",  False)),
                ("answer", anchor_scores.get("answer_hit", False)),
            ] if v
        ) or "none"

        if injected_chapter:
            log.info(
                "Memory | injected chapter=%s topic=%r  "
                "hits=%s  title=%.3f  query=%.3f  answer=%.3f  fresh_count=%d",
                injected_chapter,
                injected_topic or "(kept current)",
                hits,
                anchor_scores.get("title",  0.0),
                anchor_scores.get("query",  0.0),
                anchor_scores.get("answer", 0.0),
                new_fresh_count,
            )

        # Build resolved_query string for display / logging
        if injected_chapter:
            resolved = (
                f"{query}  "
                f"[memory: chapter={injected_chapter}"
                + (f", topic={injected_topic!r}" if injected_topic else "")
                + f", anchor={hits}"
                + f", title={anchor_scores.get('title', 0.0):.2f}"
                  f"/query={anchor_scores.get('query', 0.0):.2f}"
                  f"/answer={anchor_scores.get('answer', 0.0):.2f}]"
            )
        else:
            resolved = query

        log.info("Parsed: %s", json.dumps(parsed))

        return {
            "last_parsed":              parsed,
            "resolved_query":           resolved,
            "last_similarity":          float(similarity),
            "memory_used":              reason,
            "consecutive_fresh_turns":  new_fresh_count,
            "error":                    None,
        }

    except Exception as e:
        log.error("Parse node error: %s", e)
        return {
            "last_parsed": {
                "intent": "unknown", "chunk_type": "unknown",
                "chapter_number": None, "chapter_name": None,
                "activity_number": None, "exercise_number": None,
                "topic": query, "subject": state["subject"],
            },
            "resolved_query":             query,
            "last_similarity":            float(similarity),
            "memory_used":                reason,
            "consecutive_fresh_turns":    state.get("consecutive_fresh_turns", 0) + 1,
            "error":                      f"Parse error: {e}",
        }


def retrieve_node(state: RAGState, retriever: Retriever) -> dict:
    """
    Node 2: Retrieve top-K chunks using parsed metadata.
    chapter_number in last_parsed is already injected if this is a follow-up.
    """
    parsed = state.get("last_parsed", {})
    query  = state["last_query"]   # use raw query for embedding search

    chunks, err = retriever.retrieve_safe(parsed, query)

    if err:
        log.error("Retrieve node error: %s", err)
        return {
            "last_chunk_ids":      [],
            "last_chunk_scores":   [],
            "last_chapter_number": None,
            "last_chapter_title":  None,
            "last_filter_path":    None,
            "retrieved_chunks":    [],
            "error":               err,
        }

    chapter_number = chunks[0].chapter_number if chunks else None
    chapter_title  = chunks[0].chapter_title  if chunks else None
    filter_path    = chunks[0].filter_path    if chunks else None
    chunk_ids      = [c.chunk_id     for c in chunks]
    chunk_scores   = [float(c.score) for c in chunks]   # numpy.float32 -> float

    serialised = [
        {
            "chunk_id":        c.chunk_id,
            "subject":         c.subject,
            "chapter_number":  c.chapter_number,
            "chapter_title":   c.chapter_title,
            "section_title":   c.section_title,
            "chunk_type":      c.chunk_type,
            "activity_number": c.activity_number,
            "text":            c.text,
            "score":           float(c.score),   # numpy.float32 -> float
            "filter_path":     c.filter_path,
        }
        for c in chunks
    ]

    log.info(
        "Retrieved %d chunks | chapter=%s (%s) | ids=%s",
        len(chunks), chapter_number, chapter_title, chunk_ids,
    )

    return {
        "last_chunk_ids":      chunk_ids,
        "last_chunk_scores":   chunk_scores,
        "last_chapter_number": chapter_number,
        "last_chapter_title":  chapter_title,
        "last_filter_path":    filter_path,
        "retrieved_chunks":    serialised,
        "error":               None,
    }


def generate_node(state: RAGState, generator: Generator) -> dict:
    """Node 3: Generate spoken + display answer from retrieved chunks."""
    serialised = state.get("retrieved_chunks", [])

    if not serialised:
        return {
            "last_answer_type":   None,
            "last_spoken_answer": (
                "I could not find relevant content to answer your question. "
                "Please try rephrasing or check the chapter and subject."
            ),
            "last_display_md": (
                "## No Results Found\n\n"
                "No chunks were retrieved. Please check:\n"
                "- The subject and chapter are correct\n"
                "- The query is specific enough\n"
            ),
            "last_confidence": 0.0,
            "last_warning":    "No chunks retrieved.",
            "error":           state.get("error"),
        }

    chunks = [
        RetrievedChunk(
            chunk_id        = d["chunk_id"],
            subject         = d["subject"],
            chapter_number  = d["chapter_number"],
            chapter_title   = d["chapter_title"],
            section_title   = d["section_title"],
            chunk_type      = d["chunk_type"],
            activity_number = d["activity_number"],
            text            = d["text"],
            score           = d["score"],
            filter_path     = d["filter_path"],
        )
        for d in serialised
    ]

    answer, err = generator.generate_safe(chunks, state["last_query"])

    if err:
        log.error("Generate node error: %s", err)
        return {
            "last_answer_type":   None,
            "last_spoken_answer": None,
            "last_display_md":    None,
            "last_confidence":    0.0,
            "last_warning":       None,
            "error":              f"Generation error: {err}",
        }

    return {
        "last_answer_type":   answer.answer_type,
        "last_spoken_answer": answer.spoken_answer,
        "last_display_md":    answer.display_answer_markdown,
        "last_confidence":    float(answer.confidence),   # numpy -> float
        "last_warning":       answer.low_confidence_warning,
        "error":              None,
    }


def save_memory_node(state: RAGState) -> dict:
    """
    Node 4: Append this turn to conversation history, increment turn_count.
    AI message includes a hidden [MEMORY]...[/MEMORY] tag with retrieval metadata
    so the next turn's similarity check has accurate context.
    """
    messages = list(state.get("messages", []))
    messages.append(HumanMessage(content=state["last_query"]))

    ai_text    = state.get("last_spoken_answer") or "No answer generated."
    memory_tag = (
        f"\n[MEMORY]"
        f" chapter={state.get('last_chapter_number')}"
        f" title={state.get('last_chapter_title')}"
        f" chunks={state.get('last_chunk_ids', [])}"
        f" sim={state.get('last_similarity', 0.0):.3f}"
        f" reason={state.get('memory_used', 'unknown')}"
        f"[/MEMORY]"
    )
    messages.append(AIMessage(content=ai_text + memory_tag))

    # Trim to MAX_HISTORY_TURNS pairs (full history always in SQLite)
    max_msgs = MAX_HISTORY_TURNS * 2
    if len(messages) > max_msgs:
        messages = messages[-max_msgs:]

    turn_count = state.get("turn_count", 0) + 1
    fresh_count = state.get("consecutive_fresh_turns", 0)
    log.info(
        "Turn %d saved | student=%s | chapter=%s | chunks=%s | "
        "sim=%.3f | reason=%s | fresh_streak=%d",
        turn_count,
        state.get("student_id", "?"),
        state.get("last_chapter_number", "?"),
        state.get("last_chunk_ids", []),
        state.get("last_similarity", 0.0),
        state.get("memory_used", "?"),
        fresh_count,
    )

    return {"messages": messages, "turn_count": turn_count}


# ══════════════════════════════════════════════════════════════════════════════
# Conditional edge
# ══════════════════════════════════════════════════════════════════════════════

def _should_generate(state: RAGState) -> str:
    return "generate" if state.get("retrieved_chunks") else "save_memory"


# ══════════════════════════════════════════════════════════════════════════════
# Graph builder
# ══════════════════════════════════════════════════════════════════════════════

def build_rag_graph(retriever: Retriever, generator: Generator) -> tuple:
    """Build and compile the LangGraph StateGraph with SqliteSaver."""

    def _parse(state):    return parse_node(state, retriever)
    def _retrieve(state): return retrieve_node(state, retriever)
    def _generate(state): return generate_node(state, generator)

    builder = StateGraph(RAGState)
    builder.add_node("parse",       _parse)
    builder.add_node("retrieve",    _retrieve)
    builder.add_node("generate",    _generate)
    builder.add_node("save_memory", save_memory_node)

    builder.set_entry_point("parse")
    builder.add_edge("parse",       "retrieve")
    builder.add_conditional_edges(
        "retrieve", _should_generate,
        {"generate": "generate", "save_memory": "save_memory"},
    )
    builder.add_edge("generate",    "save_memory")
    builder.add_edge("save_memory", END)

    # Direct sqlite3 connection — avoids SqliteSaver.from_conn_string()
    # returning a context manager in newer LangGraph versions
    conn         = sqlite3.connect(str(MEMORY_DB_PATH), check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    graph = builder.compile(checkpointer=checkpointer)
    log.info("LangGraph compiled — checkpointing to %s", MEMORY_DB_PATH)
    return graph, checkpointer


# ══════════════════════════════════════════════════════════════════════════════
# Public API: MemoryGraph
# ══════════════════════════════════════════════════════════════════════════════

class MemoryGraph:
    """
    High-level wrapper around the compiled LangGraph RAG pipeline.

    Usage:
        mg = MemoryGraph(retriever, generator)
        result = mg.run("explain activity 2 from chapter 1",
                        student_id="alice", subject="Biology")
        result = mg.run("what are the materials needed",
                        student_id="alice", subject="Biology")
        # result["last_chapter_number"] is still "1" — semantic memory worked
        # result["memory_used"]  -> "semantic"
        # result["last_similarity"] -> 0.67 (example)
    """

    def __init__(self, retriever: Retriever, generator: Generator):
        self._retriever = retriever
        self.graph, self._checkpointer = build_rag_graph(retriever, generator)
        self._sessions: dict[str, str] = {}

    def _thread_id(self, student_id: str) -> str:
        if student_id not in self._sessions:
            self._sessions[student_id] = student_id
        return self._sessions[student_id]

    def run(self, query: str, student_id: str, subject: str) -> RAGState:
        """
        Run one full turn of the RAG pipeline with semantic memory.

        Returns the final RAGState. Key fields:
          last_spoken_answer, last_display_md,
          last_chapter_number, last_chapter_title,
          last_chunk_ids, last_chunk_scores,
          last_similarity, memory_used,
          last_confidence, turn_count, error
        """
        config = {"configurable": {"thread_id": self._thread_id(student_id)}}

        existing = self.graph.get_state(config)
        prev     = existing.values if existing and existing.values else {}

        input_state: RAGState = {
            "student_id":               student_id,
            "subject":                  subject,
            "messages":                 prev.get("messages", []),
            "last_query":               query,
            "resolved_query":           query,
            "last_parsed":              prev.get("last_parsed", {}),   # needed for topic injection
            # Previous turn retrieval context
            "last_chapter_number":      prev.get("last_chapter_number"),
            "last_chapter_title":       prev.get("last_chapter_title"),
            "last_chunk_ids":           prev.get("last_chunk_ids", []),
            "last_chunk_scores":        prev.get("last_chunk_scores", []),
            "last_filter_path":         prev.get("last_filter_path"),
            "last_answer_type":         prev.get("last_answer_type"),
            "last_spoken_answer":       prev.get("last_spoken_answer"),  # similarity anchor
            "last_display_md":          prev.get("last_display_md"),
            "last_confidence":          prev.get("last_confidence"),
            "last_warning":             prev.get("last_warning"),
            "last_similarity":          prev.get("last_similarity"),
            "memory_used":              prev.get("memory_used"),
            # Phase 4 — drift prevention counter carried forward
            "consecutive_fresh_turns":  prev.get("consecutive_fresh_turns", 0),
            "turn_count":               prev.get("turn_count", 0),
            "error":                    None,
            "retrieved_chunks":         [],
        }

        return self.graph.invoke(input_state, config=config)

    def get_history(self, student_id: str) -> list[dict]:
        """Return conversation history, [MEMORY] tags stripped."""
        config = {"configurable": {"thread_id": self._thread_id(student_id)}}
        state  = self.graph.get_state(config)
        if not state or not state.values:
            return []
        history = []
        for msg in state.values.get("messages", []):
            role    = "human" if isinstance(msg, HumanMessage) else "ai"
            content = re.sub(
                r"\[MEMORY\].*?\[/MEMORY\]", "", msg.content, flags=re.DOTALL
            ).strip()
            history.append({"role": role, "content": content})
        return history

    def get_session_summary(self, student_id: str) -> dict:
        """Compact summary of what this student has studied."""
        config = {"configurable": {"thread_id": self._thread_id(student_id)}}
        state  = self.graph.get_state(config)
        if not state or not state.values:
            return {}
        v = state.values
        return {
            "student_id":               v.get("student_id"),
            "subject":                  v.get("subject"),
            "turn_count":               v.get("turn_count", 0),
            "last_chapter_number":      v.get("last_chapter_number"),
            "last_chapter_title":       v.get("last_chapter_title"),
            "last_chunk_ids":           v.get("last_chunk_ids", []),
            "last_chunk_scores":        v.get("last_chunk_scores", []),
            "last_answer_type":         v.get("last_answer_type"),
            "last_confidence":          v.get("last_confidence"),
            "last_similarity":          v.get("last_similarity"),
            "memory_used":              v.get("memory_used"),
            "consecutive_fresh_turns":  v.get("consecutive_fresh_turns", 0),
            "last_topic":               (v.get("last_parsed") or {}).get("topic", ""),
        }


# ══════════════════════════════════════════════════════════════════════════════
# CLI smoke-test
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    student = "test_student"
    subject = "Biology"

    print("\n Loading pipeline ...")
    retriever = Retriever()
    generator = Generator()
    mg        = MemoryGraph(retriever, generator)
    print("  MemoryGraph ready.\n")

    queries = [
        "explain activity 2 from chapter 1",  # hard_anchor -> chapter 1
        "what are the materials needed",       # no pronoun  -> semantic check -> chapter 1
        "what is the conclusion",              # no pronoun  -> semantic check -> chapter 1
        "explain Newton's laws",               # low similarity -> fresh search
        "give me more detail",                 # semantic match -> Newton chapter
    ]

    for q in queries:
        print(f"\n{'='*60}")
        print(f"  Query   : {q!r}")
        result = mg.run(q, student_id=student, subject=subject)
        print(f"  Memory  : reason={result.get('memory_used')}  "
              f"sim={result.get('last_similarity', 0.0):.3f}")
        print(f"  Resolved: {result.get('resolved_query')!r}")
        print(f"  Chapter : {result.get('last_chapter_number')} "
              f"- {result.get('last_chapter_title')}")
        print(f"  Chunks  : {result.get('last_chunk_ids')}")
        if result.get("error"):
            print(f"  Error   : {result['error']}")
        else:
            s = result.get("last_spoken_answer", "")
            print(f"  Spoken  : {s[:120]}..." if len(s) > 120 else f"  Spoken  : {s}")

    print(f"\n{'='*60}")
    print("\nSession summary:")
    print(json.dumps(mg.get_session_summary(student), indent=2))