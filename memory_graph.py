"""
memory_graph.py — LangGraph Memory Layer  (Option B: Semantic Follow-up Detection)
=====================================================================================
Wraps the full RAG pipeline inside a LangGraph StateGraph so every turn is
automatically checkpointed to a local SQLite database (memory.db).

Follow-up detection strategy — Option B (Semantic Similarity):
  Instead of brittle keyword/pronoun matching, we embed the current query and
  compare it against THREE anchors from the previous turn:
    Anchor 1 — last_chapter_title   (short, clean, high signal)
    Anchor 2 — prev_query           (previous question, NOT current)
    Anchor 3 — last_spoken_answer   (LaTeX-stripped, capped 300 chars)

  Decision logic (in order of priority):
    1. HARD ANCHOR — query explicitly says "chapter N" / "activity N" / "exercise N"
       -> use that chapter directly, skip similarity              (fast path)
    2. INTENT-ONLY — query is a pure continuation signal ("why?", "example?", …)
       -> if previous context exists: always a follow-up
       -> if no previous context: fresh search
    3. NO PREVIOUS CONTEXT — first turn, or state has no last_chapter_number
       -> fresh search, no chapter injection
    4. SEMANTIC SIMILARITY — three-anchor per-threshold scoring
       -> ANY anchor hitting its threshold → inject last chapter
       -> No anchor hits → fresh search
    5. DRIFT RESET — N consecutive fresh turns → release chapter lock

BUG FIXES in this version (compared to original):
  FIX 1 — Thread ID is now (student_id + subject) so subject changes never
           share a SQLite checkpoint thread.  Memory NEVER leaks across subjects.

  FIX 2 — Subject-change detection in MemoryGraph.run():
           When subject changes, a clean input_state is built (chapter/answer
           fields reset to None) while the new subject-scoped thread starts fresh.
           Previous session remains intact in SQLite under the old thread ID.

  FIX 3 — Separate `prev_query` field for Anchor 2.
           The original code read state["last_query"] as the previous query anchor,
           but run() overwrites last_query with the CURRENT query before invoking,
           so Anchor 2 was always comparing the current query to itself (sim≈1.0
           on Turn 2, then stale on later turns).  Fixed by storing the previous
           query in a dedicated `prev_query` field that is never overwritten by
           the incoming turn.

  FIX 4 — "intent_only" reason resets consecutive_fresh_turns (same as
           "semantic" and "hard_anchor"), because it IS a follow-up.

  FIX 5 — get_history() and get_session_summary() use the subject-scoped
           thread ID so they always read the correct session.

State stored per turn (all checkpointed to SQLite):
  student_id, subject, messages,
  last_query, prev_query,           ← prev_query is new (FIX 3)
  resolved_query, last_parsed,
  last_chapter_number, last_chapter_title,
  last_chunk_ids, last_chunk_scores,
  last_filter_path, last_answer_type,
  last_spoken_answer, last_display_md,
  last_confidence, last_warning,
  last_similarity, memory_used,
  turn_count, consecutive_fresh_turns,
  error, retrieved_chunks

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

# ── Constants ──────────────────────────────────────────────────────────────────
MEMORY_DB_PATH    = Path("memory.db")
MAX_HISTORY_TURNS = 10   # conversation pairs kept in-memory messages list

# ── Per-anchor similarity thresholds ──────────────────────────────────────────
#
#  TITLE_THRESHOLD  = 0.25  — chapter title is short and precise.
#                             0.25+ against a clean title is a strong signal.
#                             Example: "what are the materials" vs
#                             "Acids, Bases and Salts" → ~0.32  ✓
#
#  QUERY_THRESHOLD  = 0.45  — previous query anchor is noisy because queries
#                             share intent words ("explain", "what", "how")
#                             regardless of topic. Needs a higher bar.
#                             Example: "explain Newton" vs "explain acid/base"
#                             → sim ≈ 0.31 < 0.45  → correctly fresh  ✓
#
#  ANSWER_THRESHOLD = 0.30  — spoken answer after LaTeX stripping.
#                             Medium precision, 0.30 is a reasonable middle ground.
#
# Tuning guide:
#   MORE sticky memory (context lasts longer) → lower all thresholds
#   LESS sticky memory (topic switches faster) → raise all thresholds
TITLE_THRESHOLD  = 0.25
QUERY_THRESHOLD  = 0.45
ANSWER_THRESHOLD = 0.30

# ── Drift-reset threshold ──────────────────────────────────────────────────────
# If the student asks N consecutive fresh-search turns (no anchor hit),
# we assume they have genuinely moved on and release the chapter lock.
# Set to 2 — raise to 3 if you want more tolerance before resetting.
DRIFT_RESET_AFTER = 2

# ── Intent-only query pattern ──────────────────────────────────────────────────
# Pure continuation signals with zero semantic domain content.
# Similarity scoring is useless for these — they have no subject-area words.
# If previous context exists → always treated as follow-ups.
#
# Design rules:
#   INCLUDE — query has no domain words; any subject could be the referent.
#             "explain it in detail", "elaborate", "what does that mean"
#   EXCLUDE — query names a specific topic or domain concept.
#             "explain Newton laws", "what is osmosis", "how does digestion work"
#
# The pattern is anchored (^ … $) so partial matches inside longer topic
# queries cannot accidentally fire.
_INTENT_ONLY = re.compile(
    r"^\s*("
    # ── Pure single-word / short continuations ─────────────────────────────
    r"why\??"
    r"|elaborate\??"
    r"|again\??"
    r"|repeat\??"
    r"|continue"
    r"|go\s+on"
    r"|example\??"
    r"|examples\??"
    r"|diagram\??"
    # draw a diagram/picture/figure
    r"|draw\s+(a\s+)?(diagram|picture|figure)"
    # ── simplify variants ──────────────────────────────────────────────────
    r"|simplif(y|ied|ication|ier)\??"
    r"|can\s+you\s+simplif(y|y\s+this|y\s+that|y\s+it)\??"
    # ── explain [it/that/this] [in (more) detail / further / more / again] ─
    # Catches: "explain more", "explain that", "explain it",
    #          "explain it in detail", "explain it in more detail",
    #          "explain that in detail", "explain in more detail"
    r"|explain\s+(more|again|further)\??"
    r"|explain\s+(it|that|this)(\s+(more|again|further|in\s+(more\s+)?detail))?\??"
    r"|explain\s+in\s+(more\s+)?detail\??"
    r"|explain\s+more\s+about\s+(it|this|that)\??"
    # ── elaborate [on it/that/this] ────────────────────────────────────────
    r"|elaborate(\s+on\s+(that|it|this))?\??"
    # ── expand [on it/that/this] ───────────────────────────────────────────
    r"|expand(\s+on\s+(that|it|this))?\??"
    # ── can you [elaborate / expand / explain more / explain in detail] ───
    r"|can\s+you\s+(elaborate|expand(\s+on\s+(that|it|this))?|explain\s+(more|that|it|this|in\s+(more\s+)?detail))\??"
    # ── more detail / more details / more info / more information ──────────
    r"|more\s+(detail|details|info|information|examples?|about\s+(it|this|that))?\??"
    # ── give me more / give me an example / give me details ───────────────
    r"|give\s+(me\s+)?(more|an?\s+example|details?|more\s+(detail|info|information|examples?))\??"
    # ── show me an example / show me more ──────────────────────────────────
    r"|show\s+(me\s+)?(an?\s+example|more)\??"
    # ── tell me more [about it/this/that] ──────────────────────────────────
    r"|tell\s+me\s+more(\s+about\s+(it|this|that))?\??"
    # ── what does that/it mean / what is it/that/this ─────────────────────
    r"|what\s+(does\s+(that|it|this)\s+mean|is\s+(it|that|this))\??"
    # ── how does it/that/this work ────────────────────────────────────────
    r"|how\s+does\s+(it|that|this)\s+work\??"
    # ── in (more) detail [please] ──────────────────────────────────────────
    r"|in\s+(more\s+)?detail(\s+please)?\??"
    # ── go deeper / go further / can you go deeper ─────────────────────────
    r"|(can\s+you\s+)?go\s+(deeper|further)\??"
    # ── and then? / so what happens next? ─────────────────────────────────
    r"|and\s+then\s*\??"
    r"|so\s+what\s+(happens|next)\??"
    r")\s*$",
    re.IGNORECASE,
)

# ── Hard-anchor pattern ────────────────────────────────────────────────────────
# Explicit chapter / activity / exercise number → always use that, skip memory.
_HARD_ANCHOR = re.compile(
    r"\b(chapter|activity|exercise)\s*\d+",
    re.IGNORECASE,
)

# ── LaTeX stripping patterns (for similarity anchors ONLY) ────────────────────
# NEVER applied to chunk text — only to anchor strings used for cosine scoring.
_LATEX_DISPLAY  = re.compile(r"\$\$.*?\$\$",        re.DOTALL)
_LATEX_INLINE   = re.compile(r"\$.*?\$",             re.DOTALL)
_LATEX_CMD      = re.compile(r"\\[a-zA-Z]+\{.*?\}", re.DOTALL)
_LATEX_CMD_BARE = re.compile(r"\\[a-zA-Z]+")
_LATEX_BRACES   = re.compile(r"[\{\}\^_]")


# ══════════════════════════════════════════════════════════════════════════════
# Graph State Schema
# ══════════════════════════════════════════════════════════════════════════════

class RAGState(TypedDict):
    """Complete state of one student+subject session — checkpointed to SQLite."""

    # Identity
    student_id:   str
    subject:      str

    # Conversation history (trimmed to MAX_HISTORY_TURNS pairs in memory)
    messages:     list[BaseMessage]

    # Current turn — query
    last_query:      str           # the query that arrived THIS turn
    prev_query:      Optional[str] # the query from the PREVIOUS turn (Anchor 2)
    resolved_query:  str           # last_query + injected context (for display)
    retrieval_query: str           # query sent to bi-encoder (enriched for follow-ups)
    last_parsed:     dict          # parsed metadata dict after SLM + sanitize

    # Current turn — retrieval
    last_chapter_number: Optional[str]
    last_chapter_title:  Optional[str]
    last_chunk_ids:      list[str]
    last_chunk_scores:   list[float]
    last_filter_path:    Optional[str]

    # Current turn — generation
    last_answer_type:   Optional[str]
    last_spoken_answer: Optional[str]   # also used as Anchor 3 next turn
    last_display_md:    Optional[str]
    last_confidence:    Optional[float]
    last_warning:       Optional[str]

    # Memory diagnostics
    last_similarity: Optional[float]  # best cosine score this turn
    memory_used:     Optional[str]    # "hard_anchor"|"intent_only"|"semantic"|"fresh"|"no_context"

    # Session bookkeeping
    turn_count:              int
    consecutive_fresh_turns: int   # drift-reset counter (FIX 4)
    error:                   Optional[str]
    retrieved_chunks:        list[dict]   # transient — cleared each turn


# ══════════════════════════════════════════════════════════════════════════════
# Numpy type-safety helper
# LangGraph msgpack serialiser cannot handle numpy.float32 / int32
# ══════════════════════════════════════════════════════════════════════════════

def _to_python(val):
    """Recursively convert numpy scalars/arrays to native Python types."""
    if isinstance(val, np.floating): return float(val)
    if isinstance(val, np.integer):  return int(val)
    if isinstance(val, np.ndarray):  return val.tolist()
    if isinstance(val, list):        return [_to_python(v) for v in val]
    if isinstance(val, dict):        return {k: _to_python(v) for k, v in val.items()}
    return val


# ══════════════════════════════════════════════════════════════════════════════
# LaTeX stripper (similarity anchors only)
# ══════════════════════════════════════════════════════════════════════════════

def _strip_latex(text: str) -> str:
    """
    Strip LaTeX math notation from an anchor string before embedding.
    ONLY called on chapter title / prev query / spoken answer strings.
    NEVER called on chunk text.
    """
    text = _LATEX_DISPLAY.sub(" ", text)
    text = _LATEX_INLINE.sub(" ", text)
    text = _LATEX_CMD.sub(" ", text)
    text = _LATEX_CMD_BARE.sub(" ", text)
    text = _LATEX_BRACES.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


# ══════════════════════════════════════════════════════════════════════════════
# Cosine similarity
# ══════════════════════════════════════════════════════════════════════════════

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# Semantic Follow-up Detector  (Three-Anchor Scoring)
# ══════════════════════════════════════════════════════════════════════════════

def _check_followup(
    query:    str,
    state:    RAGState,
    embed_fn,           # retriever.store.embed_query — already loaded, ~5 ms/call
) -> tuple[bool, float, str, dict]:
    """
    Decide whether `query` is a follow-up to the previous turn.

    Returns
    -------
    (is_followup, best_similarity, reason, anchor_scores)

    reason values:
      "hard_anchor"  — explicit chapter/activity/exercise number in query
      "intent_only"  — pure continuation signal with previous context
      "no_context"   — first turn or no previous chapter recorded
      "semantic"     — at least one anchor hit its threshold
      "fresh"        — no anchor hit; genuinely different topic

    anchor_scores keys:
      title, query, answer          — raw cosine scores
      title_hit, query_hit, answer_hit — whether each hit its threshold
    """
    empty_scores = {
        "title": 0.0,  "query": 0.0,  "answer": 0.0,
        "title_hit": False, "query_hit": False, "answer_hit": False,
    }

    # ── Priority 1: hard anchor ───────────────────────────────────────────────
    # Explicit "chapter N", "activity N", or "exercise N" in the query.
    # Always use that number directly — no memory injection needed.
    if _HARD_ANCHOR.search(query):
        log.info("Memory | hard_anchor detected in %r", query)
        return False, 0.0, "hard_anchor", empty_scores

    # ── Priority 2: intent-only queries ──────────────────────────────────────
    # Pure continuation signals: "why?", "example?", "explain more", etc.
    # These have zero domain words so embedding comparison is meaningless.
    if _INTENT_ONLY.match(query):
        has_ctx = bool(state.get("last_chapter_number"))
        if has_ctx:
            log.info("Memory | intent_only %r — forcing follow-up (has context)", query)
            return True, 1.0, "intent_only", {
                "title": 1.0, "query": 1.0, "answer": 1.0,
                "title_hit": True, "query_hit": True, "answer_hit": True,
            }
        else:
            log.info("Memory | intent_only %r — no context yet", query)
            return False, 0.0, "no_context", empty_scores

    # ── Priority 3: no previous context ──────────────────────────────────────
    last_title  = state.get("last_chapter_title")  or ""
    last_answer = state.get("last_spoken_answer")  or ""
    # FIX 3: use prev_query (previous turn's query), NOT last_query
    # (last_query in state IS the current query by the time parse_node runs)
    prev_q = state.get("prev_query") or ""

    if not state.get("last_chapter_number") or (not last_answer and not last_title):
        log.info("Memory | no_context — first turn or missing previous chapter")
        return False, 0.0, "no_context", empty_scores

    # ── Priority 4: three-anchor semantic scoring ─────────────────────────────
    try:
        q_vec = embed_fn(query)

        # Anchor 1 — chapter title
        # Short, precise. Low threshold (0.25) is safe.
        sim_title = 0.0
        if last_title:
            sim_title = _cosine(q_vec, embed_fn(_strip_latex(last_title)))

        # Anchor 2 — previous query  (FIX 3: use prev_q, not state["last_query"])
        # Best for vague follow-ups. Needs higher threshold (0.45) because
        # "explain X" and "explain Y" share intent words regardless of topic.
        sim_query = 0.0
        if prev_q and prev_q.strip().lower() != query.strip().lower():
            sim_query = _cosine(q_vec, embed_fn(_strip_latex(prev_q)))

        # Anchor 3 — spoken answer (LaTeX stripped, capped at 300 chars)
        sim_answer = 0.0
        if last_answer:
            sim_answer = _cosine(q_vec, embed_fn(_strip_latex(last_answer)[:300]))

        # Per-anchor threshold check — ANY hit → follow-up
        title_hit  = sim_title  >= TITLE_THRESHOLD
        query_hit  = sim_query  >= QUERY_THRESHOLD
        answer_hit = sim_answer >= ANSWER_THRESHOLD
        is_followup = title_hit or query_hit or answer_hit

        best_sim = max(sim_title, sim_query, sim_answer)

        anchor_scores = {
            "title":      round(sim_title,  3),
            "query":      round(sim_query,  3),
            "answer":     round(sim_answer, 3),
            "title_hit":  title_hit,
            "query_hit":  query_hit,
            "answer_hit": answer_hit,
        }

        hits = "+".join(
            k for k, v in [
                ("title",  title_hit),
                ("query",  query_hit),
                ("answer", answer_hit),
            ] if v
        ) or "none"

        log.info(
            "Memory | followup=%s  hits=%s  "
            "title=%.3f(th=%.2f)  query=%.3f(th=%.2f)  answer=%.3f(th=%.2f)  "
            "current_query=%r",
            is_followup, hits,
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
    Node 1: Semantic follow-up check → SLM parse → chapter/topic injection.

    Steps:
      1. Run three-anchor follow-up detection.
      2. Parse query with SLM + sanitize.
      3. Apply drift-reset counter.
      4. Inject last_chapter_number + last_topic when appropriate.
      5. Build resolved_query string for display.
    """
    query    = state["last_query"]
    embed_fn = retriever.store.embed_query   # reuses already-loaded model

    # Step 1 — follow-up detection
    is_followup, similarity, reason, anchor_scores = _check_followup(
        query, state, embed_fn
    )

    # Step 2 — SLM parse + sanitize
    try:
        raw    = parse_query_with_slm(query)
        parsed = json.loads(raw)
        parsed = sanitize(parsed, query)
        parsed["subject"] = state["subject"]   # hard override — never trust SLM here

        # ── Step 3: Drift-reset counter ──────────────────────────────────────
        # Track consecutive GENUINE fresh turns (topic changes).
        # Once DRIFT_RESET_AFTER genuine fresh turns accumulate,
        # release the chapter lock to prevent memory drift in long sessions.
        #
        # "no_context" means it is the very first turn of the session —
        # there is nothing to drift from.  It must NOT count against the
        # lock or it would prime the counter so that the very next fresh
        # query (Turn 2) immediately fires a drift reset.
        #
        # Only "fresh" (semantically different topic) counts as drift.
        prev_fresh_count = state.get("consecutive_fresh_turns", 0)

        if reason in ("semantic", "hard_anchor", "intent_only"):
            # Follow-up or explicit anchor → reset drift counter
            new_fresh_count = 0
        elif reason == "no_context":
            # First turn of session — no chapter lock exists yet, nothing to reset
            new_fresh_count = 0
        else:
            # reason == "fresh" — genuinely different topic, count it
            new_fresh_count = prev_fresh_count + 1

        # Apply drift reset when threshold reached
        chapter_locked = state.get("last_chapter_number")
        if new_fresh_count >= DRIFT_RESET_AFTER and chapter_locked:
            log.info(
                "Memory | DRIFT RESET after %d consecutive fresh turns — "
                "releasing chapter lock (was chapter=%s)",
                new_fresh_count, chapter_locked,
            )
            parsed["chapter_number"] = None   # ensure this turn retrieves fresh

        # ── Step 4: Chapter + Topic injection ────────────────────────────────
        # Only inject when:
        #   - follow-up detected
        #   - SLM found no explicit chapter in the query
        #   - drift reset did NOT just fire
        injected_chapter = None
        injected_topic   = None

        if is_followup and parsed.get("chapter_number") is None:
            if new_fresh_count < DRIFT_RESET_AFTER:
                prev_chapter = state.get("last_chapter_number")
                if prev_chapter:
                    try:
                        parsed["chapter_number"] = int(prev_chapter)
                        injected_chapter = parsed["chapter_number"]
                    except (ValueError, TypeError):
                        log.warning(
                            "Memory | could not cast last_chapter_number=%r to int",
                            prev_chapter,
                        )

        # Topic injection — anchor semantic search to the right section
        # Only when chapter was injected AND current topic is vague/empty
        if is_followup and injected_chapter is not None:
            current_topic = parsed.get("topic", "").strip()
            prev_topic    = (state.get("last_parsed") or {}).get("topic", "").strip()

            _VAGUE_TOPICS = {
                "unknown", "none", "", "topic",
                "what is it", "explain", "why", "how",
                "example", "simplify", "diagram",
            }
            topic_is_vague = (
                len(current_topic.split()) <= 4
                or current_topic.lower() in _VAGUE_TOPICS
            )

            if topic_is_vague and prev_topic:
                parsed["topic"] = prev_topic
                injected_topic  = prev_topic
                log.info(
                    "Memory | injected topic=%r (current was vague: %r)",
                    injected_topic, current_topic,
                )
            elif not current_topic and prev_topic:
                parsed["topic"] = prev_topic
                injected_topic  = prev_topic
                log.info("Memory | injected topic=%r (current was empty)", injected_topic)

        # ── Step 5: Build resolved_query display string ───────────────────────
        hits = "+".join(
            k for k, v in [
                ("title",  anchor_scores.get("title_hit",  False)),
                ("query",  anchor_scores.get("query_hit",  False)),
                ("answer", anchor_scores.get("answer_hit", False)),
            ] if v
        ) or "none"

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
        else:
            resolved = query

        log.info("Parsed: %s", json.dumps(parsed))

        # ── Step 6: Build enriched retrieval query ────────────────────────────
        # Problem: when memory is used, the student's query is vague ("explain it",
        # "elaborate", "what does that mean"). The bi-encoder inside retrieve_node
        # embeds this vague text and finds poor semantic matches even within the
        # correct chapter, because there are no domain words to anchor on.
        #
        # Solution: for follow-up turns, prepend prev_query so the bi-encoder
        # receives the full semantic context of what "it" / "that" refers to.
        #
        # Examples:
        #   "explain it"        + "explain how the heart works"
        #   → "explain how the heart works explain it"
        #   → bi-encoder now finds heart/circulatory chunks reliably ✓
        #
        #   "elaborate"         + "explain acids and bases"
        #   → "explain acids and bases elaborate"
        #   → correct chapter section retrieved ✓
        #
        # For fresh / hard_anchor / no_context: use raw query unchanged.
        # Hard-anchor queries already contain the chapter and topic.
        # Fresh queries have domain words and do not need enrichment.
        is_followup_turn = reason in ("intent_only", "semantic")
        prev_q_for_enrichment = state.get("prev_query") or ""

        if is_followup_turn and prev_q_for_enrichment:
            retrieval_query = f"{prev_q_for_enrichment} {query}"
            log.info(
                "Memory | retrieval_query enriched: %r + %r → %r",
                prev_q_for_enrichment, query, retrieval_query,
            )
        else:
            retrieval_query = query

        return {
            "last_parsed":              parsed,
            "resolved_query":           resolved,
            "retrieval_query":          retrieval_query,
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
            "retrieval_query":            query,
            "last_similarity":            float(similarity),
            "memory_used":                reason,
            "consecutive_fresh_turns":    state.get("consecutive_fresh_turns", 0) + 1,
            "error":                      f"Parse error: {e}",
        }


def retrieve_node(state: RAGState, retriever: Retriever) -> dict:
    """
    Node 2: Retrieve top-K chunks using parsed metadata.

    Uses `retrieval_query` (not `last_query`) for bi-encoder embedding.
    For follow-up turns, retrieval_query is enriched with prev_query so the
    bi-encoder has full semantic context even when last_query is vague
    ("explain it", "elaborate", "what does that mean").

    chapter_number in last_parsed is already injected by parse_node
    if this is a follow-up turn.
    """
    parsed          = state.get("last_parsed", {})
    # Use enriched retrieval_query built by parse_node.
    # Falls back to last_query if retrieval_query is missing (first turn safety).
    retrieval_query = state.get("retrieval_query") or state["last_query"]

    log.info("Retrieve | embedding query: %r", retrieval_query)
    chunks, err = retriever.retrieve_safe(parsed, retrieval_query)

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
    chunk_scores   = [float(c.score) for c in chunks]   # numpy.float32 → float

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
            "score":           float(c.score),
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
        "last_confidence":    float(answer.confidence),
        "last_warning":       answer.low_confidence_warning,
        "error":              None,
    }


def save_memory_node(state: RAGState) -> dict:
    """
    Node 4: Append this turn to conversation history and increment turn_count.

    Also advances prev_query ← last_query so the NEXT turn's Anchor 2 comparison
    uses the correct (current) query, not the incoming query of the next turn.

    AI message includes a hidden [MEMORY]…[/MEMORY] tag with retrieval metadata
    for diagnostic inspection without polluting the visible history.
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

    turn_count  = state.get("turn_count", 0) + 1
    fresh_count = state.get("consecutive_fresh_turns", 0)

    log.info(
        "Turn %d saved | student=%s | subject=%s | chapter=%s | chunks=%s | "
        "sim=%.3f | reason=%s | fresh_streak=%d",
        turn_count,
        state.get("student_id", "?"),
        state.get("subject", "?"),
        state.get("last_chapter_number", "?"),
        state.get("last_chunk_ids", []),
        state.get("last_similarity", 0.0),
        state.get("memory_used", "?"),
        fresh_count,
    )

    return {
        "messages":   messages,
        "turn_count": turn_count,
        # FIX 3: advance prev_query so next turn's Anchor 2 is correct
        "prev_query": state["last_query"],
    }


# ══════════════════════════════════════════════════════════════════════════════
# Conditional edge
# ══════════════════════════════════════════════════════════════════════════════

def _should_generate(state: RAGState) -> str:
    return "generate" if state.get("retrieved_chunks") else "save_memory"


# ══════════════════════════════════════════════════════════════════════════════
# Graph builder
# ══════════════════════════════════════════════════════════════════════════════

def build_rag_graph(retriever: Retriever, generator: Generator) -> tuple:
    """Build and compile the LangGraph StateGraph with SqliteSaver checkpointing."""

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
    # returning a context manager in newer LangGraph versions.
    conn         = sqlite3.connect(str(MEMORY_DB_PATH), check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    graph = builder.compile(checkpointer=checkpointer)
    log.info("LangGraph compiled — checkpointing to %s", MEMORY_DB_PATH)
    return graph, checkpointer


# ══════════════════════════════════════════════════════════════════════════════
# Thread-ID helper
# FIX 1 + FIX 5: Thread ID is (student_id + "::" + subject) so each
# student×subject pair has a completely isolated SQLite checkpoint thread.
# Memory NEVER leaks across subjects.
# ══════════════════════════════════════════════════════════════════════════════

def _make_thread_id(student_id: str, subject: str) -> str:
    """
    Build a deterministic, unique thread ID from student identity and subject.

    Format: "<student_id>::<subject_lowercase>"
    Examples:
      alice  + Biology   → "alice::biology"
      alice  + Physics   → "alice::physics"   (different thread, different memory)
      bob    + Biology   → "bob::biology"
    """
    return f"{student_id}::{subject.strip().lower()}"


# ══════════════════════════════════════════════════════════════════════════════
# Public API: MemoryGraph
# ══════════════════════════════════════════════════════════════════════════════

class MemoryGraph:
    """
    High-level wrapper around the compiled LangGraph RAG pipeline.

    Each (student_id, subject) pair is a fully isolated session in SQLite.
    Switching subject starts a clean session — no memory leaks.

    Usage
    -----
        mg = MemoryGraph(retriever, generator)

        # Turn 1 — explicit chapter (hard_anchor)
        r = mg.run("explain activity 2 from chapter 1",
                   student_id="alice", subject="Biology")

        # Turn 2 — follow-up, no chapter in query
        r = mg.run("what are the materials needed",
                   student_id="alice", subject="Biology")
        # r["last_chapter_number"] == "1"   ← memory worked
        # r["memory_used"]  == "semantic"   ← or "title" etc.

        # Turn 3 — subject switch → clean slate
        r = mg.run("explain Newton's first law",
                   student_id="alice", subject="Physics")
        # r["last_chapter_number"] == <whatever Physics returns>
        # NO Biology context leaked
    """

    def __init__(self, retriever: Retriever, generator: Generator):
        self._retriever = retriever
        self.graph, self._checkpointer = build_rag_graph(retriever, generator)

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _config(self, student_id: str, subject: str) -> dict:
        """Return LangGraph config dict for this student+subject session."""
        return {"configurable": {"thread_id": _make_thread_id(student_id, subject)}}

    def _load_prev(self, student_id: str, subject: str) -> dict:
        """Load previous state for this student+subject, or return empty dict."""
        existing = self.graph.get_state(self._config(student_id, subject))
        if existing and existing.values:
            return existing.values
        return {}

    # ── Public API ─────────────────────────────────────────────────────────────

    def run(self, query: str, student_id: str, subject: str) -> RAGState:
        """
        Run one full turn of the RAG pipeline with semantic memory.

        Subject-change handling (FIX 2):
          If the student switches subject, this method automatically uses a
          different SQLite thread (FIX 1) which starts with a clean state.
          No explicit reset needed — the new thread simply has no history.
          The old subject's thread remains intact in SQLite for future use.

        Returns the final RAGState dict.  Key fields:
          last_spoken_answer, last_display_md,
          last_chapter_number, last_chapter_title,
          last_chunk_ids, last_chunk_scores,
          last_similarity, memory_used,
          last_confidence, turn_count, error
        """
        config = self._config(student_id, subject)
        prev   = self._load_prev(student_id, subject)

        # Build input state, carrying forward all relevant previous-turn fields.
        # FIX 3: prev_query comes from the STORED prev_query field (which
        # save_memory_node writes as last_query of the previous turn).
        # It is NEVER the current query — that arrives in last_query below.
        input_state: RAGState = {
            # Identity
            "student_id":  student_id,
            "subject":     subject,

            # Conversation history
            "messages":    prev.get("messages", []),

            # Current turn — new query in last_query, previous query in prev_query
            "last_query":      query,
            "prev_query":      prev.get("prev_query"),   # previous turn's query (Anchor 2)
            "resolved_query":  query,
            "retrieval_query": query,   # parse_node will enrich this for follow-ups

            # Carry forward parsed dict for topic injection
            "last_parsed":   prev.get("last_parsed", {}),

            # Carry forward retrieval context (used by follow-up detector)
            "last_chapter_number": prev.get("last_chapter_number"),
            "last_chapter_title":  prev.get("last_chapter_title"),
            "last_chunk_ids":      prev.get("last_chunk_ids", []),
            "last_chunk_scores":   prev.get("last_chunk_scores", []),
            "last_filter_path":    prev.get("last_filter_path"),

            # Carry forward generation context
            "last_answer_type":   prev.get("last_answer_type"),
            "last_spoken_answer": prev.get("last_spoken_answer"),  # Anchor 3
            "last_display_md":    prev.get("last_display_md"),
            "last_confidence":    prev.get("last_confidence"),
            "last_warning":       prev.get("last_warning"),

            # Memory diagnostics
            "last_similarity": prev.get("last_similarity"),
            "memory_used":     prev.get("memory_used"),

            # Session bookkeeping
            "consecutive_fresh_turns": prev.get("consecutive_fresh_turns", 0),
            "turn_count":              prev.get("turn_count", 0),
            "error":                   None,
            "retrieved_chunks":        [],
        }

        return self.graph.invoke(input_state, config=config)

    def get_history(self, student_id: str, subject: str) -> list[dict]:
        """
        Return conversation history for this student+subject session.
        [MEMORY] tags are stripped from AI messages.

        FIX 5: Uses subject-scoped thread ID.
        """
        config = self._config(student_id, subject)
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

    def get_session_summary(self, student_id: str, subject: str) -> dict:
        """
        Compact summary of what this student has studied in the given subject.

        FIX 5: Uses subject-scoped thread ID.
        """
        config = self._config(student_id, subject)
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
            "prev_query":               v.get("prev_query"),
        }

    def list_sessions(self, student_id: str) -> list[str]:
        """
        List all subjects this student has previously studied
        (i.e. all thread IDs that start with '<student_id>::').
        Useful for showing a student their study history across subjects.
        """
        # SqliteSaver stores thread_id in the checkpoints table
        try:
            conn   = self._checkpointer.conn
            cursor = conn.execute(
                "SELECT DISTINCT thread_id FROM checkpoints WHERE thread_id LIKE ?",
                (f"{student_id}::%",),
            )
            rows = cursor.fetchall()
            subjects = []
            prefix = f"{student_id}::"
            for (tid,) in rows:
                if tid.startswith(prefix):
                    subjects.append(tid[len(prefix):])
            return subjects
        except Exception as e:
            log.warning("Could not list sessions for %s: %s", student_id, e)
            return []


# ══════════════════════════════════════════════════════════════════════════════
# CLI smoke-test
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    student = "test_student"

    print("\n Loading pipeline ...")
    retriever = Retriever()
    generator = Generator()
    mg        = MemoryGraph(retriever, generator)
    print("  MemoryGraph ready.\n")

    # ── Test sequence 1: Biology session ──────────────────────────────────────
    bio_queries = [
        ("Biology", "explain activity 2 from chapter 1"),   # hard_anchor → ch1
        ("Biology", "what are the materials needed"),        # semantic    → ch1
        ("Biology", "what is the conclusion"),               # semantic    → ch1
        ("Biology", "explain Newton's laws"),                # fresh       → different chapter
        ("Biology", "give me more detail"),                  # semantic    → Newton chapter
    ]

    # ── Test sequence 2: subject switch ──────────────────────────────────────
    switch_queries = [
        ("Physics", "explain Newton's first law chapter 2"),  # hard_anchor, Physics
        ("Physics", "what is an example of that"),            # semantic    → ch2 Physics
        ("Biology", "what are the materials needed"),         # back to Biology — should reload Biology context
    ]

    all_tests = [
        ("Biology", bio_queries),
        ("Subject switch", switch_queries),
    ]

    for group_name, tests in all_tests:
        print(f"\n{'═'*60}")
        print(f"  TEST GROUP: {group_name}")
        print(f"{'═'*60}")

        for subject, q in tests:
            print(f"\n  Subject : {subject!r}")
            print(f"  Query   : {q!r}")
            result = mg.run(q, student_id=student, subject=subject)

            print(f"  Memory  : reason={result.get('memory_used')}  "
                  f"sim={result.get('last_similarity', 0.0):.3f}")
            print(f"  Resolved: {result.get('resolved_query')!r}")
            print(f"  Chapter : {result.get('last_chapter_number')} "
                  f"— {result.get('last_chapter_title')}")
            print(f"  Chunks  : {result.get('last_chunk_ids')}")
            print(f"  prev_q  : {result.get('prev_query')!r}")

            if result.get("error"):
                print(f"  ⚠ Error : {result['error']}")
            else:
                s = result.get("last_spoken_answer") or ""
                print(f"  Spoken  : {s[:120]}{'...' if len(s) > 120 else ''}")

    print(f"\n{'═'*60}")
    print("\nBiology session summary:")
    print(json.dumps(mg.get_session_summary(student, "Biology"), indent=2))
    print("\nPhysics session summary:")
    print(json.dumps(mg.get_session_summary(student, "Physics"), indent=2))
    print("\nPrevious subjects studied:")
    print(mg.list_sessions(student))