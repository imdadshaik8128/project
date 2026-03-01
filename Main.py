"""
Main.py — Textbook Q&A Chat Interface  [Memory-Enhanced]
==========================================================
Run:
    python Main.py

Changes from original:
  - Replaced direct pipeline calls with MemoryGraph.run()
  - Student name prompt added at startup (used as SQLite thread key)
  - 'history' command shows conversation history for current student
  - 'summary' command shows what chapters/chunks have been studied
  - All conversation state auto-persisted to memory.db (SQLite)
  - Follow-up queries ("explain that again") auto-resolved using last turn state

Session flow:
  1. Student enters their name (persists their history across restarts)
  2. User selects a subject from a numbered menu
  3. Subject is locked for the session — all queries filtered to that subject
  4. Each query runs through LangGraph: resolve → parse → retrieve → generate → save
  5. Commands: 'switch' | 'history' | 'summary' | 'help' | 'exit'
"""

from __future__ import annotations

import json
import sys
import time

# ── Rich terminal renderer ─────────────────────────────────────────────────────
try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.rule import Rule
    from rich.table import Table
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None

# ── TTS ────────────────────────────────────────────────────────────────────────
try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False

from retriever import Retriever
from generator import Generator
from memory_graph import MemoryGraph

# ── Constants ──────────────────────────────────────────────────────────────────
TTS_RATE   = 165
TTS_VOLUME = 0.9

AVAILABLE_SUBJECTS = [
    "Biology",
    "Economics",
    "Geography",
    "History",
    "Maths_sem_1",
    "Maths_sem_2",
    "Physics",
    "Social_political",
]


# ══════════════════════════════════════════════════════════════════════════════
# ANSI colour helpers
# ══════════════════════════════════════════════════════════════════════════════

def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m"

BOLD   = lambda t: _c("1",  t)
GREEN  = lambda t: _c("32", t)
YELLOW = lambda t: _c("33", t)
CYAN   = lambda t: _c("36", t)
RED    = lambda t: _c("31", t)
DIM    = lambda t: _c("2",  t)


# ══════════════════════════════════════════════════════════════════════════════
# Print helpers
# ══════════════════════════════════════════════════════════════════════════════

def _print(text: str, style: str = "") -> None:
    if RICH_AVAILABLE:
        console.print(text, style=style)
    else:
        print(text)

def _rule(title: str = "") -> None:
    if RICH_AVAILABLE:
        console.print(Rule(title, style="bold cyan"))
    else:
        label = f"  {title}  " if title else ""
        print(f"\n{'─' * 64}{label}")


# ══════════════════════════════════════════════════════════════════════════════
# Banner
# ══════════════════════════════════════════════════════════════════════════════

def print_banner() -> None:
    banner = """
╔══════════════════════════════════════════════════════════════╗
║      Textbook Q&A  —  RAG System  (offline + memory)         ║
║      Retrieval · Generation · Display · TTS · LangGraph      ║
╚══════════════════════════════════════════════════════════════╝
"""
    if RICH_AVAILABLE:
        console.print(banner, style="bold cyan")
    else:
        print(BOLD(banner))


# ══════════════════════════════════════════════════════════════════════════════
# Student identification
# ══════════════════════════════════════════════════════════════════════════════

def get_student_id() -> str:
    """Ask for student name — used as the SQLite thread key for memory persistence."""
    if RICH_AVAILABLE:
        console.print("\n  [bold]Enter your name (used to save your study history):[/bold]")
    else:
        print(BOLD("\n  Enter your name (used to save your study history):"))

    while True:
        try:
            name = input(BOLD("  Name › ")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            sys.exit(0)

        if name:
            student_id = name.lower().replace(" ", "_")
            if RICH_AVAILABLE:
                console.print(f"\n  [bold green]✓ Welcome, {name}!  (id: {student_id})[/bold green]\n")
            else:
                print(GREEN(f"\n  ✓ Welcome, {name}!  (id: {student_id})\n"))
            return student_id

        if RICH_AVAILABLE:
            console.print("  [red]Please enter a name.[/red]")
        else:
            print(RED("  Please enter a name."))


# ══════════════════════════════════════════════════════════════════════════════
# Subject selection
# ══════════════════════════════════════════════════════════════════════════════

def select_subject() -> str:
    if RICH_AVAILABLE:
        console.print("\n  [bold]Select a subject to study:[/bold]\n")
        for i, subj in enumerate(AVAILABLE_SUBJECTS, 1):
            console.print(f"    [cyan]{i}[/cyan].  {subj}")
        console.print()
    else:
        print(BOLD("\n  Select a subject to study:\n"))
        for i, subj in enumerate(AVAILABLE_SUBJECTS, 1):
            print(f"    {CYAN(str(i))}.  {subj}")
        print()

    while True:
        try:
            choice = input(BOLD("  Enter number › ")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            sys.exit(0)

        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(AVAILABLE_SUBJECTS):
                selected = AVAILABLE_SUBJECTS[idx]
                if RICH_AVAILABLE:
                    console.print(f"\n  [bold green]✓ Subject locked: {selected}[/bold green]\n")
                else:
                    print(GREEN(f"\n  ✓ Subject locked: {selected}\n"))
                return selected

        if RICH_AVAILABLE:
            console.print(f"  [red]Invalid. Enter a number 1–{len(AVAILABLE_SUBJECTS)}.[/red]")
        else:
            print(RED(f"  Invalid. Enter a number 1–{len(AVAILABLE_SUBJECTS)}."))


# ══════════════════════════════════════════════════════════════════════════════
# TTS
# ══════════════════════════════════════════════════════════════════════════════

def _init_tts():
    if not TTS_AVAILABLE:
        return None
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate",   TTS_RATE)
        engine.setProperty("volume", TTS_VOLUME)
        voices = engine.getProperty("voices")
        if voices:
            for v in voices:
                if "english" in v.name.lower() or "en" in v.id.lower():
                    engine.setProperty("voice", v.id)
                    break
        return engine
    except Exception as e:
        _print(f"  ⚠  TTS init failed: {e}", style="yellow")
        return None

def speak(engine, text: str) -> None:
    if engine is None:
        _print("  (TTS unavailable — install pyttsx3 + espeak)", style="dim yellow")
        return
    try:
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        _print(f"  (TTS error: {e})", style="dim yellow")


# ══════════════════════════════════════════════════════════════════════════════
# Display helpers
# ══════════════════════════════════════════════════════════════════════════════

def display_result(state: dict) -> None:
    """Render the full result from a MemoryGraph.run() call."""

    _rule()

    # Error with no content to show
    if state.get("error") and not state.get("last_display_md"):
        if RICH_AVAILABLE:
            console.print(Panel(
                f"✗  {state['error']}\n\nCheck chapter number or activity number.",
                style="bold red",
                title="Error",
            ))
        else:
            print(RED(f"\n  ✗ {state['error']}"))
        _rule()
        return

    # Answer type badge
    answer_type = state.get("last_answer_type", "concept")
    if RICH_AVAILABLE:
        badge = (
            "[bold green]● REFERENCE LOOKUP[/bold green]"
            if answer_type == "reference"
            else "[bold blue]● CONCEPT EXPLANATION[/bold blue]"
        )
        console.print(badge)
    else:
        print(f"[ {answer_type.upper()} ]")

    # Follow-up resolution notice
    raw_q      = state.get("last_query", "")
    resolved_q = state.get("resolved_query", "")
    if resolved_q and resolved_q != raw_q:
        _print(f"  ↳ Follow-up resolved: {resolved_q!r}", style="dim yellow")

    # Low confidence warning
    if state.get("last_warning"):
        if RICH_AVAILABLE:
            console.print(Panel(
                f"⚠  {state['last_warning']}",
                style="bold yellow",
                title="Low Confidence Warning",
            ))
        else:
            print(YELLOW(f"\n⚠  {state['last_warning']}"))

    # Confidence bar
    conf = state.get("last_confidence") or 0.0
    pct  = int(conf * 100)
    bar  = "█" * (pct // 5) + "░" * (20 - pct // 5)
    _print(f"  Confidence: [{bar}] {pct}%", style="dim")

    # Memory context line — turn, chapter, chunk IDs + scores
    ch_num    = state.get("last_chapter_number") or "?"
    ch_title  = state.get("last_chapter_title")  or ""
    chunk_ids = state.get("last_chunk_ids", [])
    scores    = state.get("last_chunk_scores", [])
    turn      = state.get("turn_count", 0)
    chunk_score_pairs = " | ".join(
        f"{cid}={sc:.3f}" for cid, sc in zip(chunk_ids, scores)
    )
    sim         = state.get("last_similarity") or 0.0
    mem_reason  = state.get("memory_used") or "—"
    # Colour-code memory reason for quick reading during demo
    if RICH_AVAILABLE:
        reason_colour = {
            "semantic":    "bold green",
            "hard_anchor": "bold cyan",
            "fresh":       "bold yellow",
            "no_context":  "dim",
        }.get(mem_reason, "dim")
        mem_str = f"[{reason_colour}]{mem_reason}[/{reason_colour}]"
    else:
        mem_str = mem_reason

    if RICH_AVAILABLE:
        console.print(
            f"  [dim]Turn {turn}  ·  Chapter {ch_num}"
            + (f" — {ch_title}" if ch_title else "")
            + (f"  ·  {chunk_score_pairs}" if chunk_score_pairs else "")
            + f"[/dim]  Memory: {mem_str} [dim](sim={sim:.2f})[/dim]"
        )
    else:
        print(DIM(
            f"  Turn {turn}  ·  Chapter {ch_num} — {ch_title}"
            + (f"  ·  {chunk_score_pairs}" if chunk_score_pairs else "")
            + f"  ·  memory={mem_reason} (sim={sim:.2f})"
        ))

    # ── Display answer (markdown) ──────────────────────────────────────────────
    _rule("DISPLAY ANSWER")
    display_md = state.get("last_display_md", "")
    if RICH_AVAILABLE:
        console.print(Markdown(display_md))
    else:
        import re
        plain = re.sub(r"#{1,6}\s+", "", display_md)
        plain = re.sub(r"\*{1,3}(.+?)\*{1,3}", r"\1", plain)
        plain = re.sub(r"`(.+?)`", r"\1", plain)
        print(plain)

    # Filter path
    if state.get("last_filter_path"):
        _print(f"\n  Filter path: {state['last_filter_path']}", style="dim")

    # ── Spoken answer ─────────────────────────────────────────────────────────
    _rule("SPOKEN ANSWER  (TTS)")
    spoken = state.get("last_spoken_answer", "")
    if RICH_AVAILABLE:
        console.print(Panel(
            spoken,
            style="italic green",
            title="🔊 Speaking …",
        ))
    else:
        print(f"\n🔊  {spoken}\n")

    _rule()


def display_history(history: list[dict], student_id: str) -> None:
    _rule(f"HISTORY — {student_id}")
    if not history:
        _print("  No history yet for this student.", style="dim")
        _rule()
        return

    if RICH_AVAILABLE:
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("#",      style="dim",   width=4)
        table.add_column("Role",   style="bold",  width=8)
        table.add_column("Message")
        for i, msg in enumerate(history, 1):
            role  = "You" if msg["role"] == "human" else "AI"
            color = "cyan" if msg["role"] == "human" else "green"
            text  = msg["content"][:120] + ("…" if len(msg["content"]) > 120 else "")
            table.add_row(str(i), f"[{color}]{role}[/{color}]", text)
        console.print(table)
    else:
        for i, msg in enumerate(history, 1):
            role = "You" if msg["role"] == "human" else "AI"
            print(f"  [{i}] {BOLD(role)}: {msg['content'][:120]}")
    _rule()


def display_summary(summary: dict, student_id: str) -> None:
    _rule(f"STUDY SUMMARY — {student_id}")
    if not summary:
        _print("  No study data yet for this student.", style="dim")
        _rule()
        return

    chunk_ids = summary.get("last_chunk_ids", [])
    scores    = summary.get("last_chunk_scores", [])
    chunk_str = " | ".join(
        f"{cid} ({sc:.3f})" for cid, sc in zip(chunk_ids, scores)
    ) or "—"

    lines = [
        ("Subject",      summary.get("subject", "—")),
        ("Total turns",  str(summary.get("turn_count", 0))),
        ("Last Chapter", f"{summary.get('last_chapter_number','—')} — {summary.get('last_chapter_title','—')}"),
        ("Last Chunks",  chunk_str),
        ("Answer Type",  summary.get("last_answer_type",  "—")),
        ("Confidence",   f"{(summary.get('last_confidence') or 0.0):.0%}"),
    ]

    if RICH_AVAILABLE:
        table = Table(show_header=False)
        table.add_column("Field", style="bold cyan", width=16)
        table.add_column("Value", style="white")
        for k, v in lines:
            table.add_row(k, v)
        console.print(table)
    else:
        for k, v in lines:
            print(f"  {CYAN(k):20s} {v}")
    _rule()


def print_help(subject: str) -> None:
    if RICH_AVAILABLE:
        console.print(f"""
  [bold]Commands:[/bold]
    [cyan]switch[/cyan]   — change subject  (currently: [green]{subject}[/green])
    [cyan]history[/cyan]  — show your full conversation history
    [cyan]summary[/cyan]  — show your study progress (chapters + chunks covered)
    [cyan]help[/cyan]     — show this message
    [cyan]exit[/cyan]     — quit

  [bold]Example queries:[/bold]
    explain activity 2 from chapter 1
    what is photosynthesis chapter 3
    solve exercise 3.1 chapter 4
    explain that again          ← follow-up (memory resolves automatically)
    give me another example     ← follow-up (memory resolves automatically)
""")
    else:
        print(f"""
  Commands:
    switch   — change subject  (currently: {subject})
    history  — show your full conversation history
    summary  — show chapters and chunks studied so far
    help     — show this message
    exit     — quit

  Example queries:
    explain activity 2 from chapter 1
    what is photosynthesis chapter 3
    solve exercise 3.1 chapter 4
    explain that again          ← follow-up (uses memory)
""")


# ══════════════════════════════════════════════════════════════════════════════
# Main chat loop
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print_banner()

    # ── Load everything once ───────────────────────────────────────────────────
    _print("⏳  Loading retriever (bi-encoder + cross-encoder) …\n", style="yellow")
    retriever = Retriever()

    _print("⏳  Loading generator …", style="yellow")
    generator = Generator()

    _print("⏳  Initialising LangGraph memory (SQLite → memory.db) …", style="yellow")
    memory_graph = MemoryGraph(retriever, generator)

    _print("🔊  Initialising TTS …", style="dim")
    tts_engine = _init_tts()

    if not TTS_AVAILABLE:
        _print(
            "  ⚠  pyttsx3 not found — spoken answer will display but not play.\n"
            "     Install: pip install pyttsx3\n"
            "     Linux  : sudo apt-get install espeak",
            style="yellow",
        )

    _print("\n✓  System ready.\n", style="bold green")

    # ── Student identification ─────────────────────────────────────────────────
    student_id = get_student_id()

    # ── Subject selection ──────────────────────────────────────────────────────
    active_subject = select_subject()

    if RICH_AVAILABLE:
        console.print(
            f"  Type [cyan]help[/cyan] for all commands. "
            f"Your history is saved across sessions in [dim]memory.db[/dim].\n"
        )
    else:
        print("  Type 'help' for commands. Your history is saved in memory.db.\n")

    _rule()

    # ── Chat loop ──────────────────────────────────────────────────────────────
    while True:
        try:
            if RICH_AVAILABLE:
                console.print(
                    f"\n[bold cyan][{active_subject}] {student_id} ›[/bold cyan] ",
                    end="",
                )
                query = input()
            else:
                query = input(BOLD(f"\n[{active_subject}] {student_id} › "))

            query = query.strip()

        except (EOFError, KeyboardInterrupt):
            print()
            _print("\n  Goodbye!\n", style="bold cyan")
            sys.exit(0)

        if not query:
            continue

        cmd = query.lower()

        if cmd in {"exit", "quit", "q"}:
            _print("\n  Goodbye!\n", style="bold cyan")
            sys.exit(0)

        if cmd == "switch":
            print()
            active_subject = select_subject()
            _rule()
            continue

        if cmd == "history":
            history = memory_graph.get_history(student_id)
            display_history(history, student_id)
            continue

        if cmd == "summary":
            summary = memory_graph.get_session_summary(student_id)
            display_summary(summary, student_id)
            continue

        if cmd == "help":
            print_help(active_subject)
            continue

        print()

        # ── Run full pipeline through LangGraph ────────────────────────────────
        _print("  Running pipeline …", style="dim")
        t0 = time.perf_counter()

        result = memory_graph.run(
            query      = query,
            student_id = student_id,
            subject    = active_subject,
        )

        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        _print(
            f"  ✓ Done in {elapsed_ms}ms  |  "
            f"turn={result.get('turn_count')}  |  "
            f"chapter={result.get('last_chapter_number')}  |  "
            f"chunks={result.get('last_chunk_ids')}",
            style="dim green",
        )

        display_result(result)
        if result.get("last_spoken_answer"):
            speak(tts_engine, result["last_spoken_answer"])


if __name__ == "__main__":
    main()