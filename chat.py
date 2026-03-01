"""
chat.py — Interactive RAG Chat Loop
=====================================
Run:
    python chat.py

Session flow:
  1. User selects a subject ONCE at startup from a numbered menu
  2. All queries in this session are locked to that subject
  3. Loops until user types 'exit' or 'quit'
  4. User can switch subject mid-session with the 'switch' command
"""

import json
import sys
import textwrap

from query_parser_v2 import parse_query_with_slm
from parse_sanitizer import sanitize
from retriever import Retriever, AmbiguityError

# ── Subjects must match `subject` field values in all_chunks.json exactly ─────
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

# ── ANSI colours ──────────────────────────────────────────────────────────────
def _c(code: str, text: str) -> str:
    try:
        return f"\033[{code}m{text}\033[0m"
    except Exception:
        return text

BOLD   = lambda t: _c("1", t)
GREEN  = lambda t: _c("32", t)
YELLOW = lambda t: _c("33", t)
CYAN   = lambda t: _c("36", t)
RED    = lambda t: _c("31", t)
DIM    = lambda t: _c("2", t)

DIVIDER = DIM("─" * 64)


def print_banner():
    print(BOLD("""
╔══════════════════════════════════════════════════════════════╗
║          Textbook Q&A — RAG System (offline)                 ║
╚══════════════════════════════════════════════════════════════╝
"""))


def select_subject() -> str:
    """Show a numbered menu and return the chosen subject string."""
    print(BOLD("  Select a subject to study:\n"))
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
                print(GREEN(f"\n  ✓ Subject locked: {selected}\n"))
                return selected

        print(RED(f"  Invalid choice. Enter a number between 1 and {len(AVAILABLE_SUBJECTS)}."))


def print_result(i: int, r) -> None:
    print(f"\n{BOLD(f'[{i}]')}  {CYAN(r.chunk_id)}  {DIM(f'score={r.score}')}")
    print(f"  {DIM('Chapter  :')} {r.chapter_number} — {r.chapter_title}")
    if r.section_title:
        print(f"  {DIM('Section  :')} {r.section_title}")
    print(f"  {DIM('Type     :')} {r.chunk_type}"
          + (f"  activity={r.activity_number}" if r.activity_number else ""))
    print(f"  {DIM('Filter   :')} {r.filter_path}")
    preview = textwrap.fill(
        r.text[:300].strip().replace("\n", " ") + " …",
        width=70, initial_indent="  ", subsequent_indent="  "
    )
    print(f"\n{preview}")


def print_help(subject: str):
    print(f"""
  {BOLD('Commands:')}
    {CYAN('switch')}   — change subject (currently: {subject})
    {CYAN('exit')}     — quit
  {BOLD('Example queries:')}
    explain activity 2 from chapter 1
    what is photosynthesis chapter 3
    solve exercise 3.1 chapter 4
""")


def run():
    print_banner()

    # ── Load retriever once ────────────────────────────────────────────────────
    print(YELLOW("⏳  Loading retriever (encodes all chunks on first run) …\n"))
    retriever = Retriever()
    print(GREEN("✓  Retriever ready.\n"))

    # ── Subject selection ──────────────────────────────────────────────────────
    active_subject = select_subject()

    print(f"  Type {CYAN('help')} for usage. Type {CYAN('switch')} to change subject.\n")
    print(DIVIDER)

    # ── Chat loop ─────────────────────────────────────────────────────────────
    while True:
        try:
            raw_query = input(
                BOLD(f"\n[{active_subject}] You › ")
            ).strip()
        except (EOFError, KeyboardInterrupt):
            print("\n" + YELLOW("Bye!"))
            sys.exit(0)

        if not raw_query:
            continue

        # ── Built-in commands ──────────────────────────────────────────────────
        cmd = raw_query.lower()

        if cmd in {"exit", "quit", "q"}:
            print(YELLOW("Bye!"))
            sys.exit(0)

        if cmd == "switch":
            print()
            active_subject = select_subject()
            continue

        if cmd == "help":
            print_help(active_subject)
            continue

        print()

        # ── Step 1: SLM parse ─────────────────────────────────────────────────
        try:
            raw_parse = parse_query_with_slm(raw_query)
            parsed: dict = json.loads(raw_parse)
        except Exception as e:
            print(RED(f"  ✗ Parser error: {e}"))
            print(DIVIDER)
            continue

        # Sanitize — strip hallucinated chapter/activity numbers
        parsed = sanitize(parsed, raw_query)

        # Inject session subject — always overrides whatever SLM emitted
        parsed["subject"] = active_subject

        print(DIM(f"  ↳ Parsed: {json.dumps(parsed)}"))
        print()

        # ── Step 2: Retrieve ──────────────────────────────────────────────────
        results, err = retriever.retrieve_safe(parsed, raw_query)

        if err:
            print(RED(f"  ✗ {err}"))
            print(DIVIDER)
            continue

        if not results:
            print(YELLOW("  No results found. Try rephrasing."))
            print(DIVIDER)
            continue

        print(GREEN(f"  ✓ Top {len(results)} chunks  |  Subject: {active_subject}"))
        print(DIVIDER)

        for i, r in enumerate(results, 1):
            print_result(i, r)

        print(f"\n{DIVIDER}")


if __name__ == "__main__":
    run()
