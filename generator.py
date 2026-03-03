"""
generator.py — LangChain Answer Generator
==========================================
Migrated from raw requests → LangChain LCEL chains:

  - _build_reference_chain / _build_concept_chain : PromptTemplate | OllamaLLM
  - JSON parsing: JsonOutputParser with fallback repair (preserved from original)
  - Output schema: GeneratedAnswer (unchanged)

Two answer types, each with its own LCEL chain:
  - "reference" : activity / exercise lookup  → present the content directly
  - "concept"   : topic / concept query       → synthesise + explain across chunks

Architecture (unchanged from original):
  Retriever  →  [RetrievedChunk list]
                        │
                  Generator.generate()
                        │
               ┌────────┴────────┐
          spoken_answer     display_answer_markdown
               │
              TTS
"""

from __future__ import annotations

import json
import logging
import re
import math
from dataclasses import dataclass, field
from typing import Optional

# ── LangChain imports ──────────────────────────────────────────────────────────
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
OLLAMA_MODEL       = "mistral"
LLM_TIMEOUT        = 60
CONFIDENCE_FLOOR   = 0.40
REFERENCE_CONFIDENCE = 1.0

# ── Shared LangChain LLM instance ─────────────────────────────────────────────
_llm = OllamaLLM(
    model=OLLAMA_MODEL,
    temperature=0.2,
    top_p=0.9,
    timeout=LLM_TIMEOUT,
)

_str_parser = StrOutputParser()


# ══════════════════════════════════════════════════════════════════════════════
# Output schema  (unchanged from original)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Citation:
    chunk_id:        str
    chapter_number:  str
    chapter_title:   str
    section_title:   str
    chunk_type:      str
    activity_number: str


@dataclass
class GeneratedAnswer:
    answer_type:              str
    spoken_answer:            str
    display_answer_markdown:  str
    citations:                list[Citation]
    confidence:               float
    low_confidence_warning:   Optional[str]
    filter_path:              str

    def to_dict(self) -> dict:
        return {
            "answer_type":             self.answer_type,
            "spoken_answer":           self.spoken_answer,
            "display_answer_markdown": self.display_answer_markdown,
            "citations": [
                {
                    "chunk_id":        c.chunk_id,
                    "chapter_number":  c.chapter_number,
                    "chapter_title":   c.chapter_title,
                    "section_title":   c.section_title,
                    "chunk_type":      c.chunk_type,
                    "activity_number": c.activity_number,
                }
                for c in self.citations
            ],
            "confidence":              round(self.confidence, 4),
            "low_confidence_warning":  self.low_confidence_warning,
            "filter_path":             self.filter_path,
        }


# ══════════════════════════════════════════════════════════════════════════════
# LangChain LCEL Prompt Templates
# ══════════════════════════════════════════════════════════════════════════════

_REFERENCE_TEMPLATE = PromptTemplate(
    input_variables=["query", "context"],
    template="""You are a textbook assistant for school students.
A student asked: "{query}"

The following content was retrieved from the textbook. Present it clearly.
Do NOT add information not present in the chunks below.

---
{context}
---

You MUST respond with ONLY a JSON object. Follow these rules exactly:
- Start your response with {{ and end with }}
- Use double quotes " for all keys and string values
- Use \\n inside strings for line breaks. NEVER use triple quotes.
- Do NOT include markdown fences (no ```)
- Do NOT add any text before or after the JSON

The JSON must have EXACTLY these two keys:

"spoken_answer": A plain natural explanation in 3-5 sentences for text-to-speech.
No bullet points. No markdown symbols. No asterisks. No hashes. Write naturally.

"display_answer_markdown": A markdown response using \\n for newlines. Include:
- A heading with the activity title and chapter
- A short explanation paragraph
- Key steps as a bullet list using - prefix
- A Source line at the bottom

Example of correct format:
{{"spoken_answer": "This activity is about plants. Students observe leaves.", "display_answer_markdown": "## Activity\\n\\nThis activity explains plants.\\n\\n- Step one\\n- Step two\\n\\n**Source:** Chapter 3"}}

Now respond with the JSON for the student query above:"""
)

_CONCEPT_TEMPLATE = PromptTemplate(
    input_variables=["query", "context"],
    template="""You are a textbook assistant for school students.
A student asked: "{query}"

Answer using ONLY the textbook content provided below.
Do NOT use outside knowledge. If the chunks do not contain enough information,
say so clearly instead of guessing.

---
{context}
---

You MUST respond with ONLY a JSON object. Follow these rules exactly:
- Start your response with {{ and end with }}
- Use double quotes " for all keys and string values
- Use \\n inside strings for line breaks. NEVER use triple quotes.
- Do NOT include markdown fences (no ```)
- Do NOT add any text before or after the JSON

The JSON must have EXACTLY these two keys:

"spoken_answer": A plain natural explanation in 4-7 sentences for text-to-speech.
No bullet points. No markdown symbols. No asterisks. No hashes. Write naturally.
Do NOT say "Chunk 1 says" — explain the concept directly.

"display_answer_markdown": A markdown response using \\n for newlines. Include:
- A heading with the topic name
- A clear explanation paragraph
- Key concepts as a bullet list using - prefix
- A Sources section listing chapters referenced

Example of correct format:
{{"spoken_answer": "Osmosis is the movement of water. It happens through a membrane.", "display_answer_markdown": "## Osmosis\\n\\nOsmosis is the movement of water molecules.\\n\\n- Water moves from high to low concentration\\n- A semi-permeable membrane is required\\n\\n**Sources:** Chapter 3"}}

Now respond with the JSON for the student query above:"""
)

# ── LCEL chains: prompt | llm | str_parser ─────────────────────────────────────
_reference_chain = _REFERENCE_TEMPLATE | _llm | _str_parser
_concept_chain   = _CONCEPT_TEMPLATE   | _llm | _str_parser


# ══════════════════════════════════════════════════════════════════════════════
# Context builder  (formats retrieved chunks for the prompt)
# ══════════════════════════════════════════════════════════════════════════════

def _build_context(chunks: list, answer_type: str) -> str:
    context_blocks = []
    for i, c in enumerate(chunks, 1):
        if answer_type == "reference":
            context_blocks.append(
                f"[CHUNK {i} | {c.chunk_type.upper()} | "
                f"Chapter {c.chapter_number} | {c.section_title}]\n{c.text}"
            )
        else:
            context_blocks.append(
                f"[CHUNK {i} | {c.chunk_type.upper()} | "
                f"Chapter {c.chapter_number} – {c.chapter_title} | {c.section_title}]\n{c.text}"
            )
    return "\n\n".join(context_blocks)


# ══════════════════════════════════════════════════════════════════════════════
# TTS safety cleaner  (unchanged from original)
# ══════════════════════════════════════════════════════════════════════════════

def _clean_for_tts(text: str) -> str:
    text = re.sub(r"#{1,6}\s+", "", text)
    text = re.sub(r"\*{1,3}(.+?)\*{1,3}", r"\1", text)
    text = re.sub(r"`(.+?)`", r"\1", text)
    text = re.sub(r"^\s*[-*•]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+[.)]\s+", "", text, flags=re.MULTILINE)
    text = text.replace("Fig.", "Figure")
    text = text.replace("fig.", "Figure")
    text = text.replace("e.g.", "for example")
    text = text.replace("i.e.", "that is")
    text = text.replace("vs.", "versus")
    text = text.replace("Ch.", "Chapter")
    text = re.sub(r"\[CHUNK\s*\d+\]", "", text)
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


# ══════════════════════════════════════════════════════════════════════════════
# Confidence scoring  (unchanged from original)
# ══════════════════════════════════════════════════════════════════════════════

def _compute_confidence(chunks: list, answer_type: str) -> float:
    if answer_type == "reference":
        return REFERENCE_CONFIDENCE
    scores = [c.score for c in chunks if c.score is not None]
    if not scores:
        return 0.0
    top_score  = scores[0]
    confidence = 1.0 / (1.0 + math.exp(-top_score))
    return round(confidence, 4)


# ══════════════════════════════════════════════════════════════════════════════
# JSON repair + parse  (unchanged from original — critical for small models)
# ══════════════════════════════════════════════════════════════════════════════

def _repair_json(raw: str) -> str:
    s = raw.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```\s*$", "", s)
    s = s.strip()

    def _triple_quote_to_single(m):
        inner = m.group(1)
        inner = inner.replace('\\"', '\x00DQUOTE\x00')
        inner = inner.replace('"', '\\"')
        inner = inner.replace('\x00DQUOTE\x00', '\\"')
        inner = inner.replace('\n', '\\n')
        inner = inner.replace('\r', '')
        return f'"{inner}"'

    s = re.sub(r'"""(.*?)"""', _triple_quote_to_single, s, flags=re.DOTALL)

    result  = []
    in_str  = False
    i       = 0
    while i < len(s):
        ch = s[i]
        if ch == '\\' and in_str:
            result.append(ch)
            if i + 1 < len(s):
                result.append(s[i + 1])
                i += 2
            continue
        if ch == '"':
            in_str = not in_str
            result.append(ch)
        elif ch == '\n' and in_str:
            result.append('\\n')
        elif ch == '\r' and in_str:
            pass
        else:
            result.append(ch)
        i += 1
    s = "".join(result)

    s = re.sub(r",\s*([}\]])", r"\1", s)
    return s


def _regex_fallback(raw: str) -> dict:
    result = {}
    spoken_match = re.search(
        r'"spoken_answer"\s*:\s*"(.*?)(?<!\\)"(?=\s*[,}])', raw, re.DOTALL
    )
    if spoken_match:
        result["spoken_answer"] = spoken_match.group(1).replace('\\n', '\n')

    display_match = re.search(
        r'"display_answer_markdown"\s*:\s*"(.*?)(?<!\\)"(?=\s*[,}])', raw, re.DOTALL
    )
    if display_match:
        result["display_answer_markdown"] = display_match.group(1).replace('\\n', '\n')

    if "spoken_answer" in result and "display_answer_markdown" in result:
        log.warning("JSON parse failed — recovered both fields via regex fallback.")
        return result

    raise ValueError(
        f"Could not parse LLM response as JSON and regex fallback also failed.\n"
        f"Raw response (first 400 chars):\n{raw[:400]}"
    )


def _parse_llm_json(raw: str) -> dict:
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```\s*$", "", cleaned.strip())

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    try:
        repaired = _repair_json(raw)
        return json.loads(repaired)
    except (json.JSONDecodeError, Exception):
        pass

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            repaired = _repair_json(match.group())
            return json.loads(repaired)
        except (json.JSONDecodeError, Exception):
            pass

    return _regex_fallback(raw)


# ══════════════════════════════════════════════════════════════════════════════
# Public API: Generator  (same interface as original)
# ══════════════════════════════════════════════════════════════════════════════

class Generator:
    """
    Single entry point for answer generation.
    Drop-in replacement for the original Generator — same public API.

    Internally uses LangChain LCEL chains instead of raw requests.
    """

    def generate(
        self,
        chunks:      list,
        query:       str,
        answer_type: str = "auto",
    ) -> GeneratedAnswer:
        if not chunks:
            return self._empty_answer(query)

        if answer_type == "auto":
            answer_type = self._infer_answer_type(chunks)

        log.info("Generating answer — type=%s, chunks=%d", answer_type, len(chunks))

        citations = [
            Citation(
                chunk_id        = c.chunk_id,
                chapter_number  = c.chapter_number,
                chapter_title   = c.chapter_title,
                section_title   = c.section_title,
                chunk_type      = c.chunk_type,
                activity_number = c.activity_number,
            )
            for c in chunks
        ]

        confidence = _compute_confidence(chunks, answer_type)
        warning    = None
        if confidence < CONFIDENCE_FLOOR:
            warning = (
                f"The retrieved content may not fully answer this question "
                f"(confidence: {confidence:.0%}). "
                "Consider rephrasing or checking the chapter/subject."
            )
            log.warning("Low confidence %.3f for query: %s", confidence, query)

        # ── Build context string from chunks ───────────────────────────────────
        context = _build_context(chunks, answer_type)

        # ── Invoke the appropriate LCEL chain ─────────────────────────────────
        # LangChain chain: PromptTemplate | OllamaLLM | StrOutputParser
        # Returns raw string → we parse JSON manually (same as original)
        log.info("Invoking LangChain chain …")
        if answer_type == "reference":
            raw_response = _reference_chain.invoke({"query": query, "context": context})
        else:
            raw_response = _concept_chain.invoke({"query": query, "context": context})

        log.info("LLM responded — parsing JSON …")
        parsed = _parse_llm_json(raw_response)

        spoken_raw  = parsed.get("spoken_answer", "")
        display_raw = parsed.get("display_answer_markdown", "")

        if not spoken_raw or not display_raw:
            raise ValueError(
                "LLM response missing 'spoken_answer' or 'display_answer_markdown'. "
                f"Got keys: {list(parsed.keys())}"
            )

        spoken_clean = _clean_for_tts(spoken_raw)

        return GeneratedAnswer(
            answer_type             = answer_type,
            spoken_answer           = spoken_clean,
            display_answer_markdown = display_raw,
            citations               = citations,
            confidence              = confidence,
            low_confidence_warning  = warning,
            filter_path             = chunks[0].filter_path if chunks else "",
        )

    def generate_safe(
        self,
        chunks:      list,
        query:       str,
        answer_type: str = "auto",
    ) -> tuple[Optional[GeneratedAnswer], Optional[str]]:
        try:
            return self.generate(chunks, query, answer_type), None
        except Exception as e:
            log.error("Generator error: %s", e)
            return None, str(e)

    @staticmethod
    def _infer_answer_type(chunks: list) -> str:
        for c in chunks:
            if str(c.activity_number).strip() not in ("", "None", "none"):
                return "reference"
        return "concept"

    @staticmethod
    def _empty_answer(query: str) -> GeneratedAnswer:
        return GeneratedAnswer(
            answer_type             = "concept",
            spoken_answer           = (
                "I could not find relevant content in the textbook "
                "to answer your question. Please try rephrasing or "
                "check the chapter and subject."
            ),
            display_answer_markdown = (
                "## No Results Found\n\n"
                "The retrieval system could not find relevant chunks "
                "for your query. Please check:\n"
                "- The subject and chapter are correct\n"
                "- The query is specific enough\n"
            ),
            citations               = [],
            confidence              = 0.0,
            low_confidence_warning  = "No chunks were retrieved.",
            filter_path             = "",
        )


# ── CLI smoke-test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import json as _json
    from query_parser_v2 import parse_query_with_slm
    from retriever import Retriever

    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "explain activity 2 from chapter 3 biology"
    print(f"\nQuery: {query!r}\n")

    raw_parse = parse_query_with_slm(query)
    parsed    = _json.loads(raw_parse)
    print("Parsed metadata:", _json.dumps(parsed, indent=2))

    retriever = Retriever()
    chunks, ret_err = retriever.retrieve_safe(parsed, query)

    if ret_err:
        print(f"\n⚠  Retrieval error: {ret_err}")
        sys.exit(1)

    print(f"\nRetrieved {len(chunks)} chunks.")

    generator = Generator()
    answer, gen_err = generator.generate_safe(chunks, query)

    if gen_err:
        print(f"\n⚠  Generation error: {gen_err}")
        sys.exit(1)

    print("\n" + "═" * 60)
    print(_json.dumps(answer.to_dict(), indent=2, ensure_ascii=False))
    print("═" * 60)

    print("\n── SPOKEN (TTS) ──────────────────────────────────────────")
    print(answer.spoken_answer)

    print("\n── DISPLAY (MARKDOWN) ────────────────────────────────────")
    print(answer.display_answer_markdown)
