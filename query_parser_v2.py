"""
query_parser_v2.py — LangChain Query Intent Parser
=====================================================
Migrated from raw requests → LangChain OllamaLLM + JsonOutputParser.

Parses a raw user query into structured metadata:
  intent, chunk_type, chapter_number, activity_number, exercise_number, topic

Uses qwen2.5:0.5b-instruct via Ollama (offline, no API key needed).
"""

from __future__ import annotations

import json
import logging

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

log = logging.getLogger(__name__)

# ── Model config ───────────────────────────────────────────────────────────────
OLLAMA_MODEL = "qwen2.5:0.5b-instruct"

# ── LangChain components ───────────────────────────────────────────────────────
_llm = OllamaLLM(
    model=OLLAMA_MODEL,
    temperature=0,
    format="json",
)

_parser = JsonOutputParser()

_PARSER_TEMPLATE = PromptTemplate(
    input_variables=["query"],
    template="""You are an intent parser for a school textbook Q&A system.
Output ONLY a JSON object. No explanation. No markdown. No extra text.

Extract structured information from the query.

Return ONLY valid JSON.

Schema:
{{
  "intent": "explain | solve | define | list | unknown",
  "chunk_type": "exercise | activity | theory | unknown",
  "chapter_number": integer or null,
  "chapter_name": string or null,
  "activity_number": integer or null,
  "exercise_number" : integer or null,
  "topic": short phrase
}}

Examples:

Query: explain the activity 2 from chapter 1
Output:
{{
  "intent": "explain",
  "chunk_type": "activity",
  "chapter_number": 1,
  "chapter_name": null,
  "activity_number" : 2,
  "topic": "Activity 2"
}}

Query: solve exercise 3.1 chapter 4
Output:
{{
  "intent": "solve",
  "chunk_type": "exercise",
  "chapter_number": 4,
  "chapter_name": null,
  "exercise_number" : 3.1,
  "topic": "Exercise 3.1"
}}

Now extract from:

Query: "{query}"
Output:"""
)

# ── Build LCEL chain: prompt | llm | json_parser ───────────────────────────────
_parse_chain = _PARSER_TEMPLATE | _llm | _parser


def parse_query_with_slm(query: str) -> str:
    """
    Parse a raw user query and return a JSON string with extracted metadata.

    Compatible with original API — returns a JSON string so callers can
    json.loads() it exactly as before.

    Parameters
    ----------
    query : raw user query string

    Returns
    -------
    JSON string with keys: intent, chunk_type, chapter_number, chapter_name,
    activity_number, exercise_number, topic
    """
    log.info("Parsing query: %r", query)
    try:
        result: dict = _parse_chain.invoke({"query": query})
        return json.dumps(result)
    except Exception as e:
        log.error("Parser error: %s", e)
        # Fallback — return a minimal valid schema so pipeline doesn't crash
        fallback = {
            "intent": "unknown",
            "chunk_type": "unknown",
            "chapter_number": None,
            "chapter_name": None,
            "activity_number": None,
            "exercise_number": None,
            "topic": query,
        }
        return json.dumps(fallback)


# ── CLI smoke-test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "explain activity 2 from chapter 1"
    print(f"\nQuery: {q!r}")
    result = parse_query_with_slm(q)
    print(f"Parsed: {result}")
