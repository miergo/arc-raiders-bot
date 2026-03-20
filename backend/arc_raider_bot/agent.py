"""Core agent pipeline: cache lookup -> main agent -> validator -> store."""

from __future__ import annotations

import json
import re
import logging

from langchain_community.tools import DuckDuckGoSearchResults
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider

from arc_raider_bot.config import OLLAMA_BASE_URL, OLLAMA_MODEL, VERBOSE
from arc_raider_bot.models import ArcRaidersResponse, ValidationVerdict
from arc_raider_bot.prompts import MAIN_AGENT_PROMPT, VALIDATOR_PROMPT
from arc_raider_bot.sessions import get_history, save_history
from arc_raider_bot.knowledge_cache import KnowledgeCache

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared LLM model
# ---------------------------------------------------------------------------

_llm = OpenAIChatModel(
    model_name=OLLAMA_MODEL,
    provider=OllamaProvider(base_url=OLLAMA_BASE_URL),
)

# ---------------------------------------------------------------------------
# Main agent + web-search tool
# ---------------------------------------------------------------------------

_ddg = DuckDuckGoSearchResults(max_results=6, output_format="list")

arc_agent = Agent(
    _llm,
    system_prompt=MAIN_AGENT_PROMPT,
    output_type=ArcRaidersResponse,
    retries=5,
)


@arc_agent.tool_plain
def websearch(query: str) -> str:
    """Search the web for up-to-date information about ARC Raiders.

    Args:
        query: The search query string. Prefer including "ARC Raiders" in
               the query for more relevant results.

    Returns:
        A JSON array of search result objects, each with 'snippet', 'title',
        and 'link' keys.
    """
    if VERBOSE:
        print(f"  [tool] websearch({query!r})")
    results = _ddg.invoke(query)
    if VERBOSE:
        print(f"  [tool] got {len(results)} results")
    return json.dumps(results, ensure_ascii=False) if isinstance(results, list) else str(results)


# ---------------------------------------------------------------------------
# Validator agent
# ---------------------------------------------------------------------------

_validator = Agent(
    _llm,
    system_prompt=VALIDATOR_PROMPT,
    output_type=ValidationVerdict,
    retries=3,
)

# ---------------------------------------------------------------------------
# Knowledge cache (RAG)
# ---------------------------------------------------------------------------

_cache = KnowledgeCache()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_URL_RE = re.compile(r'https?://[^\s\'"<>\]\)]+')


def _extract_urls(text: str) -> list[str]:
    return _URL_RE.findall(text)


def _fallback_parse(raw: str) -> ArcRaidersResponse:
    """Best-effort parse when the model returns text instead of valid JSON."""
    urls = _extract_urls(raw)
    clean = _URL_RE.sub("", raw).strip()
    for token in ('"answer":', '"sources":', "{", "}", "[", "]"):
        clean = clean.replace(token, "")
    clean = clean.strip(' ",\n')
    return ArcRaidersResponse(answer=clean or raw, sources=urls)


def _build_prompt_with_cache(question: str) -> str:
    """Search the knowledge cache and augment the user prompt with any hits."""
    try:
        hits = _cache.search(question)
    except Exception as exc:
        if VERBOSE:
            print(f"[cache] Search failed: {exc}")
        return question

    if not hits:
        if VERBOSE:
            print(f"[cache] No relevant cached Q&A (db has {_cache.count} entries)")
        return question

    if VERBOSE:
        for h in hits:
            print(f"  [cache] sim={h.similarity:.2f}  q={h.question!r}")

    lines = ["\n\n--- CACHED KNOWLEDGE (from previous answers) ---"]
    for i, hit in enumerate(hits, 1):
        sources_str = ", ".join(hit.sources) if hit.sources else "(none)"
        lines.append(
            f"\nPast Q{i} (similarity {hit.similarity:.0%}): {hit.question}\n"
            f"Past A{i}: {hit.answer}\n"
            f"Past sources{i}: {sources_str}"
        )
    lines.append("--- END CACHED KNOWLEDGE ---\n")
    return question + "\n".join(lines)


# ---------------------------------------------------------------------------
# Pipeline steps (kept small and flat)
# ---------------------------------------------------------------------------

def _run_main_agent(prompt: str, history: list) -> tuple[ArcRaidersResponse, list]:
    """Call the main agent, returning (response, updated_messages)."""
    result = arc_agent.run_sync(prompt, message_history=history)
    return result.output, result.all_messages()


def _run_validator(question: str, response: ArcRaidersResponse) -> ArcRaidersResponse:
    """Run the validator and apply corrections in-place. Never raises."""
    try:
        prompt = (
            f"User question: {question}\n\n"
            f"Proposed answer:\n{response.answer}\n\n"
            f"Proposed sources:\n{json.dumps(response.sources, indent=2)}"
        )
        verdict = _validator.run_sync(prompt).output
    except Exception as exc:
        if VERBOSE:
            print(f"[validator] Skipped: {exc}")
        return response

    if verdict.is_valid:
        if VERBOSE:
            print("[validator] PASSED")
        return response

    if VERBOSE:
        print(f"[validator] FAILED — {verdict.issues}")
    if verdict.corrected_answer:
        response.answer = verdict.corrected_answer
    if verdict.corrected_sources is not None:
        response.sources = verdict.corrected_sources
    return response


def _store_in_cache(question: str, response: ArcRaidersResponse) -> None:
    try:
        _cache.store(question, response.answer, response.sources)
        if VERBOSE:
            print(f"[cache] Stored. Total entries: {_cache.count}")
    except Exception as exc:
        if VERBOSE:
            print(f"[cache] Failed to store: {exc}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ask(question: str, session_id: str | None = None) -> tuple[ArcRaidersResponse, str]:
    """Run the full pipeline and return (response, session_id)."""
    sid, history = get_history(session_id)
    augmented_prompt = _build_prompt_with_cache(question)

    if VERBOSE:
        print(f"[pipeline] session={sid[:8]}… history={len(history)} msgs")

    try:
        response, messages = _run_main_agent(augmented_prompt, history)
        save_history(sid, messages)
    except Exception as exc:
        if VERBOSE:
            print(f"[pipeline] Structured output failed, using fallback: {exc}")
        response = _fallback_parse(str(exc))

    response = _run_validator(question, response)
    _store_in_cache(question, response)
    return response, sid
