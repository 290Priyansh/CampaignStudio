"""
Local LLM access layer.

Two consumers exist in this project, and they're kept deliberately separate
so the README can explain (per section 21 of the brief) exactly what each
orchestration framework is responsible for:

  * `get_crewai_llm()`   -> a `crewai.LLM` instance, used by CrewAI Agents/
                            Tasks/Crew for the top-level multi-agent workflow
                            (delegation, role/goal/backstory prompting).

  * `get_langchain_llm()` / `get_structured_llm()` / `invoke_structured()`
                          -> real LangChain usage: `ChatOllama` +
                            `with_structured_output()` bound to this
                            project's Pydantic schemas. This is what
                            replaces the original repo's regex-scraped JSON
                            with a guaranteed-shape response, and it's what
                            the Prompt Optimizer / evaluator-feedback loop
                            relies on for reliability.

Both point at the same local Ollama server -- there is exactly one model
serving process either way. Nothing here calls a paid API.

Imports of `langchain_ollama` / `crewai` are deliberately local to each
function (not at module level) so this module can be imported -- and its
error-handling paths unit-tested -- without either package installed.
"""

from __future__ import annotations

import logging
from typing import TypeVar

from pydantic import BaseModel

from config.settings import settings
from models.exceptions import OllamaUnavailableError, StructuredOutputError

logger = logging.getLogger("LLM")

T = TypeVar("T", bound=BaseModel)


def check_ollama_available(timeout: float = 3.0) -> bool:
    """Lightweight health check against Ollama's /api/tags endpoint.

    Used by the workflow/UI to fail fast with a clear message instead of
    letting the first agent call time out deep inside a Crew run.
    """
    import requests

    try:
        resp = requests.get(f"{settings.ollama_base_url}/api/tags", timeout=timeout)
        resp.raise_for_status()
        return True
    except requests.exceptions.RequestException as exc:
        logger.error("Ollama unavailable at %s: %s", settings.ollama_base_url, exc)
        return False


def get_langchain_llm(temperature: float | None = None):
    """Return a configured `langchain_ollama.ChatOllama` instance.

    Raises OllamaUnavailableError (not ImportError) if the package isn't
    installed, so callers get one predictable exception type to handle.
    """
    try:
        from langchain_ollama import ChatOllama
    except ImportError as exc:
        raise OllamaUnavailableError(
            "langchain-ollama is not installed. Run `pip install -r requirements.txt`."
        ) from exc

    return ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=temperature if temperature is not None else settings.ollama_temperature,
        timeout=settings.ollama_request_timeout,
    )


def get_crewai_llm(temperature: float | None = None):
    """Return a `crewai.LLM` instance pointed at the same local Ollama server.

    CrewAI's `Agent`/`Crew`/`Task` machinery (via litellm under the hood)
    expects this type. It is intentionally NOT the same object as
    `get_langchain_llm()` -- CrewAI drives agent role-play and delegation,
    LangChain drives guaranteed-schema extraction. See README 'Tech Stack'.
    """
    try:
        from crewai import LLM
    except ImportError as exc:
        raise OllamaUnavailableError(
            "crewai is not installed. Run `pip install -r requirements.txt`."
        ) from exc

    return LLM(
        model=f"ollama/{settings.ollama_model}",
        base_url=settings.ollama_base_url,
        temperature=temperature if temperature is not None else settings.ollama_temperature,
    )


def get_structured_llm(schema: type[T], temperature: float | None = None):
    """Return a LangChain runnable that invokes the local LLM and parses its
    response directly into `schema`, via `with_structured_output`.

    This is the concrete mechanism that satisfies section 12/21 of the
    brief: agents return validated Pydantic objects, not raw strings that
    get regex-scraped for a ```json fence (the original repo's approach).
    """
    llm = get_langchain_llm(temperature=temperature)
    return llm.with_structured_output(schema)


def invoke_structured(schema: type[T], prompt: str, *, retries: int = 2) -> T:
    """Invoke a structured-output call against `schema`, retrying on
    malformed/unparseable responses (local models are more prone to this
    than hosted frontier models, so this matters in practice).

    Raises StructuredOutputError, chained from the last underlying
    exception, if every attempt fails -- callers should not need to know
    whether the failure was a connection error, a validation error, or a
    malformed-JSON error underneath.
    """
    structured_llm = get_structured_llm(schema)
    last_exc: Exception | None = None

    for attempt in range(1, retries + 2):
        try:
            result = structured_llm.invoke(prompt)
            return result if isinstance(result, schema) else schema.model_validate(result)
        except Exception as exc:  # noqa: BLE001 - deliberately broad, logged + re-raised typed
            last_exc = exc
            logger.warning(
                "Structured output attempt %d/%d for %s failed: %s",
                attempt, retries + 1, schema.__name__, exc,
            )

    raise StructuredOutputError(
        f"Failed to get a valid {schema.__name__} from the LLM after {retries + 1} attempts"
    ) from last_exc
