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
import json
import logging
import re
from typing import TypeVar

from pydantic import BaseModel

from config.settings import settings
from models.exceptions import CrewExecutionError, OllamaUnavailableError, StructuredOutputError

logger = logging.getLogger("LLM")

T = TypeVar("T", bound=BaseModel)


def extract_and_parse_json(schema: type[T], text: str) -> T:
    """Extract and validate JSON into a Pydantic schema from any LLM text output.

    Handles direct JSON, markdown codeblocks, nested braces/brackets, and
    partial control characters.
    """
    if not text or not text.strip():
        raise StructuredOutputError(f"Cannot parse empty response into {schema.__name__}")

    cleaned = text.strip()

    # 1. Direct model validation attempt
    try:
        return schema.model_validate_json(cleaned)
    except Exception:
        pass

    # 2. Extract from markdown ```json ... ``` or ``` ... ``` code blocks
    code_block_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned, re.IGNORECASE)
    if code_block_match:
        block_content = code_block_match.group(1).strip()
        try:
            return schema.model_validate_json(block_content)
        except Exception:
            try:
                parsed_dict = json.loads(block_content)
                return schema.model_validate(parsed_dict)
            except Exception:
                cleaned = block_content

    # 3. Find outermost braces { ... }
    start_brace = cleaned.find("{")
    end_brace = cleaned.rfind("}")
    if start_brace != -1 and end_brace != -1 and end_brace > start_brace:
        json_str = cleaned[start_brace : end_brace + 1]
        try:
            return schema.model_validate_json(json_str)
        except Exception:
            try:
                parsed_dict = json.loads(json_str)
                return schema.model_validate(parsed_dict)
            except Exception:
                pass

    # 4. Try json.loads on cleaned string directly
    try:
        parsed_dict = json.loads(cleaned)
        return schema.model_validate(parsed_dict)
    except Exception as exc:
        raise StructuredOutputError(
            f"Failed to parse valid {schema.__name__} from LLM response: {exc}"
        ) from exc


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
        request_timeout=settings.ollama_request_timeout,
    )


def get_structured_llm(schema: type[T], temperature: float | None = None):
    """Return a LangChain runnable that invokes the local LLM and parses its
    response directly into `schema`, via `with_structured_output`.
    """
    llm = get_langchain_llm(temperature=temperature)
    try:
        return llm.with_structured_output(schema, method="json_mode")
    except Exception:
        return llm.with_structured_output(schema)


def invoke_structured(schema: type[T], prompt: str, *, retries: int = 2) -> T:
    """Invoke a structured-output call against `schema`, retrying on
    malformed/unparseable responses.
    """
    last_exc: Exception | None = None

    try:
        structured_llm = get_structured_llm(schema)
        for attempt in range(1, retries + 2):
            try:
                result = structured_llm.invoke(prompt)
                return result if isinstance(result, schema) else schema.model_validate(result)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                logger.warning(
                    "Structured output attempt %d/%d for %s failed: %s",
                    attempt, retries + 1, schema.__name__, exc,
                )
    except Exception as exc:
        last_exc = exc

    # Fallback to invoking ChatOllama directly and extracting JSON
    try:
        llm = get_langchain_llm()
        raw_msg = llm.invoke(prompt)
        content = raw_msg.content if hasattr(raw_msg, "content") else str(raw_msg)
        return extract_and_parse_json(schema, str(content))
    except Exception as exc:
        raise StructuredOutputError(
            f"Failed to get a valid {schema.__name__} from the LLM after {retries + 1} attempts"
        ) from (last_exc or exc)


def execute_agent_task(agent_name: str, crew, task, schema: type[T], prompt: str | None = None) -> T:
    """Execute a CrewAI task safely, handling CrewAI output or falling back on Ollama tool call/timeout errors."""
    if not prompt:
        prompt = getattr(task, "description", None) or (
            task.kwargs.get("description", "") if hasattr(task, "kwargs") else str(task)
        )

    try:
        crew.kickoff()
    except Exception as exc:
        exc_str = str(exc)
        is_parsing_or_tool_error = any(
            kw in exc_str.lower()
            for kw in ("tool", "function", "json", "validation", "modelresponse", "{", "litellm.timeout")
        )
        if not is_parsing_or_tool_error:
            raise CrewExecutionError(f"{agent_name} crew run failed: {exc}") from exc

        logger.warning(
            "CrewAI execution for agent '%s' encountered exception/timeout: %s. Attempting JSON recovery/fallback.",
            agent_name, exc_str,
        )
        try:
            return extract_and_parse_json(schema, exc_str)
        except Exception:
            pass

        try:
            return invoke_structured(schema, prompt)
        except Exception:
            raise CrewExecutionError(f"{agent_name} crew run failed: {exc}") from exc

    # 1. Try task.output.pydantic if set and valid
    pydantic_res = getattr(task.output, "pydantic", None)
    if isinstance(pydantic_res, schema):
        return pydantic_res
    if pydantic_res is not None:
        try:
            return schema.model_validate(pydantic_res)
        except Exception:
            pass

    # 2. Try task.output.raw
    raw_res = getattr(task.output, "raw", "")
    if raw_res:
        try:
            return extract_and_parse_json(schema, str(raw_res))
        except Exception:
            pass

    # 3. If no structured output was obtained
    raise CrewExecutionError(
        f"{agent_name} agent did not return a valid structured {schema.__name__}"
    )

