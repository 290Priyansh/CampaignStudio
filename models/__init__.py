"""
Models package for CampaignStudio.

Contains LLM access helpers and custom exception types.
"""

from __future__ import annotations

from models.exceptions import (
    CreativeDirectorError,
    CrewExecutionError,
    ImageBackendUnavailableError,
    ImageGenerationError,
    OllamaUnavailableError,
    StructuredOutputError,
)
from models.llm import check_ollama_available, execute_agent_task, get_crewai_llm, get_langchain_llm

__all__ = [
    "CreativeDirectorError",
    "CrewExecutionError",
    "ImageBackendUnavailableError",
    "ImageGenerationError",
    "OllamaUnavailableError",
    "StructuredOutputError",
    "check_ollama_available",
    "execute_agent_task",
    "get_crewai_llm",
    "get_langchain_llm",
]
