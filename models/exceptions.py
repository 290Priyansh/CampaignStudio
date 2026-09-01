"""
Typed exceptions for the local model layer.

Section 24 of the project brief requires explicit handling of Ollama/model/
image-backend failures rather than swallowing them with a bare
`except Exception: pass`. Every place that talks to Ollama or a diffusion
backend raises one of these, with enough context to act on.
"""

from __future__ import annotations


class CreativeDirectorError(Exception):
    """Base class for all custom errors raised by this project."""


class OllamaUnavailableError(CreativeDirectorError):
    """Ollama server is unreachable, or the required Python package/model
    is not installed/pulled."""


class StructuredOutputError(CreativeDirectorError):
    """The LLM's response could not be parsed/validated into the requested
    Pydantic schema after all retries."""


class ImageBackendUnavailableError(CreativeDirectorError):
    """The configured image-generation backend (diffusers/comfyui) is not
    reachable or not installed/configured."""


class ImageGenerationError(CreativeDirectorError):
    """A specific image-generation call failed after the backend was
    confirmed available (e.g. OOM, invalid parameters, pipeline error)."""


class CrewExecutionError(CreativeDirectorError):
    """A CrewAI agent/task/crew run failed (agent error, timeout, or the
    task did not return the expected structured (Pydantic) output)."""
