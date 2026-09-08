"""
Centralized configuration for the AI Creative Director system.

Every tunable in this project (LLM choice, image backend, quality thresholds,
retry limits, output paths) is read from environment variables via this module.
Nothing in agents/, tools/, workflows/, or evaluation/ should hard-code a
model name, URL, or path -- they should import `settings` from here.

Copy `.env.example` to `.env` and adjust values as needed.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# CrewAI sends anonymous usage telemetry to telemetry.crewai.com by default,
# which is a real external network call -- it contradicts this project's
# "fully local" premise even though it's not a paid API. Disabled here,
# before crewai is ever imported anywhere in the codebase (agents/* import
# crewai lazily, well after this module has loaded). Override by setting
# OTEL_SDK_DISABLED=false in your shell environment if you want it back on.


def _disable_crewai_telemetry_by_default() -> None:
    os.environ.setdefault("OTEL_SDK_DISABLED", "true")


_disable_crewai_telemetry_by_default()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---------------------------------------------------------------
    # LLM (Ollama)
    # ---------------------------------------------------------------
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Base URL of the local Ollama server.",
    )
    ollama_model: str = Field(
        default="qwen2.5:7b",
        description=(
            "Ollama model tag used by all reasoning agents "
            "(Creative Director, Audience Analyst, Prompt Optimizer, etc.). "
            "Any Ollama-servable chat model works: qwen2.5:7b, gemma3:4b, "
            "llama3.1:8b, mistral:7b. Smaller VRAM/RAM budgets should use "
            "gemma3:4b or qwen2.5:3b -- see README hardware section."
        ),
    )
    ollama_temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    ollama_request_timeout: int = Field(
        default=120, description="Seconds before an LLM call is considered timed out."
    )

    # ---------------------------------------------------------------
    # Image generation backend
    # ---------------------------------------------------------------
    image_backend: Literal["diffusers"] = Field(default="diffusers")

    # Diffusers backend
    image_model: str = Field(
        default="stabilityai/stable-diffusion-xl-base-1.0",
        description="HF model id used for diffusion generation.",
    )
    image_device: Literal["cuda", "cpu", "mps"] = Field(
        default="cuda",
        description=(
            "Device for diffusion inference. CPU works but is impractically "
            "slow for SDXL (minutes per image) -- see README hardware section."
        ),
    )
    image_dtype: Literal["float16", "float32", "bfloat16"] = Field(default="float16")
    image_height: int = Field(default=512, gt=0)
    image_width: int = Field(default=512, gt=0)
    image_num_inference_steps: int = Field(default=20, gt=0)
    image_guidance_scale: float = Field(default=7.5, gt=0)

    # ---------------------------------------------------------------
    # Generation / evaluation loop
    # ---------------------------------------------------------------
    quality_threshold: float = Field(
        default=0.75, ge=0.0, le=1.0,
        description="Minimum overall_score an asset must reach to pass the quality gate.",
    )
    max_generation_attempts: int = Field(
        default=2, ge=1,
        description="Hard cap on generate->evaluate->improve-prompt cycles per asset.",
    )
    candidates_per_asset: int = Field(
        default=1, ge=1,
        description="Number of candidate images generated per attempt before ranking.",
    )

    # ---------------------------------------------------------------
    # Output / paths
    # ---------------------------------------------------------------
    output_dir: str = Field(default="outputs")

    # ---------------------------------------------------------------
    # Logging
    # ---------------------------------------------------------------
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")

    @field_validator("output_dir")
    @classmethod
    def _ensure_output_dir_exists(cls, v: str) -> str:
        Path(v).mkdir(parents=True, exist_ok=True)
        return v


# Singleton settings instance imported throughout the codebase.
settings = Settings()
